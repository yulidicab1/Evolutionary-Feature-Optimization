# vopt_core.py — Selección + síntesis (GA/ES) con one-hot, trig opcional y curva de fitness
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================================================
# Utilidades de datos
# =========================================================
def numeric_df(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Convierte (excepto target) a numérico y elimina filas con NaN en X o y."""
    out = df.copy()
    cols = [c for c in out.columns if c != target]
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out[target] = pd.to_numeric(out[target], errors="coerce")
    out = out.dropna(axis=0, subset=cols + [target]).reset_index(drop=True)
    return out


def one_hot_numeric_df(df: pd.DataFrame, target: str, max_cat_uniques: int = 20):
    """
    Devuelve:
      - df_out: DataFrame solo numérico (num bases + dummies) + target
      - meta:   dict con listas ['numeric_bases', 'dummy_cols']
    Solo one-hot para categóricas con cardinalidad <= max_cat_uniques.
    """
    df = df.copy()
    all_cols = [c for c in df.columns if c != target]

    # num/cat según pandas
    num_cols = df[all_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in all_cols if c not in num_cols]

    # elegir categóricas de baja cardinalidad
    low_card = []
    for c in cat_cols:
        nun = df[c].nunique(dropna=True)
        if nun <= max_cat_uniques:
            low_card.append(c)

    # tratar missing categóricos
    for c in low_card:
        df[c] = df[c].astype("category").cat.add_categories(["__missing__"]).fillna("__missing__")

    # dummies
    dummies = pd.get_dummies(df[low_card], drop_first=True) if low_card else pd.DataFrame(index=df.index)

    # numéricas (coerce) + target
    base = df[num_cols + [target]].copy()
    for c in num_cols + [target]:
        base[c] = pd.to_numeric(base[c], errors="coerce")
    base = base.dropna(subset=num_cols + [target])

    # alinear índices para concatenar
    dummies = dummies.loc[base.index] if not dummies.empty else dummies
    df_out = pd.concat([base.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

    meta = {
        "numeric_bases": num_cols,              # sobre estas haremos síntesis
        "dummy_cols": dummies.columns.tolist(), # estas entran como id()
    }
    return df_out, meta


def split_xy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target].copy()
    return X, y


def holdout_split(
    X: pd.DataFrame, y: pd.Series, valid_size: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=valid_size, random_state=seed)

# =========================================================
# Transformaciones (síntesis de atributos)
# =========================================================
def _identity(x: pd.Series) -> pd.Series:
    return x

def _sqrt_pos(x: pd.Series) -> pd.Series:
    z = x.copy(); z[z < 0] = np.nan
    return np.sqrt(z)

def _log1p_pos(x: pd.Series) -> pd.Series:
    z = x.copy(); z[z < 0] = np.nan
    return np.log1p(z)

def _x2(x: pd.Series) -> pd.Series:
    return x * x

def _x3(x: pd.Series) -> pd.Series:
    return x * x * x

def _recip_pos(x: pd.Series) -> pd.Series:
    z = x.copy(); z[z <= 0] = np.nan
    return 1.0 / z

# --- Trig “seguras” ---
def _norm01(arr: pd.Series) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    mn = np.nanmin(a); mx = np.nanmax(a)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(a, dtype=float)
    return (a - mn) / (mx - mn)

def _sin_norm(x: pd.Series) -> pd.Series:
    z = _norm01(x); return np.sin(2*np.pi*z)

def _cos_norm(x: pd.Series) -> pd.Series:
    z = _norm01(x); return np.cos(2*np.pi*z)

def _get_transform_catalog(enable_trig: bool) -> List[Tuple[str, callable]]:
    base = [
        ("id",   _identity),
        ("sqrt", _sqrt_pos),
        ("log1p",_log1p_pos),
        ("x2",   _x2),
        ("x3",   _x3),
        ("inv",  _recip_pos),
    ]
    if enable_trig:
        base += [("sinN", _sin_norm), ("cosN", _cos_norm)]
    return base

# =========================================================
# Espacio de features candidatas
# =========================================================
def _as_series(v, like: pd.Series) -> pd.Series:
    """Asegura Series con el mismo índice que 'like'."""
    if isinstance(v, pd.Series):
        return v
    # v puede ser np.ndarray o escalar; lo convertimos a Series alineada
    return pd.Series(v, index=like.index, dtype=float)

@dataclass
class FeatureSpace:
    names: List[str]
    builders: List[Tuple[str, str]]  # (transform_name, base_col o "c1|c2" si mul)
    matrix_train: pd.DataFrame
    matrix_valid: pd.DataFrame


def build_feature_space(
    Xtr: pd.DataFrame,
    Xva: pd.DataFrame,
    numeric_bases: List[str],
    dummy_cols: List[str],
    transforms: List[Tuple[str, callable]],
    max_num_interactions: int = 8,
) -> FeatureSpace:
    """
    - Aplica 'transforms' solo a numeric_bases (con robustez a outputs ndarray).
    - Añade dummies como id() (passthrough).
    - Interacciones: solo numérico × numérico (top por varianza).
    """
    cand_names: List[str] = []
    builders: List[Tuple[str, str]] = []
    mats_tr, mats_va = {}, {}

    # 1) Univariadas numéricas
    num_cols_present = [c for c in numeric_bases if c in Xtr.columns]
    for c in num_cols_present:
        xtr = Xtr[c]; xva = Xva[c]
        for tname, tf in transforms:
            name = f"{tname}({c})"
            try:
                vtr_raw = tf(xtr)
                vva_raw = tf(xva)
                # Asegura Series alineadas aunque el tf devuelva ndarray
                vtr = _as_series(vtr_raw, xtr)
                vva = _as_series(vva_raw, xva)
            except Exception:
                continue

            if np.mean(pd.isna(vtr)) > 0.2 or np.mean(pd.isna(vva)) > 0.2:
                continue

            vtr = vtr.fillna(vtr.median())
            vva = vva.fillna(vtr.median())  # usa mediana de train

            cand_names.append(name)
            builders.append((tname, c))
            mats_tr[name] = vtr.values
            mats_va[name] = vva.values

    # 2) Dummies (solo id)
    dummy_present = [c for c in dummy_cols if c in Xtr.columns]
    for d in dummy_present:
        name = f"id({d})"
        vtr = Xtr[d].fillna(0.0).astype(float)
        vva = Xva[d].fillna(0.0).astype(float)
        cand_names.append(name)
        builders.append(("id", d))
        mats_tr[name] = vtr.values
        mats_va[name] = vva.values

    # 3) Interacciones num × num (top por varianza)
    if num_cols_present:
        var_order = Xtr[num_cols_present].var(numeric_only=True).sort_values(ascending=False)
        top_cols = [c for c in var_order.index.tolist() if c in num_cols_present][:max_num_interactions]
        for i in range(len(top_cols)):
            for j in range(i + 1, len(top_cols)):
                c1, c2 = top_cols[i], top_cols[j]
                name = f"mul({c1},{c2})"
                vtr = Xtr[c1] * Xtr[c2]
                vva = Xva[c1] * Xva[c2]

                if np.mean(pd.isna(vtr)) > 0.2 or np.mean(pd.isna(vva)) > 0.2:
                    continue

                vtr = vtr.fillna(vtr.median())
                vva = vva.fillna(vtr.median())

                cand_names.append(name)
                builders.append(("mul", f"{c1}|{c2}"))
                mats_tr[name] = vtr.values
                mats_va[name] = vva.values

    Mtr = pd.DataFrame(mats_tr)
    Mva = pd.DataFrame(mats_va)
    return FeatureSpace(cand_names, builders, Mtr, Mva)

# =========================================================
# Evaluación (elige mejor entre Lineal y Ridge)
# =========================================================
def _fit_predict_mae_rmse(
    Xtr: pd.DataFrame, ytr: pd.Series, Xva: pd.DataFrame, yva: pd.Series
) -> Tuple[float, float]:
    # Lineal
    pipe_lin = Pipeline([("sc", StandardScaler()), ("reg", LinearRegression())])
    pipe_lin.fit(Xtr, ytr)
    p_lin = pipe_lin.predict(Xva)
    mae_lin = mean_absolute_error(yva, p_lin)
    rmse_lin = np.sqrt(mean_squared_error(yva, p_lin))

    # Ridge CV
    alphas = np.logspace(-3, 3, 13)
    pipe_ridge = Pipeline([("sc", StandardScaler()), ("reg", RidgeCV(alphas=alphas))])
    pipe_ridge.fit(Xtr, ytr)
    p_r = pipe_ridge.predict(Xva)
    mae_r = mean_absolute_error(yva, p_r)
    rmse_r = np.sqrt(mean_squared_error(yva, p_r))

    if mae_r < mae_lin:
        return mae_r, rmse_r
    else:
        return mae_lin, rmse_lin


# =========================================================
# Barra de progreso
# =========================================================
def _progress_bar(prefix: str, i: int, total: int, best_mae: float, width: int = 28, end: bool = False):
    pct = (i + 1) / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    msg = f"\r{prefix} {int(pct * 100):3d}% |{bar}| best_MAE={best_mae:.6f}"
    print(msg, end="", flush=True)
    if end:
        print("")


# =========================================================
# GA (Algoritmo genético) sobre máscaras binarias
# =========================================================
def _random_mask(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(n) < p).astype(np.uint8)

def _ensure_nonempty(mask: np.ndarray, rng: np.random.Generator):
    if mask.sum() == 0:
        mask[rng.integers(0, len(mask))] = 1

def _ga_optimize(
    fs: FeatureSpace,
    ytr: pd.Series,
    yva: pd.Series,
    pop_size: int,
    n_generations: int,
    rng: np.random.Generator,
    progress: bool,
    history: Optional[List[Tuple[int, float]]] = None,
    early_stop_patience: Optional[int] = None,
    early_stop_delta: float = 1e-4,
) -> Tuple[np.ndarray, float]:
    n = fs.matrix_train.shape[1]
    pop = np.stack([_random_mask(n, p=0.15, rng=rng) for _ in range(pop_size)], axis=0)
    for m in pop:
        _ensure_nonempty(m, rng)

    def fitness(mask: np.ndarray) -> float:
        cols = [fs.names[i] for i in range(n) if mask[i] == 1]
        Xtr = fs.matrix_train[cols]
        Xva = fs.matrix_valid[cols]
        mae, _ = _fit_predict_mae_rmse(Xtr, ytr, Xva, yva)
        return mae

    scores = np.array([fitness(m) for m in pop])
    best_idx = int(np.argmin(scores))
    best = pop[best_idx].copy()
    best_score = float(scores[best_idx])

    tourn_k, pc, pm = 3, 0.8, 0.06

    best_global = best_score
    patience_left = early_stop_patience

    for gen in range(n_generations):
        if progress:
            _progress_bar("[GA] ", gen, n_generations, best_score, end=(gen == n_generations - 1))
        if history is not None:
            history.append((gen, float(best_score)))

        def tournament():
            cand = rng.integers(0, pop.shape[0], size=tourn_k)
            j = cand[np.argmin(scores[cand])]
            return pop[j].copy()

        children = []
        while len(children) < pop.shape[0]:
            p1, p2 = tournament(), tournament()
            if rng.random() < pc:
                cx = rng.integers(1, n)
                c1 = np.concatenate([p1[:cx], p2[cx:]])
                c2 = np.concatenate([p2[:cx], p1[cx:]])
            else:
                c1, c2 = p1, p2

            for c in (c1, c2):
                flips = rng.random(n) < pm
                c[flips] ^= 1
                _ensure_nonempty(c, rng)
                children.append(c)
                if len(children) == pop.shape[0]:
                    break

        pop = np.stack(children, axis=0)
        scores = np.array([fitness(m) for m in pop])
        j = int(np.argmin(scores))
        if scores[j] < best_score:
            best_score = float(scores[j])
            best = pop[j].copy()

        # early stop por meseta
        if early_stop_patience is not None:
            if best_score + early_stop_delta < best_global:
                best_global = best_score
                patience_left = early_stop_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

    return best, best_score


# =========================================================
# ES (1+λ) simple sobre máscaras binarias
# =========================================================
def _es_optimize(
    fs: FeatureSpace,
    ytr: pd.Series,
    yva: pd.Series,
    es_generations: int,
    lam: int,
    rng: np.random.Generator,
    progress: bool,
    history: Optional[List[Tuple[int, float]]] = None,
    early_stop_patience: Optional[int] = None,
    early_stop_delta: float = 1e-4,
) -> Tuple[np.ndarray, float]:
    n = fs.matrix_train.shape[1]

    def fitness(mask: np.ndarray) -> float:
        cols = [fs.names[i] for i in range(n) if mask[i] == 1]
        Xtr = fs.matrix_train[cols]
        Xva = fs.matrix_valid[cols]
        mae, _ = _fit_predict_mae_rmse(Xtr, ytr, Xva, yva)
        return mae

    parent = _random_mask(n, p=0.2, rng=rng)
    _ensure_nonempty(parent, rng)
    parent_score = fitness(parent)

    best_global = parent_score
    patience_left = early_stop_patience

    for gen in range(es_generations):
        if progress:
            _progress_bar("[ES] ", gen, es_generations, parent_score, end=(gen == es_generations - 1))
        if history is not None:
            history.append((gen, float(parent_score)))

        children, scores = [], []
        for _ in range(lam):
            child = parent.copy()
            flips = rng.random(n) < 0.06
            child[flips] ^= 1
            _ensure_nonempty(child, rng)
            sc = fitness(child)
            children.append(child)
            scores.append(sc)

        scores = np.array([parent_score] + scores)
        idx_best = int(np.argmin(scores))
        if idx_best != 0:
            parent = children[idx_best - 1]
            parent_score = float(scores[idx_best])

        # early stop por meseta
        if early_stop_patience is not None:
            if parent_score + early_stop_delta < best_global:
                best_global = parent_score
                patience_left = early_stop_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

    return parent, parent_score


# =========================================================
# Orquestador principal
# =========================================================
def _mask_to_cols(fs: FeatureSpace, mask: np.ndarray) -> List[str]:
    return [fs.names[i] for i, b in enumerate(mask) if b == 1]

def _mask_to_synth_list(fs: FeatureSpace, mask: np.ndarray) -> List[str]:
    out = []
    for i, b in enumerate(mask):
        if not b:
            continue
        tname, base = fs.builders[i]
        if tname != "id":
            out.append(f"{tname}({base})")
    return out if out else ["id(*)"]


def optimize_dataset(
    data_path: str,
    target: str,
    seed: int = 42,
    valid_size: float = 0.2,
    optimizer: str = "ES",                 # "GA" o "ES"
    do_selection: bool = True,
    do_synthesis: bool = True,
    polynomial: bool = False,              # placeholders (no usados aquí explícitamente)
    degree: int = 2,
    ridge_alpha: Optional[float] = None,
    pop_size: int = 60,
    n_generations: int = 150,
    es_generations: int = 250,
    progress: bool = True,
    use_one_hot: bool = False,
    max_cat_uniques: int = 20,
    # --- nuevos parámetros opcionales ---
    enable_trig: bool = True,
    track_history: bool = True,
    early_stop_patience: Optional[int] = None,
    early_stop_delta: float = 1e-4,
) -> Dict:
    rng = np.random.default_rng(seed)

    # === 1) Carga y prepro ===
    df = pd.read_csv(data_path)
    if use_one_hot:
        df_prep, meta = one_hot_numeric_df(df, target, max_cat_uniques=max_cat_uniques)
        numeric_bases = meta["numeric_bases"]
        dummy_cols    = meta["dummy_cols"]
    else:
        df_prep = numeric_df(df, target)
        numeric_bases = [c for c in df_prep.columns if c != target]
        dummy_cols    = []

    X, y = split_xy(df_prep, target)
    Xtr_raw, Xva_raw, ytr_raw, yva_raw = holdout_split(X, y, valid_size=valid_size, seed=seed)

    # === 2) Baseline en RAW (Lineal vs RidgeCV) ===
    base_mae, _ = _fit_predict_mae_rmse(Xtr_raw, ytr_raw, Xva_raw, yva_raw)

    # === 3) Espacio de features (con numéricas y dummies) ===
    transforms = _get_transform_catalog(enable_trig=enable_trig)
    fs = build_feature_space(
        Xtr_raw, Xva_raw,
        numeric_bases=numeric_bases,
        dummy_cols=dummy_cols,
        transforms=transforms,
        max_num_interactions=8
    )
    n_features = fs.matrix_train.shape[1]
    if n_features == 0:
        return {
            "baseline_mae": float(base_mae),
            "optimized_mae": float(base_mae),
            "relative_improvement": 0.0,
            "synthesized": [],
            "X_train_opt": Xtr_raw.reset_index(drop=True),
            "X_valid_opt": Xva_raw.reset_index(drop=True),
            "y_train": ytr_raw.reset_index(drop=True),
            "y_valid": yva_raw.reset_index(drop=True),
            "fitness_curve": [],
        }

    # === 4) Búsqueda (GA o ES) ===
    history: List[Tuple[int, float]] = [] if track_history else []

    if optimizer.upper() == "GA":
        best_mask, _ = _ga_optimize(
            fs, ytr_raw, yva_raw,
            pop_size=pop_size,
            n_generations=n_generations,
            rng=rng,
            progress=progress,
            history=history,
            early_stop_patience=early_stop_patience,
            early_stop_delta=early_stop_delta,
        )
    else:
        best_mask, _ = _es_optimize(
            fs, ytr_raw, yva_raw,
            es_generations=es_generations,
            lam=8,
            rng=rng,
            progress=progress,
            history=history,
            early_stop_patience=early_stop_patience,
            early_stop_delta=early_stop_delta,
        )

    # === 5) Matrices optimizadas y evaluación final ===
    sel_cols = _mask_to_cols(fs, best_mask)
    Xtr_opt = fs.matrix_train[sel_cols].copy()
    Xva_opt = fs.matrix_valid[sel_cols].copy()

    opt_mae, _ = _fit_predict_mae_rmse(Xtr_opt, ytr_raw, Xva_opt, yva_raw)
    rel_impr = (base_mae - opt_mae) / base_mae if base_mae > 0 else 0.0
    synth_list = _mask_to_synth_list(fs, best_mask)

    return {
        "baseline_mae": float(base_mae),
        "optimized_mae": float(opt_mae),
        "relative_improvement": float(rel_impr),
        "synthesized": synth_list,
        "X_train_opt": Xtr_opt.reset_index(drop=True),
        "X_valid_opt": Xva_opt.reset_index(drop=True),
        "y_train": ytr_raw.reset_index(drop=True),
        "y_valid": yva_raw.reset_index(drop=True),
        "fitness_curve": history,
    }

