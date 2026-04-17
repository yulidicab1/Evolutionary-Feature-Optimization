import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .vopt_core import holdout_split, optimize_dataset
except ImportError:
    from vopt_core import holdout_split, optimize_dataset


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs_single"
DATASET_CONFIGS = {
    "airfoil": {
        "data_path": DATA_DIR / "AirfoilSelfNoise.csv",
        "target": "SSPL",
        "use_one_hot": False,
    },
    "california": {
        "data_path": DATA_DIR / "California.csv",
        "target": "MedHouseVal",
        "use_one_hot": False,
    },
}


plt.ion()
plt.close("all")


def savefig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=140, bbox_inches="tight")


def plot_fitness_curve(res, baseline_mae, title, save_path=None):
    curve = res.get("fitness_curve", [])
    if not curve:
        return
    gens = [g for g, _ in curve]
    maes = [float(m) for _, m in curve]
    plt.figure(figsize=(7, 4))
    plt.plot(gens, maes, marker="o", linewidth=1.5)
    if np.isfinite(baseline_mae):
        plt.axhline(float(baseline_mae), linestyle="--", label="MAE base")
        plt.legend()
    plt.title(title)
    plt.xlabel("Generacion")
    plt.ylabel("MAE (menor es mejor)")
    plt.tight_layout()
    if save_path:
        savefig(save_path)
    plt.show(block=False)
    plt.pause(0.001)


SEED = 42
VALID_SIZE = 0.2
MAX_CAT_UNIQUES = 30
OPTIMIZER = "ES"
GA_POP = 80
GA_NGEN = 600
ES_NGEN = 300
SAVE_OUTPUTS = True
SHOW_INDIVIDUALS = False
POINT_SIZE = 16
POINT_ALPHA = 0.6


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the evolutionary feature-optimization experiment on one dataset."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS.keys()),
        default="airfoil",
        help="Named dataset configuration bundled in the repository.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Optional custom CSV path. Overrides the selected named dataset.",
    )
    parser.add_argument(
        "--target",
        help="Target column to predict. Required when --data-path is used.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["GA", "ES"],
        default=OPTIMIZER,
        help="Evolutionary optimizer to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures and CSV/JSON outputs will be stored.",
    )
    parser.add_argument(
        "--use-one-hot",
        action="store_true",
        help="Enable one-hot encoding for low-cardinality categorical columns.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving JSON, CSV, and figure outputs.",
    )
    return parser.parse_args()


def resolve_config(args):
    config = DATASET_CONFIGS[args.dataset].copy()
    if args.data_path is not None:
        if not args.target:
            raise ValueError("--target is required when --data-path is provided.")
        config["data_path"] = args.data_path.expanduser().resolve()
        config["target"] = args.target
        config["use_one_hot"] = args.use_one_hot
    elif args.use_one_hot:
        config["use_one_hot"] = True

    config["output_dir"] = args.output_dir.expanduser().resolve()
    config["save_outputs"] = not args.no_save
    config["optimizer"] = args.optimizer
    return config


def get_baseline_predictions(Xtr_raw, ytr_raw, Xva_raw, yva_raw):
    models = {
        "K vecinos (k=7)": Pipeline(
            [("sc", StandardScaler()), ("knn", KNeighborsRegressor(n_neighbors=7))]
        ),
        "Bosque aleatorio": RandomForestRegressor(
            n_estimators=400, max_depth=None, random_state=0
        ),
    }
    preds = {}
    rows = []
    for name, mdl in models.items():
        mdl.fit(Xtr_raw, ytr_raw)
        p = mdl.predict(Xva_raw)
        preds[name] = p
        mae_val = mean_absolute_error(yva_raw, p)
        rmse_val = np.sqrt(mean_squared_error(yva_raw, p))
        r2_val = r2_score(yva_raw, p)
        rows.append((name, mae_val, rmse_val, r2_val))
    df_lb = (
        pd.DataFrame(rows, columns=["Modelo", "MAE", "RMSE", "R2"])
        .sort_values("RMSE")
        .reset_index(drop=True)
    )
    return preds, df_lb


def plot_dashboard(y_true, preds_dict, df_lb, title, save_path=None):
    my_long = "Regresion (RidgeCV) + v(X) (Optimizado evolutivo)"
    short_map = {
        my_long: "Ridge+v(X)",
        "K vecinos (k=7)": "KNN-7",
        "Bosque aleatorio": "RandomForest",
    }
    models_to_show = [my_long, "K vecinos (k=7)", "Bosque aleatorio"]

    a, b = float(np.min(y_true)), float(np.max(y_true))

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1, 1])

    ax0 = fig.add_subplot(gs[:, 0])
    df_bar = df_lb.copy()
    rmse_meta = preds_dict.get("_meta_rmse", None)
    mae_meta = preds_dict.get("_meta_mae", None)
    r2_meta = preds_dict.get("_meta_r2", None)
    if (df_bar["Modelo"] == my_long).sum() == 0:
        df_bar = pd.concat(
            [
                df_bar,
                pd.DataFrame(
                    [{"Modelo": my_long, "MAE": mae_meta, "RMSE": rmse_meta, "R2": r2_meta}]
                ),
            ],
            ignore_index=True,
        )

    df_bar = df_bar.sort_values("RMSE", na_position="last").reset_index(drop=True)
    labels_long = df_bar["Modelo"].tolist()
    labels_short = [short_map.get(m, m) for m in labels_long]
    values = df_bar["RMSE"].astype(float).tolist()

    palette = matplotlib.colormaps.get_cmap("tab10")
    colors = [palette(i % 10) for i in range(len(values))]

    bars = ax0.bar(labels_short, values, color=colors)
    ax0.set_title("RMSE por modelo")
    ax0.set_ylabel("RMSE")
    ax0.tick_params(axis="x", rotation=15)
    for rect, value in zip(bars, values):
        ax0.annotate(
            f"{value:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 1]),
    ]
    for ax, name in zip(axes, models_to_show):
        if name not in preds_dict:
            ax.axis("off")
            ax.set_title(f"{name} (sin pred)")
            continue
        y_pred = preds_dict[name]
        ax.scatter(y_true, y_pred, s=POINT_SIZE, alpha=POINT_ALPHA)
        ax.plot([a, b], [a, b])
        ax.set_xlim(a, b)
        ax.set_ylim(a, b)
        ax.set_title(name)
        ax.set_xlabel("y (verdad)")
        ax.set_ylabel("y (pred)")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if save_path:
        savefig(save_path)
    plt.show(block=False)
    plt.pause(0.001)


def run_once(config):
    data_path = Path(config["data_path"]).expanduser().resolve()
    target = config["target"]
    use_one_hot = config["use_one_hot"]
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    save_outputs = config["save_outputs"]
    optimizer = config["optimizer"]

    try:
        df = pd.read_csv(data_path, engine="python")
    except Exception:
        df = pd.read_csv(data_path, sep=";", engine="python")

    df.columns = df.columns.astype(str).str.strip()

    print("Shape(raw):", df.shape)
    print("Cols:", list(df.columns)[:40])
    if target not in df.columns:
        raise ValueError(f"TARGET '{target}' no esta en las columnas del CSV.")

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])
    print("Rows tras limpiar target:", len(df))

    def to_numeric_one_hot(df_in, target_name, max_uniques=30):
        df_in = df_in.copy()
        num_cols = df_in.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in df_in.columns if c not in num_cols and c != target_name]
        few = [c for c in cat_cols if df_in[c].nunique(dropna=True) <= max_uniques]
        df_ohe = pd.get_dummies(df_in, columns=few, drop_first=True, dummy_na=False)
        obj_left = [c for c in df_ohe.columns if df_ohe[c].dtype == "O" and c != target_name]
        df_ohe = df_ohe.drop(columns=obj_left, errors="ignore")
        return df_ohe

    if use_one_hot:
        df_num = to_numeric_one_hot(df, target, max_uniques=MAX_CAT_UNIQUES)
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        keep = [c for c in num_cols if c != target] + [target]
        df_num = df[keep].copy()

    df_num = df_num.dropna(axis=0)

    X = df_num.drop(columns=[target]).values
    y = df_num[target].values
    if X.shape[0] == 0:
        raise ValueError("Tras la preparacion los datos quedaron vacios (X.shape[0]==0).")
    Xtr_raw, Xva_raw, ytr_raw, yva_raw = holdout_split(
        X, y, valid_size=VALID_SIZE, seed=SEED
    )

    print(f"Split OK -> {X.shape} {y.shape}")
    print("Voy a lanzar el optimizador...")

    res = optimize_dataset(
        data_path=str(data_path),
        target=target,
        seed=SEED,
        valid_size=VALID_SIZE,
        optimizer=optimizer,
        do_selection=True,
        do_synthesis=True,
        polynomial=True,
        degree=2,
        ridge_alpha=1.0,
        pop_size=GA_POP,
        n_generations=GA_NGEN,
        es_generations=ES_NGEN,
        use_one_hot=use_one_hot,
        max_cat_uniques=MAX_CAT_UNIQUES,
        enable_trig=True,
        track_history=True,
        early_stop_patience=30,
        early_stop_delta=1e-4,
    )

    baseline_mae = float(res.get("baseline_mae", np.nan))
    plot_fitness_curve(
        res,
        baseline_mae,
        title=f"Curva de fitness - {data_path.name} - OPT={optimizer}",
        save_path=str(output_dir / "fitness_curve.png") if save_outputs else None,
    )

    base_preds, df_lb = get_baseline_predictions(Xtr_raw, ytr_raw, Xva_raw, yva_raw)

    ridge_vx = Pipeline(
        [("sc", StandardScaler()), ("reg", RidgeCV(alphas=np.logspace(-3, 3, 13)))]
    )
    ridge_vx.fit(res["X_train_opt"], res["y_train"])
    pred_vx = ridge_vx.predict(res["X_valid_opt"])

    mae_vx = float(mean_absolute_error(res["y_valid"], pred_vx))
    rmse_vx = float(np.sqrt(mean_squared_error(res["y_valid"], pred_vx)))
    r2_vx = float(r2_score(res["y_valid"], pred_vx))

    print(f"\n=== Resultados - {data_path.name} - OPT={optimizer} ===")
    print(f"Seed: {SEED} | Valid size: {VALID_SIZE}")
    print(f"MAE base:       {res['baseline_mae']:.6f}")
    print(f"MAE optimizado: {res['optimized_mae']:.6f}")
    print(f"Mejora relativa: {100 * res['relative_improvement']:.2f}%")

    print("\n=== Leaderboard (RMSE asc, comparadores) ===")
    print(df_lb.to_string(index=False))

    print(
        "\nRegresion (RidgeCV) + v(X) (Optimizado evolutivo)"
        f"  ->  MAE: {mae_vx:.3f} | RMSE: {rmse_vx:.3f} | R2: {r2_vx:.3f}"
    )

    preds_all = {
        "Regresion (RidgeCV) + v(X) (Optimizado evolutivo)": pred_vx,
        "_meta_mae": mae_vx,
        "_meta_rmse": rmse_vx,
        "_meta_r2": r2_vx,
    }
    preds_all.update(base_preds)

    dash_title = f"Dashboard - {data_path.name} - OPT={optimizer}"
    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
    plot_dashboard(
        y_true=yva_raw,
        preds_dict=preds_all,
        df_lb=df_lb,
        title=dash_title,
        save_path=str(output_dir / "dashboard_dispersion.png") if save_outputs else None,
    )

    if save_outputs:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = {
            "dataset": str(data_path),
            "target": target,
            "seed": SEED,
            "valid_size": VALID_SIZE,
            "optimizer": optimizer,
            "ga_pop": GA_POP,
            "ga_ngen": GA_NGEN,
            "es_ngen": ES_NGEN,
            "baseline_mae": res["baseline_mae"],
            "optimized_mae": res["optimized_mae"],
            "relative_improvement": res["relative_improvement"],
            "synthesized": res.get("synthesized", None),
            "ridge_vx_mae": mae_vx,
            "ridge_vx_rmse": rmse_vx,
            "ridge_vx_r2": r2_vx,
            "leaderboard_comparators": df_lb.to_dict(orient="records"),
            "fitness_curve": res.get("fitness_curve", None),
        }
        with open(output_dir / f"single_results_{ts}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        res["X_train_opt"].to_csv(output_dir / f"X_train_opt_{ts}.csv", index=False)
        res["X_valid_opt"].to_csv(output_dir / f"X_valid_opt_{ts}.csv", index=False)
        res["y_train"].to_csv(output_dir / f"y_train_{ts}.csv", index=False)
        res["y_valid"].to_csv(output_dir / f"y_valid_{ts}.csv", index=False)
        print(f"\nGuardado en: {output_dir}")

    plt.ioff()
    plt.show()


def main():
    args = parse_args()
    config = resolve_config(args)
    run_once(config)


if __name__ == "__main__":
    main()
