# Evolutionary Feature Optimization

This repository contains a compact academic Python project focused on regression with evolutionary feature optimization. The code builds transformed feature spaces from tabular datasets, searches for useful feature subsets with evolutionary strategies or a genetic algorithm, and compares the optimized pipeline against baseline regressors.

The project has been kept intentionally close to the original semester work.

## Objective

The main objective is to evaluate whether an evolutionary search process can improve regression performance by selecting and synthesizing input features before fitting a linear model.

In the current implementation, the project:

- Loads a CSV dataset for a regression task.
- Cleans numeric data and optionally applies one-hot encoding to low-cardinality categorical variables.
- Generates candidate transformed features, including polynomial-style and interaction-based terms.
- Uses an evolutionary optimizer (`ES` or `GA`) to search for promising feature subsets.
- Trains a regularized linear model on the optimized feature set.
- Compares results against `KNeighborsRegressor` and `RandomForestRegressor`.
- Saves plots and result files for later inspection.

## Repository Structure

```text
.
├── data/
│   ├── AirfoilSelfNoise.csv
│   └── California.csv
├── vopt/
│   ├── __init__.py
│   ├── run_single.py
│   └── vopt_core.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Included Datasets

The repository currently includes two tabular datasets already used by the code:

- `data/AirfoilSelfNoise.csv`: regression dataset with target column `SSPL`.
- `data/California.csv`: regression dataset with target column `MedHouseVal`.

No additional dataset claims are made here beyond what is present in the repository.

## Requirements

- Python 3.10+ is recommended.
- The project dependencies are listed in `requirements.txt`.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to Run

From the repository root, run the experiment module with one of the bundled dataset configurations.

Default run:

```bash
python3 -m vopt.run_single
```

Run with the California dataset:

```bash
python3 -m vopt.run_single --dataset california
```

Run with the genetic algorithm instead of the evolutionary strategy:

```bash
python3 -m vopt.run_single --dataset airfoil --optimizer GA
```

Run with a custom CSV file:

```bash
python3 -m vopt.run_single --data-path /absolute/path/to/dataset.csv --target target_column_name
```

Optional flags:

- `--output-dir /absolute/path/to/output_dir`
- `--use-one-hot`
- `--no-save`

## Expected Outputs

When output saving is enabled, the script writes results to `outputs_single/` by default:

- `fitness_curve.png`
- `dashboard_dispersion.png`
- timestamped JSON summary
- timestamped optimized train/validation CSV files

The script also prints a summary to the console, including baseline MAE, optimized MAE, relative improvement, and a baseline leaderboard.

## Technologies Used

- Python
- NumPy
- pandas
- scikit-learn
- Matplotlib

## Notes on Reproducibility

- A fixed random seed (`42`) is used in the current code.
- The repository includes the datasets required for the two predefined runs.
- The code is designed for single-run experimentation rather than for packaging as a full reusable library.

## Limitations

- The project is an academic experiment and still reflects that scope.
- Configuration is intentionally lightweight and only exposes a small command-line interface.
- The code currently focuses on regression datasets in CSV format.
- There is no automated test suite in the repository.
- Plots are shown interactively, which may behave differently depending on the local Python and Matplotlib environment.

## Author

Yulicenia Diaz Cabrera

Academic project from the subject of biologically inspired artificial intelligence in Master's degree in Computer science and technology at Universidad Carlos III de Madrid.
