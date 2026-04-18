"""
Hyperparameter Optimization for Adult Dataset (Classification)

Models:
1. Logistic Regression
2. AdaBoost
3. KNN
4. XGBoost

Search Method:
- RandomizedSearchCV
- 5-fold Stratified Cross-Validation

Optimization Metric:
- AUCPR (average_precision)

Input file path (default for Google Colab):
/content/drive/MyDrive/PATE-TransGAN/Adult/Adult_after/adult_train_preprocessed.csv
"""

from __future__ import annotations

import os
import sys
import subprocess
import warnings
from pprint import pformat

import pandas as pd
from scipy.stats import randint, uniform, loguniform

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore")



# User Configuration

TRAIN_PATH = "/content/drive/MyDrive/PATE-TransGAN/Adult/Adult_after/adult_train_preprocessed.csv"
TARGET_COL = "salary"

OPTIMIZE_METRIC = "average_precision"

RANDOM_STATE = 42
CV_SPLITS = 5
N_JOBS = 2
N_ITER_DEFAULT = 15
N_ITER_XGB = 20
VERBOSE = 2



# Helpers

def ensure_xgboost_installed() -> None:
    """Install xgboost in the current environment if missing."""
    try:
        import xgboost  # noqa: F401
    except ImportError:
        print("xgboost not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost"])


def load_dataset(train_path: str, target_col: str):
    """Load data and return (X, y) with encoded binary target if needed."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found at: {train_path}")

    df = pd.read_csv(train_path)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if y.dtype == "object" or str(y.dtype).startswith("category"):
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
    else:
        y = y.astype(int)

    return X, y


def build_adaboost(random_state: int) -> AdaBoostClassifier:
    """
    Build AdaBoost with depth-2 decision tree base estimator.
    Keeps compatibility with sklearn versions using either
    'estimator' (new) or 'base_estimator' (older).
    """
    base_tree = DecisionTreeClassifier(max_depth=2, random_state=random_state)
    try:
        return AdaBoostClassifier(estimator=base_tree, random_state=random_state)
    except TypeError:
        return AdaBoostClassifier(base_estimator=base_tree, random_state=random_state)


def run_hpo(X, y) -> dict:
    """Run RandomizedSearchCV for all requested models and return result dictionary."""
    ensure_xgboost_installed()
    from xgboost import XGBClassifier

    # Force AUCPR as requested for imbalanced classification quality.
    scoring = "average_precision"
    scoring_name = "AUCPR (Average Precision)"

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models_and_spaces = {
        "Logistic Regression": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    solver="liblinear",  # faster than saga on this search setup
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                )),
            ]),
            "param_distributions": {
                "model__C": loguniform(1e-3, 1e2),
                "model__penalty": ["l1", "l2"],
            },
            "n_iter": N_ITER_DEFAULT,
        },
        "AdaBoost": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", build_adaboost(RANDOM_STATE)),
            ]),
            "param_distributions": {
                "model__n_estimators": randint(50, 501),
                "model__learning_rate": loguniform(1e-3, 1.0),
            },
            "n_iter": N_ITER_DEFAULT,
        },
        "KNN": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier()),
            ]),
            "param_distributions": {
                "model__n_neighbors": [3, 5, 7, 9],
                "model__weights": ["uniform", "distance"],
            },
            # Full space size = 8 combinations. Keeping randomized API as requested.
            "n_iter": 8,
        },
        "XGBoost": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                    tree_method="hist",
                    device="cuda",
                )),
            ]),
            "param_distributions": {
                "model__n_estimators": randint(100, 601),
                "model__max_depth": randint(3, 11),
                "model__learning_rate": loguniform(1e-3, 0.3),
                "model__subsample": uniform(0.6, 0.4),
            },
            "n_iter": N_ITER_XGB,
        },
    }

    print("=" * 90)
    print("Hyperparameter Optimization Started")
    print(f"Optimization metric: {scoring_name}")
    print(
        f"CV strategy: StratifiedKFold(n_splits={CV_SPLITS}, shuffle=True, random_state={RANDOM_STATE})"
    )
    print("=" * 90)

    results = {}

    for model_name, cfg in models_and_spaces.items():
        print(f"\n>>> Optimizing: {model_name}")

        search = RandomizedSearchCV(
            estimator=cfg["pipeline"],
            param_distributions=cfg["param_distributions"],
            n_iter=cfg["n_iter"],
            scoring=scoring,
            cv=cv,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
            verbose=VERBOSE,
            refit=True,
        )

        search.fit(X, y)

        best_params = search.best_params_
        best_score = search.best_score_

        results[model_name] = {
            "best_params": best_params,
            "best_score": best_score,
        }

        print("-" * 90)
        print(f"{model_name} - BEST RESULTS")
        print(f"Best {scoring_name}: {best_score:.6f}")
        print("Best Parameters:")
        print(pformat(best_params, sort_dicts=True))
        print("-" * 90)

    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    for model_name, result in results.items():
        print(f"{model_name}: {scoring_name} = {result['best_score']:.6f}")

    best_params_by_model = {
        model_name: result["best_params"] for model_name, result in results.items()
    }
    print("\nBEST PARAMETERS DICTIONARY (copy-ready):")
    print(pformat(best_params_by_model, sort_dicts=True))
    print("=" * 90)

    return results


def main() -> None:
    X, y = load_dataset(TRAIN_PATH, TARGET_COL)
    run_hpo(X, y)


if __name__ == "__main__":
    main()
