import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd


RUN_NAMES    = [f"Run_{i}" for i in range(1, 6)]
AUC_FILE     = "Table_AUC.csv"
METRICS_FILE = "Table_Metrics.csv"

AUC_METRICS  = ["AUROC", "AUCPR"]
CORE_METRICS = ["Accuracy", "Recall", "F1-Score"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _find_model_column(df: pd.DataFrame) -> str:
    """Case-insensitive search so minor naming differences never skip a run."""
    lower_map = {col.lower().replace("_", " "): col for col in df.columns}
    for candidate in ["model name", "model", "classifier name", "classifier"]:
        if candidate in lower_map:
            return lower_map[candidate]
    raise ValueError(
        f"Could not find a model column. "
        f"Expected one of: ['Model Name', 'Model', 'Classifier']. "
        f"Found: {list(df.columns)}"
    )


def _load_csv_if_exists(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")
    return _normalize_columns(pd.read_csv(file_path))


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect_run_tables(
    base_dir: Path,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Walks Run_1 … Run_5 under base_dir and collects:
      - auc_tables     : Table_AUC.csv     (AUROC, AUCPR per model)
      - metrics_tables : Table_Metrics.csv (Accuracy, Recall, F1 per model)
    """
    auc_tables: List[pd.DataFrame]     = []
    metrics_tables: List[pd.DataFrame] = []

    for run_name in RUN_NAMES:
        run_dir = base_dir / run_name
        if not run_dir.exists():
            print(f"[SKIP] Missing folder: {run_dir}")
            continue

        auc_path     = run_dir / AUC_FILE
        metrics_path = run_dir / METRICS_FILE

        try:
            auc_df = _load_csv_if_exists(auc_path)
            auc_df["_Run"] = run_name
            auc_tables.append(auc_df)
            print(f"[OK]   Loaded {auc_path}")
        except Exception as exc:
            print(f"[SKIP] {run_name} - AUC table issue: {exc}")

        try:
            metrics_df = _load_csv_if_exists(metrics_path)
            metrics_df["_Run"] = run_name
            metrics_tables.append(metrics_df)
            print(f"[OK]   Loaded {metrics_path}")
        except Exception as exc:
            print(f"[SKIP] {run_name} - Metrics table issue: {exc}")

    return auc_tables, metrics_tables


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_table(
    tables: List[pd.DataFrame],
    metric_columns: List[str],
    label: str = "",
) -> pd.DataFrame:
    """
    Aggregates per-run DataFrames into mean ± std per model.

    Uses ddof=0 (population std) so a model appearing in only one run
    gets std=0.000 instead of NaN.
    """
    if not tables:
        print(f"[WARN] No tables to aggregate{' for ' + label if label else ''}.")
        return pd.DataFrame()

    combined = pd.concat(tables, ignore_index=True, sort=False)
    combined = _normalize_columns(combined)

    try:
        model_col = _find_model_column(combined)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return pd.DataFrame()

    available_metrics = [col for col in metric_columns if col in combined.columns]
    missing           = [col for col in metric_columns if col not in combined.columns]

    if missing:
        print(f"[WARN] Metrics not found in {label or 'table'}: {missing}")
    if not available_metrics:
        print(f"[WARN] None of the requested metrics found. Available: {list(combined.columns)}")
        return pd.DataFrame()

    grouped = combined.groupby(model_col, dropna=False)[available_metrics]
    mean_df = grouped.mean(numeric_only=True)
    std_df  = grouped.std(ddof=0, numeric_only=True).fillna(0.0)   # ddof=0 fix

    formatted = pd.DataFrame(index=mean_df.index)
    for metric in available_metrics:
        formatted[metric] = [
            f"{mean_df.loc[idx, metric]:.4f} ± {std_df.loc[idx, metric]:.4f}"
            for idx in mean_df.index
        ]

    formatted = formatted.reset_index()
    return formatted


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results from Run_1 to Run_5."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/content/drive/MyDrive/PATE-GAN",
        help=(
            "Base directory that contains Run_1 … Run_5 folders. "
            "Default: /content/drive/MyDrive/DP-GAN"
        ),
    )
    # parse_known_args safely ignores Jupyter/Colab kernel extra args (-f ...)
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"[INFO] Ignoring unknown arguments: {unknown_args}")

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(
            f"Base directory does not exist: {base_dir}\n"
            f"Make sure your Google Drive is mounted and the path is correct."
        )

    print(f"\n{'='*60}")
    print(f"Base directory : {base_dir}")
    print(f"Runs expected  : {RUN_NAMES}")
    print(f"{'='*60}\n")

    auc_tables, metrics_tables = collect_run_tables(base_dir)

    final_auc     = aggregate_table(auc_tables,     AUC_METRICS,  label="AUC")
    final_metrics = aggregate_table(metrics_tables, CORE_METRICS, label="Metrics")

    out_dir = base_dir / "Aggregation_Summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(df: pd.DataFrame, path: Path, label: str) -> None:
        if df.empty:
            print(f"[WARN] No aggregated {label} table generated — nothing saved.")
            return
        df.to_csv(path, index=False)
        print(f"\n[OK] Saved: {path}")
        print(f"\n--- {label} Aggregated Results ---")
        print(df.to_string(index=False))
        print()

    _save(final_auc,     out_dir / "Final_Aggregated_AUC.csv",     "AUC")
    _save(final_metrics, out_dir / "Final_Aggregated_Metrics.csv", "Metrics")

    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()