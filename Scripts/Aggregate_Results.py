import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd


RUN_NAMES = [f"Run_{i}" for i in range(1, 6)]
AUC_FILE = "Table_AUC.csv"
METRICS_FILE = "Table_Metrics.csv"
AUC_METRICS = ["AUROC", "AUCPR"]
CORE_METRICS = ["Accuracy", "Recall", "F1-Score"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _find_model_column(df: pd.DataFrame) -> str:
    candidates = ["Model Name", "Model", "Classifier"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find a model column. Expected one of: {candidates}. Found: {list(df.columns)}"
    )


def _load_csv_if_exists(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")
    return _normalize_columns(pd.read_csv(file_path))


def collect_run_tables(base_dir: Path) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    auc_tables: List[pd.DataFrame] = []
    metrics_tables: List[pd.DataFrame] = []

    for run_name in RUN_NAMES:
        run_dir = base_dir / run_name
        if not run_dir.exists():
            print(f"[SKIP] Missing folder: {run_dir}")
            continue

        auc_path = run_dir / AUC_FILE
        metrics_path = run_dir / METRICS_FILE

        try:
            auc_df = _load_csv_if_exists(auc_path)
            auc_df["_Run"] = run_name
            auc_tables.append(auc_df)
        except Exception as exc:
            print(f"[SKIP] {run_name} - AUC table issue: {exc}")

        try:
            metrics_df = _load_csv_if_exists(metrics_path)
            metrics_df["_Run"] = run_name
            metrics_tables.append(metrics_df)
        except Exception as exc:
            print(f"[SKIP] {run_name} - Metrics table issue: {exc}")

    return auc_tables, metrics_tables


def aggregate_table(
    tables: List[pd.DataFrame],
    metric_columns: List[str],
) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame()

    combined = pd.concat(tables, ignore_index=True, sort=False)
    combined = _normalize_columns(combined)

    model_col = _find_model_column(combined)
    available_metrics = [col for col in metric_columns if col in combined.columns]

    if not available_metrics:
        print(
            f"[WARN] None of requested metrics {metric_columns} found. Available: {list(combined.columns)}"
        )
        return pd.DataFrame()

    grouped = combined.groupby(model_col, dropna=False)[available_metrics]
    mean_df = grouped.mean(numeric_only=True)
    std_df = grouped.std(numeric_only=True).fillna(0.0)

    formatted = pd.DataFrame(index=mean_df.index)
    for metric in available_metrics:
        formatted[metric] = [
            f"{mean_df.loc[idx, metric]:.3f} ± {std_df.loc[idx, metric]:.3f}"
            for idx in mean_df.index
        ]

    formatted = formatted.reset_index()
    return formatted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results from Run_1 to Run_5."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/content/drive/MyDrive/PATE-GAN",
        help="Base directory that contains Run_1 ... Run_5 folders.",
    )
    # Jupyter/Colab kernels append extra CLI args (e.g., -f <kernel.json>).
    # parse_known_args keeps normal CLI behavior while safely ignoring unknowns.
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"[INFO] Ignoring unknown arguments: {unknown_args}")

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    auc_tables, metrics_tables = collect_run_tables(base_dir)

    final_auc = aggregate_table(auc_tables, AUC_METRICS)
    final_metrics = aggregate_table(metrics_tables, CORE_METRICS)

    out_dir = base_dir / "Aggregation_Summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    auc_out_path = out_dir / "Final_Aggregated_AUC.csv"
    metrics_out_path = out_dir / "Final_Aggregated_Metrics.csv"

    if not final_auc.empty:
        final_auc.to_csv(auc_out_path, index=False)
        print(f"[OK] Saved: {auc_out_path}")
    else:
        print("[WARN] No aggregated AUC table generated.")

    if not final_metrics.empty:
        final_metrics.to_csv(metrics_out_path, index=False)
        print(f"[OK] Saved: {metrics_out_path}")
    else:
        print("[WARN] No aggregated metrics table generated.")


if __name__ == "__main__":
    main()
