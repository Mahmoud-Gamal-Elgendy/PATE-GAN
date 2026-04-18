
"""Utility evaluation pipeline for the preprocessed Adult dataset.

This script trains and compares four binary classifiers (Logistic Regression,
AdaBoost, KNN, and XGBoost), uses 10-fold Stratified
Cross-Validation on the training set, and evaluates final performance on a
held-out test set using Accuracy, Recall, F1-Score, AUROC, and AUCPR.
It exports summary tables and confusion matrices to the results directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	confusion_matrix,
	f1_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


TEST_PATH = Path(
	r"/content/drive/MyDrive/PATE-TransGAN/Adult/Adult_after/adult_test_preprocessed.csv"
)
BASE_RUN_DIR = Path(r"/content/drive/MyDrive/PATE-GAN")
RUN_IDS = [1, 2, 3, 4, 5]
TARGET_COLUMN = "salary"


def load_data(train_path: Path, test_path: Path, target_column: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
	"""Load train/test CSVs and split features/target."""
	if not train_path.exists():
		raise FileNotFoundError(f"Training file not found: {train_path}")
	if not test_path.exists():
		raise FileNotFoundError(f"Test file not found: {test_path}")

	train_df = pd.read_csv(train_path)
	test_df = pd.read_csv(test_path)

	if target_column not in train_df.columns:
		raise KeyError(f"Target column '{target_column}' is missing from training data.")
	if target_column not in test_df.columns:
		raise KeyError(f"Target column '{target_column}' is missing from test data.")

	x_train = train_df.drop(columns=[target_column])
	y_train = train_df[target_column].copy()
	x_test = test_df.drop(columns=[target_column])
	y_test = test_df[target_column].copy()

	return x_train, y_train, x_test, y_test


def encode_target(y_train: pd.Series, y_test: pd.Series) -> Tuple[pd.Series, pd.Series, LabelEncoder]:
	"""Encode target labels so metrics are computed consistently."""
	label_encoder = LabelEncoder()
	y_train_encoded = pd.Series(label_encoder.fit_transform(y_train), index=y_train.index)

	unseen_labels = set(y_test.unique()) - set(label_encoder.classes_)
	if unseen_labels:
		raise ValueError(
			f"Test target contains unseen labels not present in training set: {sorted(unseen_labels)}"
		)

	y_test_encoded = pd.Series(label_encoder.transform(y_test), index=y_test.index)

	if len(label_encoder.classes_) != 2:
		raise ValueError(
			f"Expected a binary target for requested metrics, but found {len(label_encoder.classes_)} classes."
		)

	return y_train_encoded, y_test_encoded, label_encoder


def build_models() -> Dict[str, Pipeline]:
	"""Create all requested classification models."""
	scaler = ColumnTransformer(
		transformers=[("num", StandardScaler(), slice(0, None))],
		remainder="drop",
	)

	models: Dict[str, Pipeline] = {
		"Logistic Regression": Pipeline(
			steps=[
				("scaler", scaler),
				(
					"classifier",
					LogisticRegression(
						C=0.0745934328572655,
						penalty="l1",
						solver="liblinear",
						max_iter=1000,
						random_state=42,
					),
				),
			]
		),
		"AdaBoost": Pipeline(
			steps=[
				(
					"classifier",
					AdaBoostClassifier(
						estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
						n_estimators=343,
						learning_rate=0.31428808908401085,
						random_state=42,
					),
				),
			]
		),
		"KNN": Pipeline(
			steps=[
				("scaler", scaler),
				("classifier", KNeighborsClassifier(n_neighbors=9, weights="distance")),
			]
		),
		"XGBoost": Pipeline(
			steps=[
				(
					"classifier",
					XGBClassifier(
						objective="binary:logistic",
						eval_metric="logloss",
						n_estimators=363,
						learning_rate=0.04926364988526881,
						max_depth=6,
						subsample=0.6137554084460873,
						colsample_bytree=0.9,
						random_state=42,
						n_jobs=-1,
						tree_method="hist",
						device="cuda",
					),
				)
			]
		),
	}

	return models


def evaluate_models(
	models: Dict[str, Pipeline],
	x_train: pd.DataFrame,
	y_train: pd.Series,
	x_test: pd.DataFrame,
	y_test: pd.Series,
):
	"""Run CV on train data, then fit on full train and evaluate on test data."""
	cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
	cv_scoring = {
		"accuracy": "accuracy",
		"recall": "recall",
		"f1": "f1",
		"roc_auc": "roc_auc",
		"auc_pr": "average_precision",
	}

	cv_rows = []
	auc_rows = []
	metrics_rows = []
	confusion_details = []

	for model_name, model in models.items():
		print(f"Training and Evaluating: {model_name}...")
		cv_results = cross_validate(
			model,
			x_train,
			y_train,
			cv=cv,
			scoring=cv_scoring,
			n_jobs=-1,
			error_score="raise",
		)

		cv_row = {
			"Model Name": model_name,
			"CV Accuracy": cv_results["test_accuracy"].mean(),
			"CV Recall": cv_results["test_recall"].mean(),
			"CV F1-Score": cv_results["test_f1"].mean(),
			"CV AUROC": cv_results["test_roc_auc"].mean(),
			"CV AUCPR": cv_results["test_auc_pr"].mean(),
		}
		cv_rows.append(cv_row)

		model.fit(x_train, y_train)
		y_pred = model.predict(x_test)

		if not hasattr(model, "predict_proba"):
			raise AttributeError(f"Model '{model_name}' does not support predict_proba, required for AUROC/AUCPR.")
		y_proba = model.predict_proba(x_test)[:, 1]

		test_accuracy = accuracy_score(y_test, y_pred)
		test_recall = recall_score(y_test, y_pred)
		test_f1 = f1_score(y_test, y_pred)
		test_auroc = roc_auc_score(y_test, y_proba)
		test_aucpr = average_precision_score(y_test, y_proba)
		cm = confusion_matrix(y_test, y_pred)

		auc_rows.append(
			{
				"Model Name": model_name,
				"AUCPR": test_aucpr,
				"AUROC": test_auroc,
			}
		)

		metrics_rows.append(
			{
				"Model Name": model_name,
				"Accuracy": test_accuracy,
				"Recall": test_recall,
				"F1-Score": test_f1,
			}
		)

		confusion_details.append(
			"\n".join(
				[
					f"Model: {model_name}",
					"Confusion Matrix (rows=true class [0,1], cols=predicted class [0,1]):",
					str(cm),
					f"TN={cm[0, 0]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TP={cm[1, 1]}",
					"",
				]
			)
		)

	return (
		pd.DataFrame(cv_rows),
		pd.DataFrame(auc_rows),
		pd.DataFrame(metrics_rows),
		"\n".join(confusion_details),
	)


def process_single_run(run_id: int) -> bool:
	"""Process one run directory and return True when completed successfully."""
	run_dir = BASE_RUN_DIR / f"Run_{run_id}"
	train_path = run_dir / "synthetic_data.csv"
	output_dir = run_dir

	if not run_dir.exists() or not train_path.exists():
		print(
			f"Warning: Skipping Run_{run_id} because required data is missing. "
			f"Expected file: {train_path}"
		)
		return False

	output_dir.mkdir(parents=True, exist_ok=True)
	print(f"\n========== Processing Run_{run_id} ==========")

	try:
		x_train, y_train, x_test, y_test = load_data(train_path, TEST_PATH, TARGET_COLUMN)

		# Keep test feature columns aligned with training columns.
		x_test = x_test.reindex(columns=x_train.columns)
		if x_test.isnull().any().any():
			missing_cols = [col for col in x_test.columns if x_test[col].isnull().all()]
			raise ValueError(
				"Test feature set is missing one or more columns found in training data. "
				f"Missing columns: {missing_cols}"
			)

		y_train_encoded, y_test_encoded, label_encoder = encode_target(y_train, y_test)
		print(f"Encoded target classes: {list(label_encoder.classes_)} -> [0, 1]")

		models = build_models()
		cv_df, table_auc_df, table_metrics_df, confusion_text = evaluate_models(
			models,
			x_train,
			y_train_encoded,
			x_test,
			y_test_encoded,
		)

		table_auc_path = output_dir / "Table_AUC.csv"
		table_metrics_path = output_dir / "Table_Metrics.csv"
		confusion_path = output_dir / "Confusion_Matrices.txt"

		table_auc_df.to_csv(table_auc_path, index=False)
		table_metrics_df.to_csv(table_metrics_path, index=False)
		confusion_path.write_text(confusion_text, encoding="utf-8")

		print("\n=== 10-Fold Stratified CV (Training Set, Mean Scores) ===")
		print(cv_df.to_string(index=False))
		print("\n=== Test Set Evaluation Saved ===")
		print(f"AUC table: {table_auc_path}")
		print(f"Metrics table: {table_metrics_path}")
		print(f"Confusion matrices: {confusion_path}")

		return True

	except FileNotFoundError as exc:
		print(f"Warning: Skipping Run_{run_id}. {exc}")
		return False

	except Exception as exc:
		print(f"Warning: Run_{run_id} failed with error: {exc}")
		return False


def main() -> None:
	processed_runs: List[int] = []
	skipped_runs: List[int] = []

	for run_id in RUN_IDS:
		if process_single_run(run_id):
			processed_runs.append(run_id)
		else:
			skipped_runs.append(run_id)

	print("\n========== Global Summary ==========")
	print(f"Successfully processed runs: {processed_runs if processed_runs else 'None'}")
	print(f"Skipped/failed runs: {skipped_runs if skipped_runs else 'None'}")


if __name__ == "__main__":
	main()
