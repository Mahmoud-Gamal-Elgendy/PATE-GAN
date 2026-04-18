"""Preprocessing pipeline for the UCI Adult dataset.

This script loads separate training and test files, performs Adult-specific
cleaning (missing-value markers, redundant column removal, and test-label dot
cleanup), and applies leakage-safe preprocessing by fitting all transformers on
the training set only. Techniques used include most-frequent imputation,
one-hot encoding for categorical features, MinMax scaling for numeric features,
and binary target encoding, then saving processed train/test CSV outputs.
"""

import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder


# File paths
TRAIN_PATH = r"E:\Studies\PWr (M.Eng)\Thesis Prepration\PATE-TransGAN\Datasets\Adult\Adult_before\adult.data"
TEST_PATH = r"E:\Studies\PWr (M.Eng)\Thesis Prepration\PATE-TransGAN\Datasets\Adult\Adult_before\adult.test"
OUTPUT_DIR = r"E:\Studies\PWr (M.Eng)\Thesis Prepration\PATE-TransGAN\Datasets\Adult\Adult_after"


# UCI Adult column names
COLUMN_NAMES = [
	"age",
	"workclass",
	"fnlwgt",
	"education",
	"education-num",
	"marital-status",
	"occupation",
	"relationship",
	"race",
	"sex",
	"capital-gain",
	"capital-loss",
	"hours-per-week",
	"native-country",
	"salary",
]


def main() -> None:
	# Load train/test with explicit column names from UCI schema.
	df_train = pd.read_csv(
		TRAIN_PATH,
		header=None,
		names=COLUMN_NAMES,
		na_values=" ?",
		skipinitialspace=False,
	)

	# adult.test includes a first metadata row to be skipped.
	df_test = pd.read_csv(
		TEST_PATH,
		header=None,
		names=COLUMN_NAMES,
		na_values=" ?",
		skiprows=1,
		skipinitialspace=False,
	)

	# Initial cleaning
	df_train = df_train.drop(columns=["education-num"])
	df_test = df_test.drop(columns=["education-num"])

	# Remove trailing period in test labels (<=50K. / >50K.) to match train labels.
	df_test["salary"] = df_test["salary"].str.replace(".", "", regex=False)

	# Split features/target
	X_train = df_train.drop(columns=["salary"])
	y_train = df_train["salary"]

	X_test = df_test.drop(columns=["salary"])
	y_test = df_test["salary"]

	# Identify feature types
	categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
	numeric_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()

	# Missing value imputation (fit on train only)
	cat_imputer = SimpleImputer(strategy="most_frequent")
	num_imputer = SimpleImputer(strategy="most_frequent")

	X_train_cat = cat_imputer.fit_transform(X_train[categorical_cols])
	X_test_cat = cat_imputer.transform(X_test[categorical_cols])

	X_train_num = num_imputer.fit_transform(X_train[numeric_cols])
	X_test_num = num_imputer.transform(X_test[numeric_cols])

	# Categorical encoding (fit on train only)
	ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
	X_train_cat_ohe = ohe.fit_transform(X_train_cat)
	X_test_cat_ohe = ohe.transform(X_test_cat)

	ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
	X_train_cat_df = pd.DataFrame(X_train_cat_ohe, columns=ohe_feature_names, index=X_train.index)
	X_test_cat_df = pd.DataFrame(X_test_cat_ohe, columns=ohe_feature_names, index=X_test.index)

	# Ensure column names are strings
	X_train_cat_df.columns = X_train_cat_df.columns.astype(str)
	X_test_cat_df.columns = X_test_cat_df.columns.astype(str)

	# Numeric scaling to [0, 1] (fit on train only)
	scaler = MinMaxScaler()
	X_train_num_scaled = scaler.fit_transform(X_train_num)
	X_test_num_scaled = scaler.transform(X_test_num)

	X_train_num_df = pd.DataFrame(X_train_num_scaled, columns=numeric_cols, index=X_train.index)
	X_test_num_df = pd.DataFrame(X_test_num_scaled, columns=numeric_cols, index=X_test.index)

	# Combine numeric + encoded categorical features
	X_train_processed = pd.concat([X_train_num_df, X_train_cat_df], axis=1)
	X_test_processed = pd.concat([X_test_num_df, X_test_cat_df], axis=1)

	# Target encoding (fit on train only)
	target_encoder = LabelEncoder()
	y_train_encoded = target_encoder.fit_transform(y_train)
	y_test_encoded = target_encoder.transform(y_test)

	train_processed = X_train_processed.copy()
	test_processed = X_test_processed.copy()
	train_processed["salary"] = y_train_encoded
	test_processed["salary"] = y_test_encoded

	os.makedirs(OUTPUT_DIR, exist_ok=True)
	train_out = os.path.join(OUTPUT_DIR, "adult_train_preprocessed.csv")
	test_out = os.path.join(OUTPUT_DIR, "adult_test_preprocessed.csv")

	train_processed.to_csv(train_out, index=False)
	test_processed.to_csv(test_out, index=False)

	print(f"Saved preprocessed training data to: {train_out}")
	print(f"Saved preprocessed test data to: {test_out}")


if __name__ == "__main__":
	main()
