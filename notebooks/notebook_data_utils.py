import os
import pickle

import numpy as np
import pandas as pd


def load_dataframe_with_summary(csv_path, features_from_columns=True, head_rows=5):
    """Load a CSV file and print a standard summary used across notebooks."""
    df = pd.read_csv(csv_path)

    print("\nData loaded successfully")
    print(f"Shape: {df.shape}")
    if features_from_columns and "label" in df.columns:
        print(f"Features: {df.shape[1] - 1}")
    print(f"\nColumns ({len(df.columns)}):")
    print(df.columns.tolist())
    print(f"\nFirst {head_rows} rows:")
    print(df.head(head_rows))

    return df


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_prepared_datasets(
    ml_ready_dir="../data/processed/ml_ready",
    label_encoder_path="../models/label_encoder.pkl",
    as_dataframes=False,
):
    """Load train/test arrays with feature names and label encoder."""
    x_train_path = os.path.join(ml_ready_dir, "X_train_scaled.npy")
    x_test_path = os.path.join(ml_ready_dir, "X_test_scaled.npy")
    y_train_path = os.path.join(ml_ready_dir, "y_train.npy")
    y_test_path = os.path.join(ml_ready_dir, "y_test.npy")
    feature_names_path = os.path.join(ml_ready_dir, "feature_names.pkl")

    x_train = np.load(x_train_path)
    x_test = np.load(x_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)

    feature_names = _load_pickle(feature_names_path)
    label_encoder = _load_pickle(label_encoder_path)

    if as_dataframes:
        x_train = pd.DataFrame(x_train, columns=feature_names)
        x_test = pd.DataFrame(x_test, columns=feature_names)

    print("\nDataset summary:")
    print(f"Training samples: {x_train.shape[0]:,}")
    print(f"Testing samples: {x_test.shape[0]:,}")
    print(f"Number of features: {x_train.shape[1]:,}")
    print(f"Number of classes: {len(np.unique(y_train)):,}")
    print(f"Class labels: {list(label_encoder.classes_)}")

    return x_train, x_test, y_train, y_test, feature_names, label_encoder


def load_test_data(
    ml_ready_dir="../data/processed/ml_ready",
    label_encoder_path="../models/label_encoder.pkl",
):
    """Load test arrays and label encoder for evaluation notebooks."""
    x_test = np.load(os.path.join(ml_ready_dir, "X_test_scaled.npy"))
    y_test = np.load(os.path.join(ml_ready_dir, "y_test.npy"))
    label_encoder = _load_pickle(label_encoder_path)

    print("\nTest dataset summary:")
    print(f"Test samples: {x_test.shape[0]:,}")
    print(f"Features: {x_test.shape[1]:,}")
    print(f"Classes: {len(label_encoder.classes_):,}")

    return x_test, y_test, label_encoder
