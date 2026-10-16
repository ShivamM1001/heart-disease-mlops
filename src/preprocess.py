
"""Data loading and preprocessing utilities for the MLOps project."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"

COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

# Measured numerical variables
NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

# Coded categorical variables
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


def load_raw_dataset() -> pd.DataFrame:
    """Load the headerless Cleveland dataset and normalize missing values."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}. "
            "Place processed.cleveland.data in data/raw/."
        )

    df = pd.read_csv(
        DATA_FILE,
        header=None,
        names=COLUMN_NAMES,
        na_values=["?"],
        skipinitialspace=True,
    )

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df.empty:
        raise ValueError("The dataset is empty.")

    if df["num"].isna().any():
        raise ValueError("The target column contains missing values.")

    if not df["num"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError("The target contains unexpected values.")

    return df


def prepare_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create predictor features X and binary target y."""
    if "num" not in df.columns:
        raise ValueError("Dataset must contain the 'num' target column.")

    if df["num"].isna().any():
        raise ValueError("The target column contains missing values.")

    if not df["num"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError("The target contains unexpected values.")

    # 0 = absence; 1-4 = presence.
    y = (df["num"] > 0).astype("int64")
    y.name = "target"

    # Exclude the original target to prevent target leakage.
    X = df.drop(columns=["num"]).copy()

    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Create a reusable preprocessing transformer."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


if __name__ == "__main__":
    data = load_raw_dataset()
    X, y = prepare_features_target(data)

    print("Raw dataset shape:", data.shape)
    print("Feature matrix shape:", X.shape)
    print("Target shape:", y.shape)
    print("\nMissing values before imputation:")
    print(X.isna().sum())
    print("\nBinary target distribution:")
    print(y.value_counts().sort_index())
    print("\nPreprocessor created successfully.")
