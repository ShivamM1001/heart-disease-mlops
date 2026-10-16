
from pathlib import Path

import pandas as pd


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


def load_dataset() -> pd.DataFrame:
    """Load the headerless Cleveland dataset."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE,
        header=None,
        names=COLUMN_NAMES,
        na_values=["?"],
        skipinitialspace=True,
    )

    # Convert all columns to numeric where possible.
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def validate_dataset() -> None:
    """Validate dataset structure and report data quality."""
    df = load_dataset()

    assert not df.empty, "Dataset must not be empty."
    assert df.shape[1] == 14, (
        f"Expected 14 columns, found {df.shape[1]}."
    )
    assert df["num"].notna().all(), (
        "Target column contains missing values."
    )
    assert df["num"].isin([0, 1, 2, 3, 4]).all(), (
        "Unexpected target values found."
    )

    print("=" * 55)
    print("CLEVELAND HEART DISEASE DATA VALIDATION")
    print("=" * 55)

    print(f"\nDataset shape: {df.shape}")
    print(f"Number of records: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five records:")
    print(df.head())

    print("\nMissing values per column:")
    print(df.isna().sum())

    print(f"\nTotal missing cells: {df.isna().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\nOriginal target distribution:")
    print(df["num"].value_counts().sort_index())

    print("\nDataset validation passed.")


if __name__ == "__main__":
    validate_dataset()
