
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DATA_DIR / "heart_disease.csv"


def download_heart_disease() -> pd.DataFrame:
    """Download and save the UCI Heart Disease dataset."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading Heart Disease dataset from UCI...")

    dataset = fetch_ucirepo(id=45)

    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    # Normalize column names.
    features.columns = [
        str(column).strip().lower() for column in features.columns
    ]
    targets.columns = [
        str(column).strip().lower() for column in targets.columns
    ]

    # Combine input features and target.
    df = pd.concat([features, targets], axis=1)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDownload successful!")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Dataset shape: {df.shape}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five records:")
    print(df.head())

    print("\nMissing values:")
    print(df.isna().sum())

    return df


if __name__ == "__main__":
    download_heart_disease()
