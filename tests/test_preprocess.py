
import pandas as pd
import pytest

from src.preprocess import (
    build_preprocessor,
    load_raw_dataset,
    prepare_features_target,
)


def test_dataset_shape():
    df = load_raw_dataset()

    assert df.shape == (303, 14)


def test_missing_values_are_detected():
    df = load_raw_dataset()

    assert df.isna().sum().sum() == 6
    assert df["ca"].isna().sum() == 4
    assert df["thal"].isna().sum() == 2


def test_binary_target():
    df = load_raw_dataset()
    X, y = prepare_features_target(df)

    assert X.shape == (303, 13)
    assert y.shape == (303,)
    assert set(y.unique()) == {0, 1}
    assert y.value_counts().to_dict() == {0: 164, 1: 139}
    assert "num" not in X.columns


def test_preprocessor_handles_missing_values():
    df = load_raw_dataset()
    X, _ = prepare_features_target(df)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 0


def test_preprocessor_rejects_invalid_target():
    df = load_raw_dataset()
    df.loc[0, "num"] = 5

    with pytest.raises(ValueError):
        prepare_features_target(df)
