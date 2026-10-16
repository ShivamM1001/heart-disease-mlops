
"""Train, evaluate, compare, and save heart disease classifiers."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.preprocess import (
    build_preprocessor,
    load_raw_dataset,
    prepare_features_target,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


def build_models() -> dict[str, Pipeline]:
    """Create models with preprocessing inside each pipeline."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def train_and_evaluate() -> dict:
    """Run cross-validation, evaluate test performance, and save artifacts."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_dataset()
    X, y = prepare_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("=" * 65)
    print("HEART DISEASE MODEL TRAINING")
    print("=" * 65)
    print(f"Full dataset: {X.shape}")
    print(f"Training records: {len(X_train)}")
    print(f"Test records: {len(X_test)}")
    print("\nTraining target distribution:")
    print(y_train.value_counts().sort_index())

    models = build_models()
    scoring = ["accuracy", "precision", "recall", "roc_auc"]

    cv_results = {}
    test_results = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"\n{'-' * 65}")
        print(f"Model: {name}")
        print("-" * 65)

        # Each fold fits its own preprocessing transformations.
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=CV_FOLDS,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        cv_results[name] = {
            metric: {
                "mean": float(scores[f"test_{metric}"].mean()),
                "std": float(scores[f"test_{metric}"].std()),
            }
            for metric in scoring
        }

        for metric in scoring:
            values = scores[f"test_{metric}"]
            print(
                f"CV {metric}: "
                f"{values.mean():.3f} +/- {values.std():.3f}"
            )

        # Fit the full pipeline on the training partition.
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        test_results[name] = {
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "precision": float(
                precision_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "roc_auc": float(
                roc_auc_score(y_test, probabilities)
            ),
        }

        fitted_models[name] = model

        print("\nHeld-out test metrics:")
        for metric, value in test_results[name].items():
            print(f"{metric}: {value:.3f}")

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predictions,
                target_names=["Absence", "Presence"],
                zero_division=0,
            )
        )

    # Select using cross-validation ROC-AUC, not test performance.
    best_name = max(
        cv_results,
        key=lambda name: cv_results[name]["roc_auc"]["mean"],
    )
    best_model = fitted_models[best_name]

    print(f"\nSelected model by CV ROC-AUC: {best_name}")

    # Save the entire fitted preprocessing + model pipeline.
    model_path = ARTIFACT_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)

    summary = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "selection_metric": "mean cross-validation ROC-AUC",
        "selected_model": best_name,
        "cross_validation": cv_results,
        "test_metrics": test_results,
    }

    metrics_path = ARTIFACT_DIR / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    # Compare model performance using cross-validation means.
    cv_table = pd.DataFrame({
        name: {
            metric: values["mean"]
            for metric, values in metrics.items()
        }
        for name, metrics in cv_results.items()
    }).T

    print("\nCross-validation comparison:")
    print(cv_table.round(3))

    ax = cv_table[scoring].plot(
        kind="bar",
        figsize=(10, 6),
    )
    ax.set_title("Model Comparison — Cross-Validation")
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean CV score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(
        SCREENSHOT_DIR / "05_model_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Final diagnostic plots for the selected model.
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Absence", "Presence"],
        cmap="Blues",
    )
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(
        SCREENSHOT_DIR / "06_confusion_matrix.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test, y_prob
    )
    test_auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"{best_name} (AUC = {test_auc:.3f})",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Held-out Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        SCREENSHOT_DIR / "07_roc_curve.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print("Saved plots in:", SCREENSHOT_DIR)

    return summary


if __name__ == "__main__":
    train_and_evaluate()
