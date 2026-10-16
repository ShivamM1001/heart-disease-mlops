
"""Train, evaluate, compare, and track heart disease classifiers with MLflow."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
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
from sklearn.model_selection import cross_validate, train_test_split
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

EXPERIMENT_NAME = "heart-disease-classification"


def configure_mlflow() -> None:
    """Configure a persistent, local MLflow tracking database."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    database_path = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(
        f"sqlite:///{database_path.as_posix()}"
    )
    mlflow.set_experiment(EXPERIMENT_NAME)


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


def log_model_parameters(name: str, model: Pipeline) -> None:
    """Log model configuration and shared training parameters."""
    mlflow.log_param("model_name", name)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("cv_folds", CV_FOLDS)
    mlflow.log_param("dataset", "UCI Heart Disease - Cleveland")
    mlflow.log_param("selection_metric", "mean_cv_roc_auc")

    classifier = model.named_steps["classifier"]

    for key, value in classifier.get_params().items():
        if value is None or isinstance(
            value, (str, int, float, bool)
        ):
            mlflow.log_param(f"classifier_{key}", value)


def train_and_evaluate() -> dict:
    """Train models, log experiments, evaluate, and save artifacts."""
    configure_mlflow()

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
    print("HEART DISEASE MODEL TRAINING WITH MLFLOW")
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
    run_ids = {}

    for name, model in models.items():
        print(f"\n{'-' * 65}")
        print(f"Model: {name}")
        print("-" * 65)

        with mlflow.start_run(run_name=name) as run:
            run_ids[name] = run.info.run_id
            log_model_parameters(name, model)

            mlflow.set_tag("model_family", name)
            mlflow.set_tag("dataset_scope", "Cleveland subset")

            # Preprocessing is fitted separately within each CV fold.
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
                    "mean": float(
                        scores[f"test_{metric}"].mean()
                    ),
                    "std": float(
                        scores[f"test_{metric}"].std()
                    ),
                }
                for metric in scoring
            }

            for metric in scoring:
                values = scores[f"test_{metric}"]
                mean_value = float(values.mean())
                std_value = float(values.std())

                print(
                    f"CV {metric}: "
                    f"{mean_value:.3f} +/- {std_value:.3f}"
                )

                mlflow.log_metric(
                    f"cv_{metric}_mean", mean_value
                )
                mlflow.log_metric(
                    f"cv_{metric}_std", std_value
                )

            # Fit the complete pipeline on the training partition.
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

            for metric, value in test_results[name].items():
                mlflow.log_metric(f"test_{metric}", value)
                print(f"Test {metric}: {value:.3f}")

            print("\nClassification report:")
            print(
                classification_report(
                    y_test,
                    predictions,
                    target_names=["Absence", "Presence"],
                    zero_division=0,
                )
            )

            # Store the fitted pipeline in this model's MLflow run.
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format="pickle",
            )

            fitted_models[name] = model

    # Select using cross-validation ROC-AUC, not test performance.
    best_name = max(
        cv_results,
        key=lambda name: cv_results[name]["roc_auc"]["mean"],
    )
    best_model = fitted_models[best_name]

    print(f"\nSelected model by CV ROC-AUC: {best_name}")

    # Save the fitted preprocessing + classifier pipeline locally.
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

    cv_table = pd.DataFrame({
        name: {
            metric: values["mean"]
            for metric, values in metrics.items()
        }
        for name, metrics in cv_results.items()
    }).T

    print("\nCross-validation comparison:")
    print(cv_table.round(3))

    # Model comparison plot.
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

    comparison_path = SCREENSHOT_DIR / "05_model_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Diagnostic plots for the selected model.
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

    confusion_path = SCREENSHOT_DIR / "06_confusion_matrix.png"
    plt.savefig(confusion_path, dpi=150, bbox_inches="tight")
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
    plt.plot(
        [0, 1], [0, 1],
        linestyle="--",
        label="Random baseline",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Held-out Test Set")
    plt.legend()
    plt.tight_layout()

    roc_path = SCREENSHOT_DIR / "07_roc_curve.png"
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Add final diagnostic artifacts and selection status to each run.
    client = mlflow.tracking.MlflowClient()

    for name, run_id in run_ids.items():
        client.set_tag(
            run_id,
            "selected_model",
            str(name == best_name).lower(),
        )
        client.log_artifact(run_id, str(metrics_path))

        if name == best_name:
            for plot_path in (
                comparison_path,
                confusion_path,
                roc_path,
            ):
                client.log_artifact(run_id, str(plot_path))

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print("Saved plots in:", SCREENSHOT_DIR)
    print(f"MLflow experiment: {EXPERIMENT_NAME}")
    print(f"MLflow database: {PROJECT_ROOT / 'mlflow.db'}")

    return summary


if __name__ == "__main__":
    train_and_evaluate()
