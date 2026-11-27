"""Step 4: Data Testing.

Generates predictions and cancellation probabilities on the held-out test set,
compares model performance across classification metrics, prints the Random
Forest confusion matrix, and plots the precision-recall threshold curve.
"""

from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from step2_data_preparation import ARTIFACTS_DIR, PREPARED_DATA_PATH
from step3_data_modeling import TRAINED_MODELS_PATH

TEST_RESULTS_PATH = ARTIFACTS_DIR / "test_results.joblib"


def test_models(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, dict[str, Any]]:
    """Run predictions and compute test metrics for each trained model."""
    results: dict[str, dict[str, Any]] = {}

    print("\nModel Comparison & Accuracy Justification")
    for name, model in models.items():
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probas),
        }
        print(
            f"{name:20s} | Acc: {metrics['accuracy']:.4f} | "
            f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1']:.4f} | AUC: {metrics['roc_auc']:.4f}"
        )

        results[name] = {
            "preds": preds,
            "probas": probas,
            "metrics": metrics,
        }

    rf_preds = results["Random Forest"]["preds"]
    print("\nRandom Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_preds))
    return results


def plot_precision_recall_threshold(
    y_test: pd.Series,
    rf_probas: Any,
    show_plots: bool = True,
) -> None:
    """Plot Random Forest precision and recall across decision thresholds."""
    precision, recall, thresholds = precision_recall_curve(y_test, rf_probas)

    plt.figure()
    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.plot(thresholds, recall[:-1], label="Recall")
    plt.xlabel("Threshold")
    plt.title("Random Forest Precision & Recall vs. Threshold")
    plt.legend()
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()


def run_data_testing(
    prepared: dict[str, Any] | None = None,
    models: dict[str, Any] | None = None,
    show_plots: bool = True,
    save_artifacts: bool = True,
) -> dict[str, dict[str, Any]]:
    """Load prepared data and trained models (if needed) and evaluate on X_test."""
    if prepared is None:
        prepared = joblib.load(PREPARED_DATA_PATH)
    if models is None:
        models = joblib.load(TRAINED_MODELS_PATH)

    X_test = prepared["X_test"]
    y_test = prepared["y_test"]

    test_results = test_models(models, X_test, y_test)
    plot_precision_recall_threshold(
        y_test,
        test_results["Random Forest"]["probas"],
        show_plots=show_plots,
    )

    if save_artifacts:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(test_results, TEST_RESULTS_PATH)
        print(f"Saved test results to {TEST_RESULTS_PATH}")

    return test_results


if __name__ == "__main__":
    run_data_testing()
