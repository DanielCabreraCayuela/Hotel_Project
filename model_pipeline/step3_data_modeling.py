"""Step 3: Data Modeling.

Trains candidate classification models (Random Forest, Decision Tree, and
Logistic Regression) on the prepared training dataset and optionally supports
LazyPredict benchmarking and GridSearchCV hyperparameter tuning.
"""

from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from step2_data_preparation import ARTIFACTS_DIR, PREPARED_DATA_PATH

TRAINED_MODELS_PATH = ARTIFACTS_DIR / "trained_models.joblib"


def run_lazy_predict_benchmark(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    sample_size: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Benchmark multiple classifiers on a subsample using LazyClassifier (optional)."""
    # https://pypi.org/project/lazypredict/
    from lazypredict.Supervised import LazyClassifier

    # Subsample to avoid excessive RAM usage in LazyPredict
    X_small = X_train.sample(n=sample_size, random_state=random_state)
    y_small = y_train.loc[X_small.index]

    classifier = LazyClassifier()
    models, _ = classifier.fit(X_small, X_test, y_small, y_test)
    print(models)
    return models


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """Run GridSearchCV for RandomForestClassifier hyperparameter tuning (optional)."""
    # https://www.geeksforgeeks.org/machine-learning/hyperparameter-tuning-using-gridsearchcv-and-kerasclassifier/
    base_rf = RandomForestClassifier(n_jobs=-1)
    params = {
        "n_estimators": [500, 1000],
        "max_depth": [10, 20],
    }
    gs = GridSearchCV(estimator=base_rf, param_grid=params, cv=5, n_jobs=-1)
    gs.fit(X_train, y_train)
    print(f"Best params are: {gs.best_params_}")
    print(f"Accuracy is: {gs.best_score_}")
    return gs


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 87,
) -> dict[str, Any]:
    """Train Random Forest, Logistic Regression, and Decision Tree classifiers."""
    # Tuned hyperparameters (from prior GridSearchCV: max_depth=15, n_estimators=1500)
    rf_clf = RandomForestClassifier(
        max_depth=15,
        n_estimators=1500,
        n_jobs=1,
        class_weight="balanced",
        random_state=random_state,
    )
    rf_clf.fit(X_train, y_train)

    log_reg_clf = LogisticRegression(
        class_weight="balanced",
        random_state=random_state,
    )
    log_reg_clf.fit(X_train, y_train)

    dec_tree_clf = DecisionTreeClassifier(class_weight="balanced")
    dec_tree_clf.fit(X_train, y_train)

    return {
        "Random Forest": rf_clf,
        "Logistic Regression": log_reg_clf,
        "Decision Tree": dec_tree_clf,
    }


def run_data_modeling(
    prepared: dict[str, Any] | None = None,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """Load prepared training data (if needed) and fit all models."""
    if prepared is None:
        prepared = joblib.load(PREPARED_DATA_PATH)

    models = train_models(prepared["X_train"], prepared["y_train"])
    print("\n=== Trained Models ===")
    for name in models:
        print(f"- {name}")

    if save_artifacts:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(models, TRAINED_MODELS_PATH)
        print(f"Saved trained models to {TRAINED_MODELS_PATH}")

    return models


if __name__ == "__main__":
    run_data_modeling()
