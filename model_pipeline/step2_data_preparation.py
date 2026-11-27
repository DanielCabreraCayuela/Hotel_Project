#one hot encoding used this tutorial: https://www.youtube.com/watch?v=rsyrZnZ8J2o
"""Step 2: Data Preparation.

Encodes the target variable, engineers domain features, one-hot encodes
categorical predictors, generates post-encoding correlation charts, splits the
dataset into train/test sets, preserves unscaled booking values for financial
evaluation, and scales features with MinMaxScaler.
"""

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from step1_data_understanding import (
    ARTIFACTS_DIR,
    DATA_PATH,
    FIGURES_DIR,
    load_raw_data,
)

PREPARED_DATA_PATH = ARTIFACTS_DIR / "prepared_data.joblib"

CATEGORICAL_COLS = [
    "room_type_reserved",
    "type_of_meal_plan",
    "market_segment_type",
]
DROP_COLS = ["Booking_ID", "arrival_year"]
RAW_NUMERIC_COLS_FOR_CAT_CHART = [
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "lead_time",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "avg_price_per_room",
    "no_of_special_requests",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map target labels and create engineered stay, guest, value, and history features."""
    prepared = df.copy()
    # https://stackoverflow.com/questions/23307301/replacing-column-values-in-a-pandas-dataframe
    prepared["booking_status"] = prepared["booking_status"].map(
        {"Not_Canceled": 0, "Canceled": 1}
    )
    prepared["total_nights"] = (
        prepared["no_of_week_nights"] + prepared["no_of_weekend_nights"]
    )
    prepared["total_guests"] = prepared["no_of_children"] + prepared["no_of_adults"]
    prepared["total_booking_value"] = (
        prepared["avg_price_per_room"] * prepared["total_nights"]
    )
    prepared["total_previous_bookings"] = (
        prepared["no_of_previous_cancellations"]
        + prepared["no_of_previous_bookings_not_canceled"]
    )
    return prepared


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns and drop identifier/year columns."""
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(
        transform="pandas"
    )
    ohe_transform = ohe.fit_transform(df[CATEGORICAL_COLS])

    ohe_df = pd.concat([df, ohe_transform], axis=1).drop(
        columns=CATEGORICAL_COLS + DROP_COLS
    )
    return ohe_df


def _plot_correlation_bar(
    df: pd.DataFrame,
    title: str,
    output_filename: str,
    figsize: tuple[float, float],
    xlim_pad: float,
    annotate_binary_rates: bool = False,
    show_plots: bool = True,
) -> None:
    """Plot a horizontal bar chart of Pearson correlation with booking_status."""
    corr = df.corr()["booking_status"].drop("booking_status").sort_values()

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#4C72B0" if val < 0 else "#DD8452" for val in corr]
    bars = ax.barh(corr.index, corr.values, color=colors, edgecolor="none", height=0.72)

    ax.axvline(0, color="black", linewidth=0.9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    offset_step = 0.005 if annotate_binary_rates else 0.008
    font_size = 8 if annotate_binary_rates else 8.5

    for bar, (col_name, val) in zip(bars, corr.items()):
        if annotate_binary_rates:
            unique_vals = set(df[col_name].unique())
            if unique_vals.issubset({0, 1, 0.0, 1.0}):
                subset = df[df[col_name] == 1]["booking_status"]
                label_text = f"{val:+.2f}  ({subset.mean():.1%} cancel, n={len(subset):,})"
            else:
                label_text = f"{val:+.2f}"
        else:
            label_text = f"{val:+.2f}"

        offset = offset_step if val >= 0 else -offset_step
        ha = "left" if val >= 0 else "right"
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            label_text,
            va="center",
            ha=ha,
            fontsize=font_size,
            color="#222222",
        )

    ax.set_title(title, fontsize=12 if annotate_binary_rates else 13, fontweight="bold", pad=12)
    ax.set_xlabel("Pearson Correlation Coefficient (r)", fontsize=10)
    ax.set_xlim(corr.min() - xlim_pad, corr.max() + xlim_pad)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / output_filename, dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_feature_correlations(ohe_df: pd.DataFrame, show_plots: bool = True) -> None:
    """Save and display correlation charts for all predictors and categorical/engineered predictors."""
    # Chart 1: Unified correlation ranking across all predictors
    _plot_correlation_bar(
        ohe_df,
        title="Pearson Correlation with Booking Cancellation (All Features)",
        output_filename="all_features_correlation.png",
        figsize=(10, 11),
        xlim_pad=0.07,
        annotate_binary_rates=False,
        show_plots=show_plots,
    )

    # Chart 2: Categorical, binary & engineered features with cancel rates and sample counts
    categorical_df = ohe_df.drop(columns=RAW_NUMERIC_COLS_FOR_CAT_CHART)
    _plot_correlation_bar(
        categorical_df,
        title="Correlation & Group Cancellation Rates: Categorical, Binary & Engineered Features",
        output_filename="categorical_features_correlation.png",
        figsize=(10.5, 9),
        xlim_pad=0.11,
        annotate_binary_rates=True,
        show_plots=show_plots,
    )


def split_and_scale(
    ohe_df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 87,
) -> dict[str, Any]:
    """Split features/target into train/test sets and apply MinMaxScaler fit on train only."""
    X = ohe_df.drop(columns=["booking_status"])
    y = ohe_df["booking_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Save unscaled booking values (in Euros) before scaling X_test for financial evaluation
    test_booking_value = X_test["total_booking_value"].copy()

    scaler = MinMaxScaler().set_output(transform="pandas")
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "test_booking_value": test_booking_value,
        "scaler": scaler,
    }


def run_data_preparation(
    raw_df: pd.DataFrame | None = None,
    csv_path: Path = DATA_PATH,
    show_plots: bool = True,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """Run feature engineering, encoding, correlation plots, splitting, and scaling."""
    if raw_df is None:
        raw_df = load_raw_data(csv_path)

    featured_df = engineer_features(raw_df)
    ohe_df = encode_features(featured_df)

    plot_feature_correlations(ohe_df, show_plots=show_plots)

    prepared = split_and_scale(ohe_df)
    print("\n=== Scaled Training Features Preview ===")
    print(prepared["X_train"].head())

    if save_artifacts:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(prepared, PREPARED_DATA_PATH)
        print(f"Saved prepared data to {PREPARED_DATA_PATH}")

    return prepared


if __name__ == "__main__":
    run_data_preparation()
