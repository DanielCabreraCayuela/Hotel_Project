"""Step 5: Evaluation.

Translates model outputs into business insights and decision rules:
- Subquestion 1: Top cancellation predictors overall and among hotel-controlled features.
- Financial impact: Expected vs. actual monetary loss from cancellations.
- Subquestion 2: Dynamic diminishing-returns ("Knee") threshold and two-panel
  cumulative gains and marginal return curves.
"""

from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step1_data_understanding import PROJECT_DIR
from step2_data_preparation import PREPARED_DATA_PATH
from step3_data_modeling import TRAINED_MODELS_PATH
from step4_data_testing import TEST_RESULTS_PATH

HOTEL_CONTROLLED_PREFIXES = (
    "avg_price_per_room",
    "price_per_person",
    "total_booking_value",
    "room_type_reserved_",
    "type_of_meal_plan_",
    "required_car_parking_space",
    "market_segment_type_",
)


def evaluate_feature_drivers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    rf_model: Any,
    log_reg_model: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank overall cancellation predictors and hotel-controlled predictors (Subquestion 1)."""
    print("\nWhich factors/features most strongly predict hotel cancellations?")

    direction = ["+" if w >= 0 else "-" for w in log_reg_model.coef_[0]]
    sq1_df = pd.DataFrame({
        "Feature": X_train.columns,
        "RF_Importance": rf_model.feature_importances_,
        "LogReg_Direction": direction,
    }).sort_values("RF_Importance", ascending=False)

    print("\n=== Top 15 Cancellation Predictors (Subquestion 1) ===")
    print(sq1_df.head(15).to_string(index=False))

    hotel_controlled_cols = [
        col for col in X_train.columns if col.startswith(HOTEL_CONTROLLED_PREFIXES)
    ]
    bivariate_corr = X_train[hotel_controlled_cols].apply(lambda col: col.corr(y_train))

    sq1_hotel_df = sq1_df.rename(columns={"Feature": "Hotel_Controlled_Feature"})
    sq1_hotel_df = sq1_hotel_df[
        sq1_hotel_df["Hotel_Controlled_Feature"].isin(hotel_controlled_cols)
    ].copy()
    sq1_hotel_df["Pearson_r"] = sq1_hotel_df["Hotel_Controlled_Feature"].map(bivariate_corr)
    sq1_hotel_df = sq1_hotel_df.sort_values("RF_Importance", ascending=False).reset_index(
        drop=True
    )

    print("\n=== Subquestion 1: Hotel-Controlled Features Associated with Cancellations ===")
    print(sq1_hotel_df.round(4).to_string(index=False))
    return sq1_df, sq1_hotel_df


def evaluate_diminishing_returns_cutoff(
    y_test: pd.Series,
    rf_probs: np.ndarray,
    test_booking_value: pd.Series,
    show_plots: bool = True,
) -> pd.DataFrame:
    """Compute expected monetary loss, dynamic knee cutoff rule, and two-panel gains plot."""
    print("\nWhat is the expected monetary loss from cancellations in the test set?")
    expected_loss_per_booking = rf_probs * test_booking_value

    print(f"\nTotal Portfolio Value in Test Set: €{test_booking_value.sum():,.2f}")
    print(f"Total Expected Monetary Loss (Predicted): €{expected_loss_per_booking.sum():,.2f}")
    print(
        f"Actual Monetary Loss from True Cancellations: "
        f"€{test_booking_value[y_test == 1].sum():,.2f}"
    )

    sq2_df = pd.DataFrame({
        "Actual_Canceled": y_test.values,
        "Cancel_Prob": rf_probs,
        "Booking_Value_EUR": test_booking_value.values,
    })
    sq2_df["Expected_Loss_EUR"] = sq2_df["Cancel_Prob"] * sq2_df["Booking_Value_EUR"]
    sq2_df["Actual_Loss_EUR"] = sq2_df["Actual_Canceled"] * sq2_df["Booking_Value_EUR"]

    # Rank bookings from highest to lowest Expected Monetary Loss
    sq2_df = sq2_df.sort_values("Expected_Loss_EUR", ascending=False).reset_index(drop=True)

    total_bookings = len(sq2_df)
    total_expected_loss = sq2_df["Expected_Loss_EUR"].sum()
    total_actual_loss = sq2_df["Actual_Loss_EUR"].sum()

    # Normalized fractions [0, 1] and percentages [0, 100]
    x_frac = np.arange(1, total_bookings + 1) / total_bookings
    y_actual_frac = sq2_df["Actual_Loss_EUR"].cumsum().values / total_actual_loss
    y_expected_frac = sq2_df["Expected_Loss_EUR"].cumsum().values / total_expected_loss

    sq2_df["Cum_Bookings_Pct"] = x_frac * 100
    sq2_df["Cum_Actual_Loss_Pct"] = y_actual_frac * 100
    sq2_df["Cum_Expected_Loss_Pct"] = y_expected_frac * 100

    # Calculate Diminishing Returns ("Knee") Point using Expected Loss Curve.
    # Maximum net gain over random baseline (y - x) occurs where marginal slope dy/dx = 1.0,
    # guaranteeing a smooth, out-of-sample-safe geometric knee without peeking at y_test.
    net_gain_over_random = y_expected_frac - x_frac
    knee_idx = int(np.argmax(net_gain_over_random))

    optimal_pct_flagged = sq2_df.loc[knee_idx, "Cum_Bookings_Pct"]
    eur_cutoff_optimal = sq2_df.loc[knee_idx, "Expected_Loss_EUR"]
    captured_expected_pct = sq2_df.loc[knee_idx, "Cum_Expected_Loss_Pct"]
    captured_actual_eur = sq2_df.loc[:knee_idx, "Actual_Loss_EUR"].sum()
    captured_actual_pct = sq2_df.loc[knee_idx, "Cum_Actual_Loss_Pct"]
    true_cancel_rate_opt = sq2_df.loc[:knee_idx, "Actual_Canceled"].mean() * 100

    print("\n=== Subquestion 2: Data-Driven Diminishing Returns ('Knee') Decision Rule ===")
    print(f"Total Test Bookings:                  {total_bookings:,}")
    print(f"Total Actual Revenue Lost (Test):     €{total_actual_loss:,.2f}")
    print(f"Optimal Cutoff (Marginal Slope = 1):  Expected Loss >= €{eur_cutoff_optimal:,.2f}")
    print(
        f"Bookings Flagged at Knee Point:       "
        f"{knee_idx + 1:,} reservations ({optimal_pct_flagged:.1f}% of total)"
    )
    print(f"Share of Expected Loss Captured:      {captured_expected_pct:.1f}%")
    print(
        f"Actual Revenue Loss Captured:         "
        f"€{captured_actual_eur:,.2f} ({captured_actual_pct:.1f}% of all lost revenue)"
    )
    print(f"True Cancellation Rate in Flagged:    {true_cancel_rate_opt:.1f}%")

    # Compute Marginal Return Rate (Rolling 5% window derivative dy/dx) for Panel 2
    window = max(1, int(total_bookings * 0.05))
    marginal_gain = pd.Series(y_actual_frac).diff(window) / pd.Series(x_frac).diff(window)

    # Two-Panel Plot: Cumulative Gains + Marginal Diminishing Returns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left Panel: Cumulative Gains Curve with Dynamic Knee Point ---
    ax1.plot(
        sq2_df["Cum_Bookings_Pct"],
        sq2_df["Cum_Actual_Loss_Pct"],
        color="#DD8452",
        linewidth=2.3,
        label="Actual Loss Captured (RF Ranking)",
    )
    ax1.plot(
        sq2_df["Cum_Bookings_Pct"],
        sq2_df["Cum_Expected_Loss_Pct"],
        color="#55A868",
        linestyle="-.",
        linewidth=1.8,
        label="Expected Loss Curve (Used for Knee)",
    )
    ax1.plot([0, 100], [0, 100], "k--", linewidth=1, label="Random Baseline (Slope = 1.0)")

    point_x = optimal_pct_flagged
    point_y = captured_actual_pct

    ax1.axvline(point_x, color="#4C72B0", linestyle=":", linewidth=1.3)
    ax1.axhline(point_y, color="#4C72B0", linestyle=":", linewidth=1.3)
    ax1.scatter([point_x], [point_y], color="#4C72B0", s=70, zorder=5)

    ax1.annotate(
        f"Diminishing-Returns Knee:\n"
        f"Expected Loss ≥ €{eur_cutoff_optimal:,.2f}\n"
        f"({point_x:.1f}% bookings → {point_y:.1f}% actual loss)",
        xy=(point_x, point_y),
        xytext=(min(point_x + 10, 52), max(point_y - 22, 20)),
        arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=1.2),
        fontsize=9,
        fontweight="bold",
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F4F7FB", edgecolor="#4C72B0", lw=1),
    )

    ax1.set_title(
        "Cumulative Gains: Optimal Diminishing-Returns Cutoff",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax1.set_ylabel("% of Total Cancellation Revenue Loss Captured", fontsize=10)
    ax1.set_ylim(0, 102)
    ax1.legend(loc="lower right", fontsize=8.5)

    # --- Right Panel: Marginal Return per 1% of Bookings Targeted ---
    ax2.plot(
        sq2_df["Cum_Bookings_Pct"],
        marginal_gain,
        color="#4C72B0",
        linewidth=2,
        label="Marginal Loss Captured per 1% Targeted",
    )
    ax2.axhline(
        1.0,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label="Break-Even Efficiency Line (1% Gain / 1% Effort)",
    )
    ax2.axvline(
        point_x,
        color="#4C72B0",
        linestyle=":",
        linewidth=1.3,
        label=f"Knee Cutoff ({point_x:.1f}% Targeted)",
    )

    ax2.fill_between(
        sq2_df["Cum_Bookings_Pct"],
        marginal_gain,
        1.0,
        where=(marginal_gain >= 1.0),
        color="#55A868",
        alpha=0.18,
        label="High-Efficiency Zone (>1x Return)",
    )

    ax2.set_title(
        "Marginal Return Curve (Diminishing Returns Diagnostic)",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax2.set_ylabel("Marginal % of Loss Captured per 1% Bookings Targeted", fontsize=10)
    ax2.set_ylim(0, max(3.5, float(np.nanmax(marginal_gain)) * 1.05))
    ax2.legend(loc="upper right", fontsize=8.5)

    for ax in (ax1, ax2):
        ax.set_xlabel("% of Reservations Targeted (Ranked by Expected Loss)", fontsize=10)
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.35, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(
        PROJECT_DIR / "subquestion2_diminishing_returns.png",
        dpi=300,
        bbox_inches="tight",
    )
    if show_plots:
        plt.show()
    plt.close(fig)

    return sq2_df


def run_evaluation(
    prepared: dict[str, Any] | None = None,
    models: dict[str, Any] | None = None,
    test_results: dict[str, dict[str, Any]] | None = None,
    show_plots: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run business evaluation for Subquestion 1 and Subquestion 2."""
    if prepared is None:
        prepared = joblib.load(PREPARED_DATA_PATH)
    if models is None:
        models = joblib.load(TRAINED_MODELS_PATH)
    if test_results is None:
        test_results = joblib.load(TEST_RESULTS_PATH)

    sq1_df, sq1_hotel_df = evaluate_feature_drivers(
        prepared["X_train"],
        prepared["y_train"],
        models["Random Forest"],
        models["Logistic Regression"],
    )
    sq2_df = evaluate_diminishing_returns_cutoff(
        prepared["y_test"],
        test_results["Random Forest"]["probas"],
        prepared["test_booking_value"],
        show_plots=show_plots,
    )

    return {
        "sq1_overall": sq1_df,
        "sq1_hotel_controlled": sq1_hotel_df,
        "sq2_ranking": sq2_df,
    }


if __name__ == "__main__":
    run_evaluation()
