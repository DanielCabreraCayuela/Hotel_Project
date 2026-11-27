#Code written by Naoufal, edited by Daniel using LLM and restructred to be divided into functions for better modularity and reusability.
"""Step 1: Data Understanding.

Loads the raw hotel reservations dataset, checks data quality (summary
statistics, missing values, and duplicate rows), and generates exploratory
visualizations of booking cancellations and key predictors.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "Hotel Reservations.csv"


def load_raw_data(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw Hotel Reservations CSV and strip whitespace from column names."""
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    return df

#number and % of cancellations
def _plot_cancellation_rate_bar(
    df: pd.DataFrame,
    group_col: str,
    title: str,
    xlabel: str,
    ylim_factor: float = 1.15,
    sort_descending: bool = False,
    show_plots: bool = True,
) -> None:
    """Plot a bar chart of cancellation rate and sample size grouped by a column."""
    summary = df.groupby(group_col)["booking_status"].agg(["mean", "count"])
    if sort_descending:
        summary = summary.sort_values("mean", ascending=False)

    ax = summary["mean"].plot(kind="bar", rot=0, figsize=(8, 5), color="#4C72B0")
    labels = [
        f"{rate:.1%}\n(n={cnt:,})"
        for rate, cnt in zip(summary["mean"], summary["count"])
    ]
    ax.bar_label(ax.containers[0], labels=labels, padding=3, fontsize=9)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cancellation Rate")
    ax.set_ylim(0, summary["mean"].max() * ylim_factor)
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()


def inspect_data_quality(df: pd.DataFrame) -> None:
    """Print summary statistics, missing-value counts, and duplicate-row counts."""
    print("=== Dataset Summary Statistics ===")
    print(df.describe())

    # https://stackoverflow.com/questions/59694988/python-pandas-dataframe-find-missing-values
    missing = df.isnull().sum()
    print("\n=== Missing Values per Column ===")
    print(missing)
    # Every column returns 0, so there are no missing values.

    duplicates_true_false = df.drop("Booking_ID", axis=1).duplicated()
    duplicate_count = int(duplicates_true_false.sum())
    print(f"\nThere are {duplicate_count} duplicate rows")
    # About 30% of rows are duplicates, likely friend groups booking separate rooms
    # (same channel, stay duration, booking date, etc.), so they represent real bookings
    #Therefore, we will not drop duplicates, but we need to take into account that when splitting the data into train/test sets, we should use stratified sampling to ensure that the same group of bookings does not appear in both sets; otherwise it could lead to data leakage and overfitting.


def plot_exploratory_charts(df: pd.DataFrame, show_plots: bool = True) -> None:
    """Generate exploratory plots for class balance, lead time, requests, and segments."""
    # Number and % of cancellations
    class_label_distribution = df["booking_status"].value_counts()
    fig = plt.figure(figsize=(6, 6))
    ax = class_label_distribution.plot.bar(rot=0)
    ax.bar_label(
        ax.containers[0],
        labels=[f"{v:,} ({v / len(df):.1%})" for v in class_label_distribution],
        padding=3,
    )
    ax.set_title("Class Imbalance of Booking Cancellations")
    ax.set_xlabel("Cancelled (0 or 1)")
    ax.grid(False)
    fig.savefig(PROJECT_DIR / "class_label_distribution.png")
    if show_plots:
        plt.show()
    plt.close(fig)

    # Boxplot of lead time by booking status
    plt.figure(figsize=(7, 5))
    ax = sns.boxplot(
        x="booking_status",
        y="lead_time",
        data=df,
        palette=["#4C72B0", "#DD8452"],
        hue="booking_status",
        legend=False,
    )
    ax.set_title("Lead Time Distribution by Booking Status")
    ax.set_xlabel("Booking Status")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not Canceled (0)", "Canceled (1)"])
    ax.set_ylabel("Lead Time (Days)")
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()

    # Bar chart of cancellation rate by number of special requests
    _plot_cancellation_rate_bar(
        df,
        group_col="no_of_special_requests",
        title="Cancellation Rate by Number of Special Requests",
        xlabel="Number of Special Requests",
        ylim_factor=1.15,
        show_plots=show_plots,
    )

    # Bar chart of cancellation rate by market segment type
    _plot_cancellation_rate_bar(
        df,
        group_col="market_segment_type",
        title="Cancellation Rate by Market Segment",
        xlabel="Market Segment Type",
        ylim_factor=1.18,
        sort_descending=True,
        show_plots=show_plots,
    )


def run_data_understanding(
    csv_path: Path = DATA_PATH,
    show_plots: bool = True,
) -> pd.DataFrame:
    """Execute the full Data Understanding step and return the raw DataFrame."""
    raw_df = load_raw_data(csv_path)

    # Map target on a copy for EDA summaries and plots
    eda_df = raw_df.copy()
    # https://stackoverflow.com/questions/23307301/replacing-column-values-in-a-pandas-dataframe
    eda_df["booking_status"] = eda_df["booking_status"].map(
        {"Not_Canceled": 0, "Canceled": 1}
    )

    inspect_data_quality(eda_df)
    plot_exploratory_charts(eda_df, show_plots=show_plots)
    return raw_df


if __name__ == "__main__":
    run_data_understanding()
