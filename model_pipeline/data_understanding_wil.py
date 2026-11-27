"""Code behind the business understanding slide (3) and the data understanding slide (4) - Wil
Group 9 - Data Science and AI for Business.
Run from the root of the repository, with "Hotel Reservations.csv" next to this file
(or change DATA_PATH below). The "# %%" lines split the file into cells, like in Colab.
Each comment says what the line does -> [which number or chart on the slides it gives].
This is AI generated code! (Claude)
"""

# %% Load the data
import pandas as pd                                                           # tables
import matplotlib.pyplot as plt                                               # charts
DATA_PATH = "Hotel Reservations.csv"                                          # path of the dataset in the repository (change it if the file is in another folder)
du_raw = pd.read_csv(DATA_PATH)                                               # reads the CSV: one row = one booking -> [used by everything below]

# %% Slide 4 - tiles: the dataset
print("Bookings:", du_raw.shape[0])                                           # number of rows -> [tile "36,275 bookings"]
print("Variables:", du_raw.shape[1])                                          # number of columns -> [tile "19 variables"]
du_date_cols = du_raw.columns[du_raw.columns.str.startswith("arrival_")]      # the 3 columns whose name starts with "arrival_" (year, month, day) -> [tiles "2017-2018" + "37 impossible dates"]
du_arrival = pd.to_datetime(du_raw[du_date_cols].set_axis(["year", "month", "day"], axis=1),
                            errors="coerce")                                  # one real date per booking; a date that does not exist becomes NaT (empty) -> [tiles "2017-2018" + "37"]
print("Arrivals from", du_arrival.min().date(), "to", du_arrival.max().date())   # first and last arrival date -> [tile "2017-2018, Jul 2017 - Dec 2018"]
du_cancelled = (du_raw["booking_status"] == "Canceled").astype(int)           # 1 = cancelled, 0 = not cancelled -> [tile "32.8%" + chart 3]
print("Cancelled (%):", round(du_cancelled.mean() * 100, 1))                  # share of cancelled bookings -> [tile "32.8% cancelled"]

# %% Slide 4 - tiles: data quality
print("Missing values:", du_raw.isnull().sum().sum())                         # empty cells in the whole dataset -> [tile "0 missing values"]
du_identifier = du_raw.columns[du_raw.nunique() == len(du_raw)]               # column with a different value on every row = the booking ID -> [tile "10,275 duplicate copies"]
print("Duplicate copies:", du_raw.drop(columns=du_identifier).duplicated().sum())   # rows identical to an earlier row apart from the ID -> [tile "10,275 duplicate copies"]
print("Impossible arrival dates:", du_arrival.isna().sum())                   # dates that do not exist in the calendar (all 29 Feb 2018) -> [tile "37 impossible dates"]

# %% Slide 4 - chart 1: lead time distribution
du_lead = du_raw["lead_time"]                                                 # lead time of every booking, in days -> [chart 1]
ax = du_lead.plot.hist(bins=40, color="#1E75BE", figsize=(6.6, 2.6))          # histogram: number of bookings per lead time range -> [chart 1]
ax.axvline(du_lead.mean(), color="black", linestyle="--",
           label=f"mean {du_lead.mean():.0f} days")                           # dashed line at the mean -> [chart 1: "mean 85 days"]
ax.axvline(du_lead.median(), color="orange", linewidth=2,
           label=f"median {du_lead.median():.0f} days")                       # line at the median -> [chart 1: "median 57 days"]
ax.set_title("Lead time: mean 85 vs median 57 days")                         # chart title -> [chart 1]
ax.set_xlabel("Days between booking and arrival")                             # x axis name -> [chart 1]
ax.set_ylabel("Bookings")                                                     # y axis name -> [chart 1]
ax.legend(handles=ax.get_lines())                                            # legend for the two lines only -> [chart 1]
plt.tight_layout()                                                            # avoids cut-off labels -> [chart 1]
plt.show()                                                                    # shows the chart -> [chart 1]

# %% Slide 4 - chart 2: bookings per market segment (also "mostly online" on slide 3)
du_segment = du_raw["market_segment_type"].value_counts(normalize=True) * 100   # % of bookings per channel -> [chart 2 + slide 3 "mostly online"]
du_segment = du_segment.sort_values()                                         # smallest first, so the biggest bar is on top -> [chart 2]
ax = du_segment.plot.barh(color="#1E75BE", figsize=(6.6, 2.6))                # horizontal bar chart, one bar per channel -> [chart 2]
for du_i, du_v in enumerate(du_segment):                                      # goes through each bar -> [chart 2]
    ax.text(du_v + 0.5, du_i, f"{du_v:.0f}%" if du_v >= 1 else f"{du_v:.1f}%", va="center")   # writes the % next to the bar -> [chart 2]
ax.set_title("Market segment: most bookings come online (the mode)")          # chart title -> [chart 2]
ax.set_xlabel("% of bookings")                                                # x axis name -> [chart 2]
ax.set_ylabel("")                                                             # no y axis name (the channels are written) -> [chart 2]
ax.set_xlim(0, 75)                                                            # room for the % labels -> [chart 2]
plt.tight_layout()                                                            # avoids cut-off labels -> [chart 2]
plt.show()                                                                    # shows the chart -> [chart 2]

# %% Slide 4 - chart 3: correlation with cancellation, and the note under it
du_binary = du_raw.columns[du_raw.isin([0, 1]).all()]                         # columns that only contain 0 and 1 (yes/no) -> [chart 3: kept out, they are categories]
du_numerical = du_raw.select_dtypes("number").columns.drop(du_binary).drop(du_date_cols)   # the 9 numerical variables -> [chart 3]
du_num = du_raw[du_numerical].copy()                                          # table with only these 9 variables -> [chart 3]
du_num["cancelled"] = du_cancelled                                            # adds the target as 1/0 so it can be correlated -> [chart 3]
du_corr = du_num.corr()["cancelled"].drop("cancelled").sort_values()         # correlation of each variable with cancellation -> [chart 3]
du_colors = ["#1E75BE" if du_v > 0 else "#3F4A55" for du_v in du_corr]       # blue = more cancellations, dark = fewer -> [chart 3]
ax = du_corr.plot.barh(color=du_colors, figsize=(6.4, 4.9))                  # one bar per variable -> [chart 3]
ax.axvline(0, color="black", linewidth=0.8)                                  # zero line -> [chart 3]
for du_i, du_v in enumerate(du_corr):                                         # goes through each bar -> [chart 3]
    ax.text(du_v + (0.01 if du_v >= 0 else -0.01), du_i, f"{du_v:.2f}",
            va="center", ha="left" if du_v >= 0 else "right", fontsize=8)     # writes the value next to the bar -> [chart 3: 0.44 ... -0.25]
ax.set_title("Correlation with cancellation")                                 # chart title -> [chart 3]
ax.set_xlabel("correlation (-1 to 1)")                                        # x axis name -> [chart 3]
ax.set_xlim(-0.5, 0.6)                                                        # room for the value labels -> [chart 3]
plt.tight_layout()                                                            # avoids cut-off labels -> [chart 3]
plt.show()                                                                    # shows the chart -> [chart 3]
du_inputs = du_raw[du_numerical].corr().abs()                                # correlation between every pair of the 9 inputs (sign removed) -> [note under chart 3]
print("Highest correlation between two inputs:", round(du_inputs.where(du_inputs < 1).max().max(), 2))   # ignores the 1s (a variable with itself) -> [note "highest 0.47"]
