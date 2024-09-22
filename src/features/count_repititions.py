import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
df = df[df["label"] != "rest"]

# Sum of squares
acc_r = (df["acc_x"] ** 2) + (df["acc_y"] ** 2) + (df["acc_z"] ** 2)
gyr_r = (df["gyr_x"] ** 2) + (df["gyr_y"] ** 2) + (df["gyr_z"] ** 2)

df["acc_r"] = np.sqrt(acc_r)
df["gyr_r"] = np.sqrt(gyr_r)


# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------

bench_df = df[df["label"] == "bench"]
row_df = df[df["label"] == "row"]
ohp_df = df[df["label"] == "ohp"]
squat_df = df[df["label"] == "squat"]
dead_df = df[df["label"] == "dead"]


# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------

plot_df = bench_df
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_r"].plot()

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_r"].plot()

### Observation
# Repitition patterns are clearer for accelration when it comes to bench press


# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------

fs = 1000 / 200
LowPass = LowPassFilter()

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------

bench_set = bench_df[bench_df["set"] == bench_df["set"].unique()[0]]
squat_set = squat_df[squat_df["set"] == squat_df["set"].unique()[0]]
ohp_set = ohp_df[ohp_df["set"] == ohp_df["set"].unique()[0]]
dead_set = dead_df[dead_df["set"] == dead_df["set"].unique()[0]]
row_set = row_df[row_df["set"] == row_df["set"].unique()[0]]

bench_set["acc_r"].plot()
column = "acc_r"
LowPass.low_pass_filter(
    squat_set, col=column, sampling_frequency=fs, cutoff_frequency=0.4, order=10
)[column + "_lowpass"].plot()

squat_set["acc_r_lowpass"].plot()


### Observation
# It's pretty easy to distinguish the peaks of the plot for all the exercises excluding rows
# using the sum of squares for acceleration
# We need to try using other methods for rows


# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------


def count_reps(df, cutoff=0.4, order=10, column="acc_r"):
    fs = 1000 / 200

    # Functions returns the peaks in the signal
    data = LowPass.low_pass_filter(
        df, col=column, sampling_frequency=fs, cutoff_frequency=cutoff, order=order
    )
    index = argrelextrema(data[column + "_lowpass"].values, np.greater)
    peaks = data.iloc[index]

    fig, ax = plt.subplots()
    plt.plot(df[f"{column}_lowpass"])
    plt.plot(peaks[f"{column}_lowpass"], "o", color="red")
    ax.set_ylabel(f"{column}_lowpass")
    exercise = df["label"].iloc[0].title()
    category = df["category"].iloc[0].title()
    plt.title(f"{category} {exercise}: {len(peaks)} Reps")
    plt.show()

    return len(peaks)


# Use function to find best parameters for low pass filter
count_reps(bench_set, cutoff=0.4)
count_reps(squat_set, cutoff=0.35)
count_reps(row_set, cutoff=0.65, column="gyr_x")
count_reps(ohp_set, cutoff=0.35)
count_reps(dead_set, cutoff=0.5)

# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------

df["reps"] = df["category"].apply(lambda x: 5 if x == "heavy" else 10)
rep_df = df.groupby(["label", "category", "set"])["reps"].max().reset_index()

for s in rep_df["set"].unique():
    subset = df[df["set"] == s]
    column = "acc_r"

    if subset["label"].iloc[0] == "bench":
        if subset["label"].iloc[0] == "medium":
            cutoff = 0.565
        if subset["label"].iloc[0] == "heavy":
            cutoff = 0.4
    if subset["label"].iloc[0] == "squat":
        cutoff = 0.35
    if subset["label"].iloc[0] == "row":
        cutoff = 0.65
        column = "gyr_x"
    if subset["label"].iloc[0] == "ohp":
        cutoff = 0.5
        order = 20
    if subset["label"].iloc[0] == "dead":
        cutoff = 0.5

    reps = count_reps(subset, cutoff=cutoff, column=column)
    rep_df.loc[rep_df["set"] == s, "reps_pred"] = reps

rep_df


# --------------------------------------------------------------
# Optimize parameters
# --------------------------------------------------------------
bench_medium = df[(df["label"] == "ohp") & (df["category"] == "medium")]

for set in bench_medium["set"].unique():
    cutoff = 0.5
    order = 20
    column = "acc_r"
    subset = bench_medium[bench_medium["set"] == set]
    count_reps(subset, cutoff=cutoff, column=column, order=order)


# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------

error = mean_absolute_error(rep_df["reps"], rep_df["reps_pred"]).round(2)
rep_df.groupby(["label", "category"])["reps", "reps_pred"].mean().plot.bar()

# Observation
# Mean absolute error of 1, more changing of parameters for each category and label
# would help to decrease the score
