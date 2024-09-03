import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenet.pkl")

predictor_columns = list(df.columns[:6])

# Plot Settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

# Interpolate data
for col in predictor_columns:
    df[col] = df[col].interpolate()

df.info()


# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
set_duration_data = []

for set in df["set"].unique():
    start = df[df["set"] == set].index[0]
    stop = df[df["set"] == set].index[-1]
    duration = stop - start
    df.loc[df["set"] == set, "duration"] = duration.total_seconds()

    # Get the category for this set
    category = df[df["set"] == set]["category"].iloc[0]

    # Append the calculated duration, set, and category to the list
    set_duration_data.append(
        {"category": category, "set": set, "duration": duration.total_seconds()}
    )

# Convert the list to a DataFrame for easier aggregation
duration_df = pd.DataFrame(set_duration_data)
avg_duration_by_category = duration_df.groupby("category")["duration"].mean()

# Duration for each repetition
avg_duration_by_category.iloc[0] / 5  # Heavy set one rep duration
avg_duration_by_category.iloc[1] / 10  # Medium set one rep duration


# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()

fs = 1000 / 200  # 5 instances per second
cutoff = 1.3  # Adjust to see if it improves model

df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)

# Plot to see effects of filter
subset = df_lowpass[df_lowpass["set"] == 20]

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="Raw Data")
ax[1].plot(subset["acc_y"].reset_index(drop=True), label="Butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
plt.show()


# Apply butterworth lowpass to all columns
for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)

    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]


# --------------------------------------------------------------
# Principal component analysis PCA
# Reduce complexity of data and make it easier to analyze or make predictions
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

# Determine optimal amount of principal components (Elbow technique)
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("principle component number")
plt.ylabel("explained variance")
plt.show()

# Observation
# 3 is the optimal number of components

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

# Plot to see effects of pca
subset = df_pca[df_pca["set"] == 20]
subset[["pca_1", "pca_2", "pca_3"]].plot()


# --------------------------------------------------------------
# Sum of squares attributes
# Impartial to device orientation
# --------------------------------------------------------------

df_squared = df_pca.copy()
acc_r = (
    (df_squared["acc_x"] ** 2) + (df_squared["acc_y"] ** 2) + (df_squared["acc_z"] ** 2)
)
gyr_r = (
    (df_squared["gyr_x"] ** 2) + (df_squared["gyr_y"] ** 2) + (df_squared["gyr_z"] ** 2)
)

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

# Plot to see effects of sum of squares
subset = df_squared[df_squared["set"] == 14]
subset[["acc_r", "gyr_r"]].plot(subplots=True)


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()
NumAbs = (
    NumericalAbstraction()
)  # Abstract statistical properties for a given window size

predictor_columns = predictor_columns + ["pca_1", "pca_2", "pca_3", "acc_r", "gyr_r"]

# Trial and error for window size
window_size = int(
    1000 / 200
)  # 1000 miliseconds divide 200ms, we get a window size of 5 for 1 sec

# Run Temporal abstraction based on the subset of sets (Similar exercise)
df_temporal_list = []
for set in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == set].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], window_size, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], window_size, "std")
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)

# visualize
subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()

# --------------------------------------------------------------
# Frequency features
# Decompose original signal into different frequencies
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()
fs = int(1000 / 200)
window_size = int(2800 / 200)  # average length of repetition

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], window_size, fs)
df_freq.columns

# Visualize results
subset = df_freq[df_freq["set"] == 1]
subset[["acc_y"]].plot()
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_0.357_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14",
    ]
].plot()

df_freq_list = []
for set in df_freq["set"].unique():
    print(f"Applying Fourier transform for set {set}")
    subset = df_freq[df_freq["set"] == set].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, window_size, fs)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
# Allow of a certain percentage of overlap

# Drop all missing values
df_freq.dropna(inplace=True)

# Get rid of 50% of rows (Skipping every other row)
df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()
cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = kMeans(k=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
