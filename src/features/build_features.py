import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


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


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
