import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

# View plot for one set
set_df = df[df["set"] == 1]
plt.plot(set_df["acc_y"])

# Set index to number of measurements
plt.plot(set_df["acc_y"].reset_index(drop=True))


# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

# Plot the y acceleration for all exercises
for label in df["label"].unique():
    subset = df[df["label"] == label]

    fig, ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

# Plot the first 100 samples
for label in df["label"].unique():
    subset = df[df["label"] == label]

    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()


# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------

mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100


# --------------------------------------------------------------
# Compare medium vs heavy sets
# --------------------------------------------------------------

category_df = df.query("label == 'ohp'").query("participant == 'A'").reset_index()

# Create the plot
fig, ax = plt.subplots()

category_df.groupby("category")["acc_y"].plot()

# Set labels and show legend
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend(title="Category")
plt.show()

# Observations
# For squat, the acceleration for the y axis is higher for light weights
# In general medium weighs have more variance in acc_y,


# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

participant_df = (
    df.query("label == 'squat'")
    .query("category == 'heavy'")
    .sort_values("participant")
    .reset_index()
)

# Create the plot
fig, ax = plt.subplots()
participant_df.groupby("participant")["acc_y"].plot()

# Set labels and show legend
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend(title="Participants")
plt.show()

# Observations
# For bench, participants have similar acc_y
# Squat, participant D has different acc_y

df.groupby(["label", "participant"])["category"].unique()


# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

label = "squat"
participant = "A"
all_axis_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
)

# Create the plot
fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot()

# Set labels and show legend
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend(title="Participants")
plt.show()


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------\
df.sort_values(["label", "participant"], inplace=True)
labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )
        if len(all_axis_df) > 0:
            # Create the plot
            fig, ax = plt.subplots()
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)

            # Set labels and show legend
            ax.set_ylabel("acc_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} (Participant {participant})".title())
            plt.show()

# Gyroscope: Orientation based on earth
for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )
        if len(all_axis_df) > 0:
            # Create the plot
            fig, ax = plt.subplots()
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)

            # Set labels and show legend
            ax.set_ylabel("gyr_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} (Participant {participant})".title())
            plt.show()


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------
label = "squat"
participant = "A"
combined_plot_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
)

# Create the plot
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))

combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])


# Set labels and show legend
ax[0].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)

ax[1].set_xlabel("samples")


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------
for label in labels:
    for participant in participants:
        combined_plot_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )
        if len(combined_plot_df) > 0:
            # Create the plot
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

            # Set title for the first plot
            ax[0].set_title(
                f"Activity: {label}, Participant: {participant}",
                fontsize=16,
                loc="left",
                pad=20,
            )

            # Set labels and show legend
            ax[0].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )

            ax[0].set_ylabel("Accelerometer")
            ax[1].set_ylabel("Gyroscope")
            ax[1].set_xlabel("samples")

            plt.savefig(f"../../reports/figures/{label.title()} ({participant}).png")
            plt.show()
