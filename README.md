# Fitness Workout Tracker

## Overview

The Fitness Workout Tracker project aims to build a machine learning model to analyze and differentiate between various workouts, including Overhead Press, Bench Press, Row, Squats, and Deadlift. This system is designed for tracking sets, reps, and exercise types across multiple participants, making it easier to monitor and evaluate workout performance.

## Features

- **Workout Differentiation:** Machine learning model to classify different workouts.
- **Participant Tracking:** Data collection and analysis for 5 different participants.
- **Exercise Tracking:** Record sets, reps, and types of exercises performed.
- **Data Visualization:** Insightful charts and reports to visualize workout trends and performance.

## Getting Started

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/fitness-workout-tracker.git
    cd fitness-workout-tracker
    ```

2. Install required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. **Data Preparation:**
   Place your workout data files in the `data/raw` directory. Ensure the data is formatted as expected.

2. **Data Processing:**
   Run the data processing script to clean and prepare the data:
    ```bash
    python scripts/process_data.py
    ```

3. **Model Training:**
   Train the machine learning model using:
    ```bash
    python scripts/train_model.py
    ```

4. **Evaluation:**
   Evaluate the model performance:
    ```bash
    python scripts/evaluate_model.py
    ```

5. **Visualization:**
   Generate visualizations to review workout trends:
    ```bash
    python scripts/visualize_data.py
    ```

### Directory Structure

- `data/`
  - `external/` - External data sources.
  - `interim/` - Intermediate data files.
  - `processed/` - Processed data ready for modeling.
  - `raw/` - Raw data files.
- `scripts/` - Python scripts for data processing, model training, evaluation, and visualization.
- `notebooks/` - Jupyter notebooks for exploratory data analysis and experimentation.
- `requirements.txt` - List of Python packages required for the project.

## Contributing

Contributions are welcome! Please follow the standard GitHub flow:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## Contact

For any questions or feedback, please reach out to [shaunting.ck@gmail.com].

