# 🎬 Audience Demand Prediction System

## 📊 Time-Series Forecasting for Theatre Audience Counts

A machine learning project that combines theatre metadata, online booking activity, point-of-sale ticket data, and historical audience visits to predict daily audience counts for individual theatres. The notebook covers data integration, exploratory analysis, time-based feature engineering, model comparison, and generation of a prediction file.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Data-Pandas%20%7C%20NumPy-150458)
![Scikit--learn](https://img.shields.io/badge/ML-Scikit--learn-F7931E)
![Forecasting](https://img.shields.io/badge/Task-Time%20Series%20Forecasting-2E8B57)

---

## 📋 Table of Contents

- [🎬 Audience Demand Prediction System](#-audience-demand-prediction-system)
  - [📊 Time-Series Forecasting for Theatre Audience Counts](#-time-series-forecasting-for-theatre-audience-counts)
  - [📋 Table of Contents](#-table-of-contents)
  - [📌 Project Overview](#-project-overview)
    - [Key objectives](#key-objectives)
  - [🗃️ Dataset Description](#️-dataset-description)
  - [🔄 Workflow](#-workflow)
  - [🧠 Feature Engineering](#-feature-engineering)
  - [📈 Exploratory Data Analysis](#-exploratory-data-analysis)
  - [🤖 Models and Evaluation](#-models-and-evaluation)
    - [Evaluation](#evaluation)
  - [🔮 Prediction Output](#-prediction-output)
  - [🛠️ Tech Stack](#️-tech-stack)
  - [⚙️ Installation and Setup](#️-installation-and-setup)
    - [✅ Prerequisites](#-prerequisites)
    - [➡️ Steps](#️-steps)
  - [📁 Project Structure](#-project-structure)
  - [📌 Notes](#-notes)
  - [📄 License](#-license)

## 📌 Project Overview

The goal is to forecast the **daily audience count for each theatre** using historical attendance, booking activity, theatre characteristics, and calendar patterns.

The project integrates data from two platforms:

- **BookNow** — online booking and theatre information.
- **CinePOS** — point-of-sale booking activity and theatre information.

These sources are combined with historical audience visits and calendar information to build a unified dataset for regression modelling.

### Key objectives

- Combine booking, theatre, and audience datasets into a consistent modelling table.
- Explore audience distributions, trends, seasonality, theatre-level differences, and geographic patterns.
- Engineer calendar, historical-lag, and rolling-statistic features.
- Compare multiple tree-based regression algorithms using a validation set.
- Fit a final model and generate predictions in the required submission format.

## 🗃️ Dataset Description

The notebook expects the following CSV files to be available in the working directory:

| File | Description |
|---|---|
| `cinePOS_theaters.csv` | Theatre metadata from CinePOS |
| `booknow_theaters.csv` | Theatre metadata from BookNow |
| `movie_theater_id_relation.csv` | Mapping between BookNow and CinePOS theatre IDs |
| `cinePOS_booking.csv` | Booking records from CinePOS |
| `booknow_booking.csv` | Booking records from BookNow |
| `booknow_visits.csv` | Historical daily audience counts |
| `date_info.csv` | Calendar/date information |
| `sample_submission.csv` | Required prediction output format |

**Target variable:** `audience_count`  
**Prediction key:** `ID`, containing theatre and show-date information.

## 🔄 Workflow

1. **Load the data** — read the theatre metadata, booking records, audience visits, date information, and sample submission.
2. **Integrate data sources** — map theatre identifiers and aggregate booking activity to the theatre/date level.
3. **Explore the data** — inspect distributions, trends, correlations, theatre-level audience patterns, and geographic distribution.
4. **Engineer features** — create calendar variables, cyclical date encodings, historical lags, and rolling statistics.
5. **Preprocess features** — impute missing values and encode categorical variables.
6. **Train and compare models** — evaluate several tree-based regression approaches on a validation set using the R² score.
7. **Build predictions** — prepare test features, apply the fitted preprocessing pipeline, predict audience counts, and clip negative predictions to zero.
8. **Export results** — write predictions to `submission.csv`.

## 🧠 Feature Engineering

The notebook creates several groups of features to capture different demand patterns.

- **Calendar features:** year, month, day, weekday, weekend indicator, week of year, quarter, week of month, month boundaries, and day of year.
- **Cyclical features:** sine and cosine transformations for day-of-year and weekday to represent recurring calendar patterns.
- **Lag features:** prior audience counts at 1, 7, 14, and 30-day intervals.
- **Rolling features:** trailing means over 3, 7, 14, and 30 observations.
- **Variability features:** rolling standard deviations and coefficients of variation.
- **Theatre features:** theatre type, area, latitude, longitude, and theatre identifiers.
- **Booking activity:** aggregated booking/ticket activity from the available booking platforms.

The lag and rolling calculations use shifted historical values so that the current observation is not included in its own historical summary.

## 📈 Exploratory Data Analysis

The notebook includes visual analysis of:

- Audience-count distribution.
- Average audience count across discrete categories.
- Audience trends for high-volume theatres.
- Correlations between numerical features.
- Aggregate audience counts over time.
- Geographic distribution of theatres.

These analyses are used to investigate changes in audience volume, temporal patterns, and differences between theatres before modelling.

## 🤖 Models and Evaluation

The notebook compares the following regression models:

| Model | Approach |
|---|---|
| Decision Tree Regressor | Single tree baseline |
| Random Forest Regressor | Bagging ensemble of decision trees |
| Extra Trees Regressor | Randomized tree ensemble |
| Gradient Boosting Regressor | Sequential boosting |
| HistGradientBoosting Regressor | Histogram-based gradient boosting |
| XGBoost Regressor | Gradient-boosted trees |
| LightGBM Regressor | Gradient boosting using leaf-wise tree growth |

Categorical preprocessing uses target encoding for the high-cardinality theatre identifier and ordinal encoding for theatre type and area. The remaining features are passed through the column transformer. Numerical features are not scaled because the selected models are tree-based.

### Evaluation

- **Validation metric:** R² score.
- **Model selection:** compare validation scores and identify the strongest-performing candidate.
- **Final training:** refit the selected final configuration on the full processed training dataset before producing test predictions.

The notebook's written findings identify LightGBM as the top model in the initial validation comparison, with HistGradientBoosting performing closely. The final prediction section explicitly fits a `RandomForestRegressor` configuration on the full processed dataset. These are distinct steps in the notebook; the final fitted model should not be assumed to be the same as the highest-scoring initial validation model.

## 🔮 Prediction Output

Running the final prediction cells generates:

```text
submission.csv
```

The output contains the two required columns:

| Column | Description |
|---|---|
| `ID` | Theatre/date identifier from the sample submission |
| `audience_count` | Predicted non-negative audience count |

The notebook also plots historical and predicted aggregate daily audience counts together for a visual comparison over time.

## 🛠️ Tech Stack

**Language and data analysis**
- Python
- Pandas
- NumPy

**Visualisation**
- Matplotlib
- Seaborn

**Machine learning and preprocessing**
- Scikit-learn
- XGBoost
- LightGBM

**Development environment**
- Jupyter Notebook

## ⚙️ Installation and Setup

### ✅ Prerequisites

- Python 3.x
- Jupyter Notebook or JupyterLab
- The dataset CSV files listed above

### ➡️ Steps

1. **Clone or download the project** and open a terminal in the project directory.

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   macOS/Linux:

   ```bash
   source .venv/bin/activate
   ```

3. **Install the dependencies:**

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn==1.7.0 xgboost lightgbm jupyter
   ```

4. **Place all required CSV files** in the notebook's working directory, keeping the filenames listed in the Dataset Description section.

5. **Launch Jupyter:**

   ```bash
   jupyter notebook
   ```

6. Open `notebook.ipynb` and run the notebook cells in order, from data loading through prediction generation.

> **Note:** The notebook installs/uses Scikit-learn 1.7.0 for compatibility. If you run it in a hosted notebook environment, install the required packages in that environment and ensure the CSV paths match the notebook's working directory.

## 📁 Project Structure

```text
Audience-Demand-Prediction/
├── notebook.ipynb  # End-to-end analysis and modelling notebook
├── README.md                            # Project documentation
├── cinePOS_theaters.csv                 # Input: CinePOS theatre metadata
├── booknow_theaters.csv                 # Input: BookNow theatre metadata
├── movie_theater_id_relation.csv        # Input: theatre ID mapping
├── cinePOS_booking.csv                  # Input: CinePOS booking records
├── booknow_booking.csv                  # Input: BookNow booking records
├── booknow_visits.csv                   # Input: historical audience counts
├── date_info.csv                        # Input: calendar data
├── sample_submission.csv                # Input: required output template
└── submission.csv                       # Generated predictions
```

## 📌 Notes

- Results depend on the dataset version, preprocessing, package versions, and validation split.
- The notebook contains a hyperparameter-tuning section whose code is commented out; the main model comparison and final prediction steps can be run independently of that optional section.
- The notebook expects the input files to be available locally and writes `submission.csv` to the current working directory.

## 📄 License

This project is licensed under the [GNU General Public License v3.0](LICENSE).