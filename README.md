# Predictive Analytics Using Historical Data

An end-to-end Machine Learning platform that automatically detects whether a dataset is a **Time Series** or standard **Regression** problem, performs preprocessing, engineers advanced features, trains and optimizes multiple models, compares performance metrics, forecasts future trends with error-propagating confidence intervals, and deploys via a professional **Streamlit** dashboard.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Workflow & Architecture](#workflow--architecture)
3. [Key Features](#key-features)
4. [Dataset Description](#dataset-description)
5. [Technologies Used](#technologies-used)
6. [Installation & Setup](#installation--setup)
7. [Usage Guide (CLI & Dashboard)](#usage-guide-cli--dashboard)
8. [Results & Model Comparison](#results--model-comparison)
9. [Future Enhancements](#future-enhancements)

---

## Project Overview

Making decisions based on historical data requires robust feature engineering and modeling. This project provides a production-grade codebase that processes tabular datasets. 

By feeding it a dataset, the system automatically:
- Identifies whether the problem is chronological (Time Series forecasting) or tabular (Regression).
- Applies cleaning (imputation, duplicate removal, IQR outlier capping).
- Customizes feature engineering (lag/rolling features for time series; correlation-based feature interactions for regression).
- Evaluates candidate models (ARIMA, SARIMA, Prophet, Random Forest, XGBoost, LSTM, GRU).
- Selects the best-performing model based on test-set root mean squared error (RMSE) or R² score.
- Saves the entire pre-fitted pipeline (scaler, encoders, model) to disk.
- Exposes predictions through both a Command Line Interface (CLI) and an interactive Streamlit Web App.

---

## Workflow & Architecture

The application is structured as follows:

```text
predictive-analytics/
│
├── data/
│   ├── raw/
│   │   └── default_stock_data.csv    # Raw Apple stock dataset (or synthetic equivalent)
│   └── processed/
│       └── forecast_results.csv     # Saved prediction/forecast output CSV
│
├── notebooks/
│   ├── EDA.ipynb                    # Jupyter Notebook illustrating cleaning & EDA
│   └── Model_Training.ipynb         # Jupyter Notebook showing training & comparisons
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py        # Cleaning, type auto-detection, and scaling
│   ├── feature_engineering.py       # Lags, rolling stats, date extraction & interactions
│   ├── train_model.py               # Model initializers, Keras serialization, tuning
│   ├── predict.py                   # Recursive forecasting & confidence intervals
│   ├── evaluate.py                  # Evaluation metrics & residual analysis
│   └── generate_notebooks.py        # Script to programmatically generate notebooks
│
├── models/
│   └── saved_model.pkl              # Saved best-performing model and pipeline
│
├── visualizations/                  # Directory for saved static plots
│
├── app/
│   └── streamlit_app.py             # Multi-page interactive Streamlit dashboard
│
├── requirements.txt                 # Project dependencies
├── README.md                        # Project documentation (this file)
└── main.py                          # CLI runner for end-to-end training/prediction
```

---

## Key Features

1. **Auto-Detection:** Detects datetime indices or features. If a temporal sequence is found, it activates time-series models; otherwise, standard regression models.
2. **Recursive Multi-step Forecasting:** ML models (Random Forest, XGBoost) require previous predictions as inputs to compute subsequent lags. We implement a custom recursive forecast loop to predict any arbitrary horizon (e.g., 7, 30, 90 days).
3. **Error Propagation Confidence Intervals:** For recursive ML/DL forecasts, prediction error increases over time. The platform models this using:
   $$\hat{y}_t \pm z_{\alpha/2} \times \sigma_{\text{residuals}} \times \sqrt{t}$$
   representing expanding uncertainty.
4. **Resilient Lazy Loading:** Optional packages (`prophet`, `tensorflow`) are imported dynamically. If they are not present, the system issues a warning and runs the remaining models (ARIMA, SARIMA, RF, XGBoost) smoothly.
5. **Interactive Web App:** A multi-page dashboard built using Streamlit and Plotly containing tabs for Dataset Diagnostic Report, Univariate & Bivariate EDA, Model Training, Forecasting, and Residual Analysis.

---

## Dataset Description

The codebase ships with two default datasets (auto-generated or downloaded on the fly if yfinance is available):
- **Stock Market AAPL (Time Series):** Daily stock market pricing containing `Open`, `High`, `Low`, `Close`, `Volume`. The model forecasts `Close_AAPL` price.
- **Advertising vs Sales (Regression):** Tabular regression data detailing `TV_Budget`, `Radio_Budget`, `Newspaper_Budget`, `Region`, `Discount`, and the target variable `Sales`.

Users can also upload any custom CSV or Excel dataset directly through the Streamlit web interface.

---

## Technologies Used

- **Language:** Python 3.8+
- **Data Wrangling:** Pandas, NumPy
- **Visualizations:** Plotly, Seaborn, Matplotlib
- **Machine Learning:** Scikit-Learn, XGBoost, Statsmodels, Prophet (optional)
- **Deep Learning:** TensorFlow / Keras (optional)
- **Dashboard:** Streamlit
- **Serialization:** Joblib, Pickle

---

## Installation & Setup

1. **Clone or Download the Project:**
   Ensure the directory structure matches the layout above.

2. **Set up a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Guide (CLI & Dashboard)

### 1. Command Line Interface (CLI)

The CLI script `main.py` handles model training, comparison, and prediction saving.

*   **Train models on default data (Stock Market):**
    ```bash
    python main.py --mode train
    ```
    This prints a model comparison table, selects the best model, and saves it to `models/saved_model.pkl`.

*   **Train models with Hyperparameter Tuning:**
    ```bash
    python main.py --mode train --tune
    ```

*   **Train on custom dataset:**
    ```bash
    python main.py --mode train --data path/to/dataset.csv --target TargetColumnName
    ```

*   **Forecast next 90 days using the saved model:**
    ```bash
    python main.py --mode predict --days 90 --output data/processed/forecast_results.csv
    ```

### 2. Streamlit Web Dashboard

Launch the interactive web application:
```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser. Navigating through the sidebar allows you to upload custom datasets, run EDA, select target variables, compare training metrics, forecast future values, and analyze residuals.

---

## Results & Model Comparison

Running the time-series forecasting pipeline on Apple stock prices (`Close_AAPL` target) yields the following metrics:

| Model | MAE | RMSE | MAPE (%) | SMAPE (%) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RandomForest** | **10.51** | **14.46** | **4.36%** | **4.42%** | **Selected Best** |
| XGBoost | 10.96 | 15.00 | 4.58% | 4.64% | - |
| SARIMA | 26.76 | 33.19 | 12.64% | 11.48% | - |

*Note: Metrics are evaluated on a 20% validation set (out-of-sample data split chronologically).*

---

## Future Enhancements

- **Deep Learning Support:** Expand TensorFlow architectures to include bidirectional LSTMs, Transformers, and Temporal Fusion Transformers.
- **Multivariate Forecasting:** Enable recursive multivariate forecasting where multiple target variables are forecasted jointly.
- **AutoML Integration:** Integrate automated feature selection algorithms (e.g. Boruta, RFE) and Bayesian hyperparameter tuning.
