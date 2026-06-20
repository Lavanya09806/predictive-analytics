import os
import json

def create_eda_notebook(filepath):
    """
    Create a beautifully structured Jupyter Notebook for Exploratory Data Analysis.
    """
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Exploratory Data Analysis (EDA)\n",
                "This notebook guides you through the Exploratory Data Analysis phase of the Predictive Analytics project. We cover data loading, automatic type detection, cleaning, univariate/bivariate distributions, correlation analysis, and time-series trend analysis."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Set plot styling\n",
                "sns.set_theme(style=\"whitegrid\")\n",
                "plt.rcParams[\"figure.figsize\"] = (12, 6)\n",
                "plt.rcParams[\"font.size\"] = 12"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Load Data\n",
                "We will load our default stock market or synthetic dataset. If the dataset does not exist, we will use our helper to download/generate it."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.data_preprocessing import load_data, detect_dataset_type, generate_default_timeseries\n",
                "\n",
                "data_path = '../data/raw/default_stock_data.csv'\n",
                "if not os.path.exists(data_path):\n",
                "    df_raw = generate_default_timeseries(data_path)\n",
                "else:\n",
                "    df_raw = load_data(data_path)\n",
                "\n",
                "print(f\"Dataset shape: {df_raw.shape}\")\n",
                "df_raw.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Auto-Detect Dataset Type\n",
                "We check if the dataset is Time Series or standard tabular Regression based on date columns."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "dataset_type, datetime_col = detect_dataset_type(df_raw)\n",
                "print(f\"Detected Dataset Type: {dataset_type.upper()}\")\n",
                "print(f\"Detected Datetime Column: {datetime_col}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Data Cleaning\n",
                "Clean the dataset by removing duplicates, handling missing values, and capping outliers."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.data_preprocessing import clean_data\n",
                "\n",
                "df_cleaned = clean_data(df_raw, datetime_col=datetime_col)\n",
                "print(f\"Cleaned Dataset shape: {df_cleaned.shape}\")\n",
                "df_cleaned.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Univariate Analysis\n",
                "Examine distributions of numeric features using histograms and boxplots."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Find a numeric column to plot (e.g. target Close or Sales)\n",
                "target_candidates = ['Close', 'Sales', df_cleaned.columns[0]]\n",
                "target_col = next((c for c in target_candidates if c in df_cleaned.columns), df_cleaned.columns[0])\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(15, 6))\n",
                "\n",
                "# Histogram & Density\n",
                "sns.histplot(df_cleaned[target_col], kde=True, ax=axes[0], color='royalblue')\n",
                "axes[0].set_title(f'Distribution of {target_col}')\n",
                "\n",
                "# Boxplot to see outliers\n",
                "sns.boxplot(y=df_cleaned[target_col], ax=axes[1], color='coral')\n",
                "axes[1].set_title(f'Boxplot of {target_col}')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Bivariate Analysis\n",
                "Examine relationships between columns. Plot correlation heatmap and scatter plots."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Heatmap of correlations\n",
                "numeric_df = df_cleaned.select_dtypes(include=[np.number])\n",
                "sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)\n",
                "plt.title('Correlation Heatmap')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Time Series Specific Analysis (Trend & Rolling Stats)\n",
                "If dataset is Time Series, plot trend and rolling averages."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if dataset_type == 'time_series':\n",
                "    df_cleaned['Rolling_Mean_30'] = df_cleaned[target_col].rolling(window=30).mean()\n",
                "    df_cleaned['Rolling_Std_30'] = df_cleaned[target_col].rolling(window=30).std()\n",
                "    \n",
                "    plt.figure(figsize=(14, 7))\n",
                "    plt.plot(df_cleaned[target_col], label='Original Close Price', alpha=0.5)\n",
                "    plt.plot(df_cleaned['Rolling_Mean_30'], label='30-Day Rolling Mean', color='red', linewidth=2)\n",
                "    plt.plot(df_cleaned['Rolling_Std_30'], label='30-Day Rolling Std', color='green', linestyle='--')\n",
                "    plt.title(f'{target_col} Price Trend & Rolling Statistics')\n",
                "    plt.legend()\n",
                "    plt.show()\n",
                "else:\n",
                "    print(\"Dataset is classified as standard Regression, skipping time-series rolling plots.\")"
            ]
        }
    ]
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(notebook, f, indent=2)
    print(f"Created EDA notebook at {filepath}")

def create_model_training_notebook(filepath):
    """
    Create a beautifully structured Jupyter Notebook for Model Training and Evaluation.
    """
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Model Training, Tuning, and Evaluation\n",
                "This notebook implements feature engineering, train-test splitting, hyperparameter tuning, model training (traditional models, ML regressors, deep learning), and automated best-model selection."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import joblib\n",
                "\n",
                "from src.data_preprocessing import load_data, clean_data, detect_dataset_type, preprocess_and_scale\n",
                "from src.feature_engineering import engineer_features\n",
                "from src.train_model import train_regression_model, train_traditional_time_series, train_lstm_gru, save_pipeline\n",
                "from src.evaluate import evaluate_model, compile_comparison, analyze_residuals"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Load, Preprocess, and Engineer Features"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "raw_path = '../data/raw/default_stock_data.csv'\n",
                "df_raw = load_data(raw_path)\n",
                "dataset_type, datetime_col = detect_dataset_type(df_raw)\n",
                "df_clean = clean_data(df_raw, datetime_col=datetime_col)\n",
                "\n",
                "# Identify target column\n",
                "target_col = 'Close' if 'Close' in df_clean.columns else df_clean.columns[-1]\n",
                "is_ts = (dataset_type == 'time_series')\n",
                "\n",
                "# Generate features\n",
                "df_feat = engineer_features(df_clean, target_col=target_col, is_time_series=is_ts)\n",
                "print(f\"Feature matrix shape: {df_feat.shape}\")\n",
                "df_feat.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Split Dataset and Standardize Features"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# For time series, split chronologically; for regression, split randomly\n",
                "if is_ts:\n",
                "    train_size = int(len(df_feat) * 0.8)\n",
                "    df_train = df_feat.iloc[:train_size]\n",
                "    df_test = df_feat.iloc[train_size:]\n",
                "else:\n",
                "    from sklearn.model_selection import train_test_split\n",
                "    df_train, df_test = train_test_split(df_feat, test_size=0.2, random_state=42)\n",
                "\n",
                "# Preprocess and scale features\n",
                "X_train, y_train, encoders, scaler = preprocess_and_scale(df_train, target_col=target_col, scaling_method='standard')\n",
                "X_test, y_test, _, _ = preprocess_and_scale(df_test, target_col=target_col, encoders=encoders, scaler=scaler)\n",
                "\n",
                "print(f\"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Model Training & Evaluation\n",
                "We will train Random Forest and XGBoost Regressors (common to both dataset types) and evaluate their performance."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "models_to_test = ['RandomForest', 'XGBoost']\n",
                "results = {}\n",
                "trained_models = {}\n",
                "\n",
                "for m_name in models_to_test:\n",
                "    print(f\"Training {m_name}...\")\n",
                "    # Train model (with tuning)\n",
                "    model = train_regression_model(X_train, y_train, model_name=m_name, tune=True)\n",
                "    trained_models[m_name] = model\n",
                "    \n",
                "    # Predict and evaluate\n",
                "    preds = model.predict(X_test)\n",
                "    metrics = evaluate_model(y_test, preds, is_time_series=is_ts)\n",
                "    results[m_name] = metrics\n",
                "\n",
                "# Display comparison\n",
                "df_compare = compile_comparison(results, is_time_series=is_ts)\n",
                "df_compare"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Residual Analysis of Best Model"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "best_model_name = df_compare.index[0]\n",
                "best_model = trained_models[best_model_name]\n",
                "test_preds = best_model.predict(X_test)\n",
                "\n",
                "residuals, stats = analyze_residuals(y_test, test_preds)\n",
                "print(f\"Best Model: {best_model_name}\")\n",
                "print(f\"Residual Stats: {stats}\")\n",
                "\n",
                "plt.figure(figsize=(10, 5))\n",
                "plt.hist(residuals, bins=30, color='salmon', edgecolor='black', alpha=0.7)\n",
                "plt.axvline(0, color='red', linestyle='--', linewidth=2)\n",
                "plt.title('Residuals Histogram (Distribution of Errors)')\n",
                "plt.xlabel('Error')\n",
                "plt.ylabel('Frequency')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Save the Winning Model\n",
                "We package the best model, the scaler, the encoders, and the features list into a pipeline dictionary, then save it."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "pipeline_dict = {\n",
                "    'model_type': dataset_type,\n",
                "    'model_name': best_model_name,\n",
                "    'model': best_model,\n",
                "    'scaler': scaler,\n",
                "    'encoders': encoders,\n",
                "    'target_col': target_col,\n",
                "    'datetime_col': datetime_col,\n",
                "    'features': list(X_train.columns),\n",
                "    'metrics': results[best_model_name]\n",
                "}\n",
                "\n",
                "save_path = '../models/saved_model.pkl'\n",
                "save_pipeline(pipeline_dict, save_path)"
            ]
        }
    ]
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(notebook, f, indent=2)
    print(f"Created Model Training notebook at {filepath}")

if __name__ == '__main__':
    create_eda_notebook('notebooks/EDA.ipynb')
    create_model_training_notebook('notebooks/Model_Training.ipynb')
