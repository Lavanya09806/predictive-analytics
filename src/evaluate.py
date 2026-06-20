import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE) as a percentage.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE) as a percentage.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Avoid division by zero
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100

def evaluate_regression(y_true, y_pred):
    """
    Calculate regression evaluation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE (%)': mape
    }

def evaluate_timeseries(y_true, y_pred):
    """
    Calculate time-series forecasting evaluation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = calculate_mape(y_true, y_pred)
    smape = calculate_smape(y_true, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape,
        'SMAPE (%)': smape
    }

def evaluate_model(y_true, y_pred, is_time_series=False):
    """
    Unified evaluation interface.
    """
    if is_time_series:
        return evaluate_timeseries(y_true, y_pred)
    else:
        return evaluate_regression(y_true, y_pred)

def compile_comparison(model_results, is_time_series=False):
    """
    Compile a dictionary of model names and their metrics into a pandas DataFrame.
    Sorts by RMSE ascending by default.
    """
    df = pd.DataFrame(model_results).T
    if is_time_series:
        # Sort by RMSE (lower is better)
        df = df.sort_values(by='RMSE')
    else:
        # Sort by R2 descending (higher is better)
        df = df.sort_values(by='R2', ascending=False)
    return df

def analyze_residuals(y_true, y_pred):
    """
    Compute residual statistics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuals = y_true - y_pred
    
    stats = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'min': np.min(residuals),
        'max': np.max(residuals),
        'skew': pd.Series(residuals).skew(),
        'kurtosis': pd.Series(residuals).kurt()
    }
    
    return residuals, stats
