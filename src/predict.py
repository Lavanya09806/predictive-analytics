import numpy as np
import pandas as pd
import warnings
from src.train_model import predict_keras_model

warnings.filterwarnings('ignore')

def predict_regression(model, X_scaled):
    """
    Generate predictions for standard regression models.
    """
    return model.predict(X_scaled)

def get_forecast_confidence_intervals(preds, std_residuals, is_arima=False, arima_result=None, is_prophet=False, prophet_forecast=None):
    """
    Calculate confidence intervals for predictions.
    For ML/DL models, we scale the interval with sqrt(t) to represent error propagation.
    """
    periods = len(preds)
    
    if is_arima and arima_result is not None:
        try:
            # Statsmodels ARIMA/SARIMA forecast results
            forecast_obj = arima_result.get_forecast(steps=periods)
            conf_df = forecast_obj.conf_int(alpha=0.05)
            # Find columns
            lower_col = conf_df.columns[0]
            upper_col = conf_df.columns[1]
            return conf_df[lower_col].values, conf_df[upper_col].values
        except Exception as e:
            print(f"Failed to extract ARIMA confidence intervals ({e}), falling back to residual estimation.")
            
    if is_prophet and prophet_forecast is not None:
        try:
            # Prophet returns yhat_lower and yhat_upper
            # Extract last 'periods' rows
            lower = prophet_forecast['yhat_lower'].values[-periods:]
            upper = prophet_forecast['yhat_upper'].values[-periods:]
            return lower, upper
        except Exception as e:
            print(f"Failed to extract Prophet confidence intervals ({e}), falling back to residual estimation.")
            
    # Fallback / ML/DL models error-propagation confidence intervals
    # CI = pred +- 1.96 * std_err * sqrt(t)
    z_score = 1.96
    lower_bounds = []
    upper_bounds = []
    
    for t in range(1, periods + 1):
        # Error propagates as sqrt(t) for recursive forecasts
        margin_of_error = z_score * std_residuals * np.sqrt(t)
        lower_bounds.append(preds[t-1] - margin_of_error)
        upper_bounds.append(preds[t-1] + margin_of_error)
        
    return np.array(lower_bounds), np.array(upper_bounds)

def forecast_recursive_ml(model_dict, df_history, periods, target_col):
    """
    Perform recursive multi-step forecasting for Machine Learning and Deep Learning models.
    """
    model = model_dict['model']
    scaler = model_dict['scaler']
    encoders = model_dict['encoders']
    features_list = model_dict['features']
    model_name = model_dict['model_name']
    
    # Store the history of target values
    target_history = list(df_history[target_col].values)
    dates_history = list(df_history.index)
    
    # Identify freq of index to generate future dates
    if df_history.index.freq is not None:
        freq = df_history.index.freq
    else:
        # Infer frequency or fallback to Daily
        freq = pd.infer_freq(df_history.index)
        if freq is None:
            freq = 'D'
            
    future_dates = pd.date_range(start=dates_history[-1], periods=periods + 1, freq=freq)[1:]
    
    predictions = []
    
    # We need to recreate the same feature engineering step-by-step
    # Lags used in feature engineering: 1, 2, 7, 14, 30
    # Rolling windows: 7, 30
    lags = [1, 2, 7, 14, 30]
    windows = [7, 30]
    
    for i, next_date in enumerate(future_dates):
        # Create a single-row feature dict
        feat_dict = {}
        
        # 1. Add Date features
        feat_dict['Year'] = next_date.year
        feat_dict['Month'] = next_date.month
        feat_dict['Day'] = next_date.day
        feat_dict['Week'] = int(next_date.isocalendar()[1])
        feat_dict['Quarter'] = (next_date.month - 1) // 3 + 1
        feat_dict['DayOfWeek'] = next_date.dayofweek
        
        # 2. Add Lag features
        for lag in lags:
            feat_dict[f'{target_col}_lag_{lag}'] = target_history[-lag]
            
        # 3. Add Rolling Window statistics
        for w in windows:
            recent_vals = target_history[-w:]
            feat_dict[f'{target_col}_rolling_mean_{w}'] = np.mean(recent_vals)
            feat_dict[f'{target_col}_rolling_std_{w}'] = np.std(recent_vals)
            feat_dict[f'{target_col}_rolling_min_{w}'] = np.min(recent_vals)
            feat_dict[f'{target_col}_rolling_max_{w}'] = np.max(recent_vals)
            
        # Create single row DataFrame
        X_next = pd.DataFrame([feat_dict])
        
        # Ensure correct column ordering matching feature list
        X_next = X_next[features_list]
        
        # Scale features
        # X_next only contains numeric columns (since dates/lags/rolling stats are numeric)
        numeric_features = X_next.select_dtypes(include=[np.number]).columns.tolist()
        if scaler is not None and len(numeric_features) > 0:
            X_next[numeric_features] = scaler.transform(X_next[numeric_features])
            
        # Predict
        if model_name in ['LSTM', 'GRU']:
            pred_val = predict_keras_model(model, X_next)[0]
        else:
            pred_val = model.predict(X_next)[0]
            
        predictions.append(pred_val)
        target_history.append(pred_val)
        
    return pd.DataFrame({
        'Date': future_dates,
        'Predicted_Value': predictions
    }).set_index('Date')

def forecast_time_series(model_dict, df_history, periods, target_col):
    """
    Unified forecasting interface for all Time Series models.
    """
    model_name = model_dict['model_name']
    model = model_dict['model']
    
    if model_name in ['ARIMA', 'SARIMA']:
        # Statsmodels forecast
        preds = model.forecast(steps=periods)
        # Create date index
        if df_history.index.freq is not None:
            freq = df_history.index.freq
        else:
            freq = pd.infer_freq(df_history.index) or 'D'
        future_dates = pd.date_range(start=df_history.index[-1], periods=periods + 1, freq=freq)[1:]
        
        forecast_df = pd.DataFrame({
            'Predicted_Value': preds.values
        }, index=future_dates)
        
        lower, upper = get_forecast_confidence_intervals(
            preds.values, std_residuals=0.0, is_arima=True, arima_result=model
        )
        forecast_df['Lower_CI'] = lower
        forecast_df['Upper_CI'] = upper
        return forecast_df
        
    elif model_name == 'Prophet':
        # Prophet forecast
        forecast_prophet = predict_prophet_model(model, periods)
        future_dates = forecast_prophet['ds'].values
        preds = forecast_prophet['yhat'].values
        
        forecast_df = pd.DataFrame({
            'Predicted_Value': preds
        }, index=future_dates)
        
        lower, upper = get_forecast_confidence_intervals(
            preds, std_residuals=0.0, is_prophet=True, prophet_forecast=forecast_prophet
        )
        forecast_df['Lower_CI'] = lower
        forecast_df['Upper_CI'] = upper
        return forecast_df
        
    else:
        # Machine Learning / Deep Learning
        forecast_df = forecast_recursive_ml(model_dict, df_history, periods, target_col)
        
        # Estimate confidence intervals using RMSE from model evaluation
        rmse = model_dict.get('metrics', {}).get('RMSE', 1.0)
        # If RMSE is close to 0, use std of target as proxy
        if rmse < 1e-4:
            rmse = df_history[target_col].std() or 1.0
            
        lower, upper = get_forecast_confidence_intervals(
            forecast_df['Predicted_Value'].values, std_residuals=rmse
        )
        forecast_df['Lower_CI'] = lower
        forecast_df['Upper_CI'] = upper
        return forecast_df
