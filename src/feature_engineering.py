import pandas as pd
import numpy as np

def extract_date_features(df, datetime_col):
    """
    Extract calendar components from a datetime column.
    """
    df_feat = df.copy()
    
    # Check if index is datetime
    if isinstance(df_feat.index, pd.DatetimeIndex):
        dates = df_feat.index
    elif datetime_col in df_feat.columns:
        dates = pd.to_datetime(df_feat[datetime_col])
    else:
        return df_feat
        
    df_feat['Year'] = dates.year
    df_feat['Month'] = dates.month
    df_feat['Day'] = dates.day
    df_feat['Week'] = dates.isocalendar().week.astype(int)
    df_feat['Quarter'] = dates.quarter
    df_feat['DayOfWeek'] = dates.dayofweek
    
    return df_feat

def engineer_time_series_features(df, target_col, lags=[1, 2, 7, 14, 30], windows=[7, 30]):
    """
    Generate lag and rolling features for time-series forecasting.
    Only features based on the target column are kept to allow multi-step forecasting.
    """
    df_feat = df[[target_col]].copy()
    
    # 1. Date features
    df_feat = extract_date_features(df_feat, datetime_col=None)
    
    # 2. Lag features
    for lag in lags:
        df_feat[f'{target_col}_lag_{lag}'] = df_feat[target_col].shift(lag)
        
    # 3. Rolling window statistics
    for w in windows:
        df_feat[f'{target_col}_rolling_mean_{w}'] = df_feat[target_col].rolling(window=w).mean()
        df_feat[f'{target_col}_rolling_std_{w}'] = df_feat[target_col].rolling(window=w).std()
        df_feat[f'{target_col}_rolling_min_{w}'] = df_feat[target_col].rolling(window=w).min()
        df_feat[f'{target_col}_rolling_max_{w}'] = df_feat[target_col].rolling(window=w).max()
        
    # Drop rows with NaN values introduced by shift/rolling
    df_feat = df_feat.dropna()
    
    return df_feat

def engineer_regression_features(df, target_col):
    """
    Generate engineered features for regression models.
    Creates interaction terms and extracts date components if date columns are found.
    """
    df_feat = df.copy()
    
    # If any column is datetime, extract features
    for col in df_feat.columns:
        if col != target_col:
            if pd.api.types.is_datetime64_any_dtype(df_feat[col]) or col.lower() in ['date', 'time', 'timestamp']:
                try:
                    df_feat[col] = pd.to_datetime(df_feat[col], errors='coerce')
                    df_feat = extract_date_features(df_feat, datetime_col=col)
                    df_feat = df_feat.drop(columns=[col]) # Drop the raw date string/object
                except Exception:
                    pass
                    
    # Generate correlation-based interaction terms for numeric features
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
        
    if len(numeric_cols) >= 2:
        # Calculate correlations of numeric columns with target (if target is numeric)
        if pd.api.types.is_numeric_dtype(df_feat[target_col]):
            corrs = df_feat[numeric_cols].apply(lambda x: x.corr(df_feat[target_col])).abs()
            top_features = corrs.nlargest(2).index.tolist()
            
            # Create interaction term (multiplication)
            col1, col2 = top_features[0], top_features[1]
            df_feat[f'{col1}_x_{col2}'] = df_feat[col1] * df_feat[col2]
            
            # Create interaction term (addition/subtraction or ratio if safe)
            if (df_feat[col2] != 0).all():
                df_feat[f'{col1}_div_{col2}'] = df_feat[col1] / (df_feat[col2] + 1e-5)
    
    return df_feat

def engineer_features(df, target_col, is_time_series=False):
    """
    Unified entry point for feature engineering.
    """
    if is_time_series:
        return engineer_time_series_features(df, target_col)
    else:
        return engineer_regression_features(df, target_col)
