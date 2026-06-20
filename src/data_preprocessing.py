import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

def load_data(file_path):
    """
    Load data from a CSV or Excel file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

def detect_dataset_type(df):
    """
    Automatically detect whether a dataset is Time Series or Regression.
    Returns 'time_series' if a date/time column is detected, otherwise 'regression'.
    Also returns the name of the detected datetime column.
    """
    datetime_col = None
    
    # Check column names and data types for datetime patterns
    for col in df.columns:
        # Check if the column is already datetime type
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_col = col
            break
        # Try parsing strings to datetime if column name contains time-related terms
        col_lower = col.lower()
        if any(term in col_lower for term in ['date', 'time', 'timestamp', 'year', 'month', 'day']):
            try:
                temp_parsed = pd.to_datetime(df[col], errors='coerce')
                # If at least 70% of the column parses to valid dates, we consider it a datetime column
                if temp_parsed.notna().sum() / len(df) > 0.7:
                    datetime_col = col
                    break
            except Exception:
                pass

    if datetime_col is not None:
        return 'time_series', datetime_col
    return 'regression', None

def clean_data(df, datetime_col=None):
    """
    Clean dataset: remove duplicates, handle missing values, cap outliers, and fix inconsistencies.
    """
    df_clean = df.copy()
    
    # 1. Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    # If time series, sort by datetime column and convert to datetime index
    if datetime_col and datetime_col in df_clean.columns:
        df_clean[datetime_col] = pd.to_datetime(df_clean[datetime_col], errors='coerce')
        # Drop rows where date could not be parsed
        df_clean = df_clean.dropna(subset=[datetime_col])
        df_clean = df_clean.sort_values(by=datetime_col)
        df_clean.set_index(datetime_col, inplace=True)
    
    # Identify numeric and categorical columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns
    
    # 2. Handle missing values
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            # Use median for numerical columns
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            # Use mode for categorical columns
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col] = df_clean[col].fillna(mode_val[0])
            else:
                df_clean[col] = df_clean[col].fillna("Unknown")
                
    # 3. Detect and handle outliers in numeric columns (using IQR method)
    # Only cap outliers for continuous numeric columns with more than 10 unique values
    for col in numeric_cols:
        if df_clean[col].nunique() > 10:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Cap the values
            df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
            
    # 4. Clean inconsistent text data (strip spaces, lowercase categorical values)
    for col in categorical_cols:
        if df_clean[col].dtype == object:
            df_clean[col] = df_clean[col].astype(str).str.strip()
            
    return df_clean

def preprocess_and_scale(df, target_col, scaling_method='standard', encoders=None, scaler=None):
    """
    Encode categorical features and scale numeric features.
    If encoders/scaler are provided, they are reused (important for validation/test sets).
    """
    df_transformed = df.copy()
    
    # Separate target variable from features
    if target_col in df_transformed.columns:
        y = df_transformed[target_col]
        X = df_transformed.drop(columns=[target_col])
    else:
        y = None
        X = df_transformed

    # Identify features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Encode categorical features
    if encoders is None:
        encoders = {}
        fit_encoders = True
    else:
        fit_encoders = False
        
    for col in categorical_features:
        if fit_encoders:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is not None:
                # Handle unseen labels by mapping them to the first label if seen in fit
                classes = dict(zip(le.classes_, le.transform(le.classes_)))
                X[col] = X[col].astype(str).map(classes).fillna(0).astype(int)
            else:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
                
    # Scale numeric features
    if len(numeric_features) > 0:
        if scaler is None:
            if scaling_method == 'standard':
                scaler = StandardScaler()
            elif scaling_method == 'minmax':
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            X[numeric_features] = scaler.fit_transform(X[numeric_features])
        else:
            X[numeric_features] = scaler.transform(X[numeric_features])
            
    return X, y, encoders, scaler

def generate_default_timeseries(file_path):
    """
    Download real AAPL stock data using yfinance if possible;
    otherwise generate high-quality synthetic time series and save it to file_path.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Try downloading AAPL using yfinance
    try:
        import yfinance as yf
        print("Attempting to download default Stock dataset (AAPL) using yfinance...")
        # Get AAPL history for 5 years
        df = yf.download("AAPL", start="2021-01-01", end="2026-01-01")
        if not df.empty:
            df.reset_index(inplace=True)
            # Flatten multi-index columns if present (can occur in newer yfinance versions)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in df.columns]
            
            # Ensure the primary Date column is named 'Date'
            if 'Date' in df.columns:
                pass
            elif 'index' in df.columns:
                df.rename(columns={'index': 'Date'}, inplace=True)
            else:
                df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
                
            # Rename closing column for standard prediction
            df.to_csv(file_path, index=False)
            print(f"Downloaded and saved stock data to {file_path}. Shape: {df.shape}")
            return df
    except Exception as e:
        print(f"yfinance download failed ({e}). Generating realistic synthetic Stock/Time Series dataset instead...")
        
    # Synthetic time-series generation
    np.random.seed(42)
    num_samples = 1200  # Approx 3.3 years of daily data or 5 years of business days
    date_range = pd.date_range(start="2022-01-01", periods=num_samples, freq="B") # Business days
    
    # Trend
    trend = np.linspace(150, 280, num_samples)
    # Seasonality: weekly (5 days) and yearly (252 business days)
    weekly_season = 4 * np.sin(2 * np.pi * date_range.dayofweek / 5)
    yearly_season = 12 * np.sin(2 * np.pi * date_range.dayofyear / 252)
    # Random walk with drift + noise
    noise = np.random.normal(0, 2, num_samples)
    random_walk = np.cumsum(np.random.normal(0.05, 0.8, num_samples))
    
    prices = trend + weekly_season + yearly_season + random_walk + noise
    # Ensure prices are positive
    prices = np.clip(prices, 50, None)
    
    df = pd.DataFrame({
        'Date': date_range,
        'Close': np.round(prices, 2),
        'Open': np.round(prices - np.random.uniform(-3, 3, num_samples), 2),
        'High': np.round(prices + np.random.uniform(0.5, 5, num_samples), 2),
        'Low': np.round(prices - np.random.uniform(0.5, 5, num_samples), 2),
        'Volume': np.random.randint(1000000, 5000000, num_samples)
    })
    
    df.to_csv(file_path, index=False)
    print(f"Generated synthetic time series and saved to {file_path}. Shape: {df.shape}")
    return df

def generate_default_regression(file_path):
    """
    Generate synthetic regression dataset (Advertising spend vs Product Sales) and save it to file_path.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    np.random.seed(42)
    num_samples = 1000
    
    tv_ad = np.random.uniform(10, 300, num_samples)
    radio_ad = np.random.uniform(5, 100, num_samples)
    newspaper_ad = np.random.uniform(0, 50, num_samples)
    region = np.random.choice(['North', 'East', 'South', 'West'], num_samples)
    discount_level = np.random.choice(['Low', 'Medium', 'High'], num_samples)
    
    # Interaction terms and categorical coefficients
    region_coeffs = {'North': 5.0, 'East': 8.5, 'South': -2.0, 'West': 12.0}
    discount_coeffs = {'Low': 0.0, 'Medium': 15.0, 'High': 35.0}
    
    base_sales = 50.0
    sales = (
        base_sales +
        0.35 * tv_ad +
        0.75 * radio_ad +
        0.05 * newspaper_ad +
        0.002 * tv_ad * radio_ad +  # interaction
        np.array([region_coeffs[r] for r in region]) +
        np.array([discount_coeffs[d] for d in discount_level]) +
        np.random.normal(0, 10, num_samples)  # noise
    )
    
    # Ensure sales are positive
    sales = np.clip(sales, 10, None)
    
    df = pd.DataFrame({
        'TV_Budget': np.round(tv_ad, 2),
        'Radio_Budget': np.round(radio_ad, 2),
        'Newspaper_Budget': np.round(newspaper_ad, 2),
        'Region': region,
        'Discount': discount_level,
        'Sales': np.round(sales, 2)
    })
    
    df.to_csv(file_path, index=False)
    print(f"Generated synthetic regression dataset and saved to {file_path}. Shape: {df.shape}")
    return df
