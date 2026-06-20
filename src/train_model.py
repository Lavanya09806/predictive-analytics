import os
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

# Suppress warnings
warnings.filterwarnings('ignore')

# Lazy loading helpers for TensorFlow and Prophet
def train_lstm_gru(X_train, y_train, X_val, y_val, model_type='lstm', epochs=20, batch_size=32):
    """
    Train a simple LSTM or GRU model using TensorFlow/Keras.
    Returns a model wrapper containing weights and architecture JSON, or None if TensorFlow is missing.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LSTM, GRU, Input
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        print("TensorFlow/Keras is not installed. Skipping Deep Learning models.")
        return None

    # Reshape input to 3D: [samples, time_steps, features]
    # Here, time_steps = 1 because we are using engineered lag features in a tabular layout
    X_t = np.expand_dims(X_train.values if isinstance(X_train, pd.DataFrame) else X_train, axis=1)
    X_v = np.expand_dims(X_val.values if isinstance(X_val, pd.DataFrame) else X_val, axis=1)
    
    input_shape = (X_t.shape[1], X_t.shape[2])
    
    model = Sequential()
    model.add(Input(shape=input_shape))
    if model_type == 'lstm':
        model.add(LSTM(32, activation='relu', return_sequences=False))
    else:
        model.add(GRU(32, activation='relu', return_sequences=False))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(
        X_t, y_train,
        validation_data=(X_v, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Store Keras model weights and architecture JSON for pickle-safe storage
    model_data = {
        'type': model_type,
        'config': model.to_json(),
        'weights': model.get_weights(),
        'feature_cols': list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None
    }
    return model_data

def predict_keras_model(model_data, X):
    """
    Reconstruct Keras model from weights and config, and make predictions.
    """
    import tensorflow as tf
    from tensorflow.keras.models import model_from_json
    
    model = model_from_json(model_data['config'])
    model.set_weights(model_data['weights'])
    
    X_3d = np.expand_dims(X.values if isinstance(X, pd.DataFrame) else X, axis=1)
    preds = model.predict(X_3d, verbose=0).flatten()
    return preds

def train_prophet_model(df_train, target_col):
    """
    Train a Prophet model.
    """
    try:
        from prophet import Prophet
    except ImportError:
        print("Prophet is not installed. Skipping Prophet model.")
        return None
        
    # Prophet requires ds (datestamp) and y (target) columns
    prophet_df = pd.DataFrame({
        'ds': df_train.index,
        'y': df_train[target_col]
    }).reset_index(drop=True)
    
    # Suppress output logging from cmdstanpy
    import logging
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_df)
    return model

def predict_prophet_model(model, periods):
    """
    Generate future predictions using Prophet.
    """
    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)
    # Return forecasted values corresponding to future period
    return forecast.tail(periods)

def train_traditional_time_series(y_train, order=(1,1,1), seasonal_order=(1,1,1,7), model_type='arima'):
    """
    Train ARIMA or SARIMA model on the endogenous variable.
    """
    if model_type == 'arima':
        model = ARIMA(y_train, order=order)
    else:
        model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    
    fitted_model = model.fit(disp=False)
    return fitted_model

def get_tuning_grid(model_name):
    """
    Return parameter grids for hyperparameter tuning.
    """
    if model_name == 'RandomForest':
        return {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    elif model_name == 'XGBoost':
        return {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
    elif model_name in ['Ridge', 'Lasso']:
        return {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        }
    return {}

def train_regression_model(X_train, y_train, model_name='LinearRegression', tune=False):
    """
    Train standard regression models with optional hyperparameter tuning.
    """
    if model_name == 'LinearRegression':
        model = LinearRegression()
    elif model_name == 'Ridge':
        model = Ridge()
    elif model_name == 'Lasso':
        model = Lasso()
    elif model_name == 'RandomForest':
        model = RandomForestRegressor(random_state=42)
    elif model_name == 'XGBoost':
        model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
    else:
        raise ValueError(f"Unknown regression model: {model_name}")
        
    if tune and model_name in ['Ridge', 'Lasso', 'RandomForest', 'XGBoost']:
        param_grid = get_tuning_grid(model_name)
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=5,
            cv=3,
            random_state=42,
            n_jobs=-1,
            scoring='neg_mean_squared_error'
        )
        search.fit(X_train, y_train)
        return search.best_estimator_
    else:
        model.fit(X_train, y_train)
        return model

def save_pipeline(model_dict, filepath):
    """
    Save the trained model and preprocessing pipeline components.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model_dict, filepath)
    print(f"Saved pipeline to {filepath}")

def load_pipeline(filepath):
    """
    Load the saved model and pipeline components.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No saved pipeline found at {filepath}")
    return joblib.load(filepath)
