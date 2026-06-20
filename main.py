import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data_preprocessing import (
    load_data, detect_dataset_type, clean_data, 
    preprocess_and_scale, generate_default_timeseries, generate_default_regression
)
from src.feature_engineering import engineer_features
from src.train_model import (
    train_regression_model, train_traditional_time_series, 
    train_lstm_gru, save_pipeline, load_pipeline, train_prophet_model
)
from src.evaluate import evaluate_model, compile_comparison, analyze_residuals
from src.predict import forecast_time_series, predict_regression

def run_training(data_path, target_col, tune, model_save_path):
    print("="*60)
    print("STARTING MODEL TRAINING PIPELINE")
    print("="*60)
    
    # 1. Load Data
    if not data_path:
        # Default to stock market time series dataset
        data_dir = os.path.join("data", "raw")
        data_path = os.path.join(data_dir, "default_stock_data.csv")
        if not os.path.exists(data_path):
            generate_default_timeseries(data_path)
            
    print(f"Loading dataset from: {data_path}")
    df = load_data(data_path)
    
    # 2. Auto-Detect Dataset Type
    dataset_type, datetime_col = detect_dataset_type(df)
    print(f"Detected Dataset Type: {dataset_type.upper()}")
    if datetime_col:
        print(f"Datetime column: {datetime_col}")
        
    # 3. Clean Data
    print("Cleaning dataset...")
    df_clean = clean_data(df, datetime_col=datetime_col)
    
    # 4. Determine Target Column
    if not target_col:
        # Default target columns: look for columns containing 'close', 'sales', or 'target' case-insensitively
        close_cols = [c for c in df_clean.columns if 'close' in c.lower()]
        sales_cols = [c for c in df_clean.columns if 'sales' in c.lower()]
        target_cols = [c for c in df_clean.columns if 'target' in c.lower()]
        
        if close_cols:
            target_col = close_cols[0]
        elif sales_cols:
            target_col = sales_cols[0]
        elif target_cols:
            target_col = target_cols[0]
        else:
            # Pick the last numerical column
            num_cols = df_clean.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                target_col = num_cols[-1]
            else:
                target_col = df_clean.columns[-1]
                
    print(f"Target column selected: {target_col}")
    
    # 5. Feature Engineering
    is_ts = (dataset_type == 'time_series')
    print("Engineering features...")
    df_feat = engineer_features(df_clean, target_col=target_col, is_time_series=is_ts)
    print(f"Feature matrix shape: {df_feat.shape}")
    
    # 6. Train/Test Split
    # Split chronologically for time series, randomly for regression
    if is_ts:
        train_size = int(len(df_feat) * 0.8)
        df_train = df_feat.iloc[:train_size]
        df_test = df_feat.iloc[train_size:]
    else:
        # Random split for regression
        df_train, df_test = train_test_split(df_feat, test_size=0.2, random_state=42)
        
    print(f"Train samples: {len(df_train)}, Test samples: {len(df_test)}")
    
    # 7. Scale features
    X_train, y_train, encoders, scaler = preprocess_and_scale(
        df_train, target_col=target_col, scaling_method='standard'
    )
    X_test, y_test, _, _ = preprocess_and_scale(
        df_test, target_col=target_col, encoders=encoders, scaler=scaler
    )
    
    feature_cols = list(X_train.columns)
    
    # 8. Train Candidates
    results = {}
    trained_models = {}
    
    if is_ts:
        # Candidates for Time Series
        # A. Traditional Models (SARIMA on raw target)
        print("Training SARIMA model...")
        try:
            # We train ARIMA/SARIMA on the clean train target (which has datetime index)
            y_train_raw = df_train[target_col]
            sarima_model = train_traditional_time_series(y_train_raw, model_type='sarima')
            trained_models['SARIMA'] = sarima_model
            # Evaluate on test target
            y_test_raw = df_test[target_col]
            sarima_preds = sarima_model.forecast(steps=len(y_test_raw))
            results['SARIMA'] = evaluate_model(y_test_raw, sarima_preds, is_time_series=True)
        except Exception as e:
            print(f"SARIMA training failed: {e}")
            
        # B. Prophet Model
        print("Training Prophet model...")
        try:
            prophet_model = train_prophet_model(df_train, target_col)
            if prophet_model is not None:
                trained_models['Prophet'] = prophet_model
                # Evaluate
                future_df = prophet_model.make_future_dataframe(periods=len(df_test), freq='D')
                forecast_df = prophet_model.predict(future_df)
                prophet_preds = forecast_df['yhat'].values[-len(df_test):]
                results['Prophet'] = evaluate_model(df_test[target_col].values, prophet_preds, is_time_series=True)
        except Exception as e:
            print(f"Prophet training failed: {e}")
            
        # C. ML Models (RF, XGBoost)
        for m_name in ['RandomForest', 'XGBoost']:
            print(f"Training {m_name} (with tuning={tune})...")
            try:
                model = train_regression_model(X_train, y_train, model_name=m_name, tune=tune)
                trained_models[m_name] = model
                preds = model.predict(X_test)
                results[m_name] = evaluate_model(y_test, preds, is_time_series=True)
            except Exception as e:
                print(f"{m_name} training failed: {e}")
                
        # D. Deep Learning (LSTM, GRU)
        for dl_name in ['lstm', 'gru']:
            print(f"Training {dl_name.upper()} model...")
            try:
                # Use split of X_train for validation
                X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
                dl_model_data = train_lstm_gru(X_tr, y_tr, X_val, y_val, model_type=dl_name)
                if dl_model_data is not None:
                    trained_models[dl_name.upper()] = dl_model_data
                    
                    # Predict using helper
                    from src.train_model import predict_keras_model
                    preds = predict_keras_model(dl_model_data, X_test)
                    results[dl_name.upper()] = evaluate_model(y_test, preds, is_time_series=True)
            except Exception as e:
                print(f"{dl_name.upper()} training failed: {e}")
                
    else:
        # Candidates for standard Regression
        regression_candidates = ['LinearRegression', 'Ridge', 'Lasso', 'RandomForest', 'XGBoost']
        for m_name in regression_candidates:
            print(f"Training {m_name} (with tuning={tune})...")
            try:
                model = train_regression_model(X_train, y_train, model_name=m_name, tune=tune)
                trained_models[m_name] = model
                preds = model.predict(X_test)
                results[m_name] = evaluate_model(y_test, preds, is_time_series=False)
            except Exception as e:
                print(f"{m_name} training failed: {e}")
                
    # 9. Compare and select best
    df_compare = compile_comparison(results, is_time_series=is_ts)
    print("\n" + "="*40)
    print("MODEL COMPARISON RESULTS")
    print("="*40)
    print(df_compare.to_string())
    print("="*40)
    
    best_model_name = df_compare.index[0]
    print(f"BEST MODEL SELECTED: {best_model_name}")
    
    best_model = trained_models[best_model_name]
    
    # 10. Save Pipeline
    pipeline_dict = {
        'model_type': dataset_type,
        'model_name': best_model_name,
        'model': best_model,
        'scaler': scaler,
        'encoders': encoders,
        'target_col': target_col,
        'datetime_col': datetime_col,
        'features': feature_cols,
        'metrics': results[best_model_name]
    }
    
    save_pipeline(pipeline_dict, model_save_path)
    print(f"Training completed successfully. Winning model saved to {model_save_path}")
    print("="*60)
    
    return pipeline_dict

def run_prediction(data_path, days, output_path, model_save_path):
    print("="*60)
    print("STARTING PREDICTION / FORECASTING PIPELINE")
    print("="*60)
    
    # 1. Load pipeline
    print(f"Loading saved pipeline from {model_save_path}")
    try:
        pipeline = load_pipeline(model_save_path)
    except FileNotFoundError:
        print(f"Error: No saved model found at {model_save_path}. Run training first.")
        return
        
    model_type = pipeline['model_type']
    model_name = pipeline['model_name']
    target_col = pipeline['target_col']
    datetime_col = pipeline['datetime_col']
    scaler = pipeline['scaler']
    encoders = pipeline['encoders']
    features = pipeline['features']
    
    print(f"Loaded {model_name} model (Type: {model_type.upper()})")
    
    # 2. Load latest data history
    if not data_path:
        if model_type == 'time_series':
            data_path = os.path.join("data", "raw", "default_stock_data.csv")
        else:
            data_path = os.path.join("data", "raw", "default_regression_data.csv")
            if not os.path.exists(data_path):
                generate_default_regression(data_path)
                
    print(f"Loading data from: {data_path}")
    df = load_data(data_path)
    
    # Clean the data
    df_clean = clean_data(df, datetime_col=datetime_col)
    
    if model_type == 'time_series':
        # Generate Forecast
        print(f"Generating recursive time-series forecast for the next {days} days...")
        forecast_df = forecast_time_series(pipeline, df_clean, periods=days, target_col=target_col)
        
        # Save to output path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        forecast_df.to_csv(output_path)
        print(f"Forecast saved to {output_path}")
        print("\nFORECAST PREDICTIONS:")
        print(forecast_df.to_string())
        
    else:
        # Standard tabular regression predictions
        # Preprocess features (drop target if present)
        X_eval, _, _, _ = preprocess_and_scale(
            df_clean, target_col=target_col, encoders=encoders, scaler=scaler
        )
        
        # Predict
        print("Generating regression predictions...")
        preds = predict_regression(pipeline['model'], X_eval[features])
        
        results_df = df_clean.copy()
        results_df['Predicted_Sales'] = preds
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path)
        print(f"Predictions saved to {output_path}")
        print("\nREGRESSION PREDICTIONS (First 10 rows):")
        print(results_df[[target_col, 'Predicted_Sales']].head(10).to_string())
        
    print("="*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="End-to-End Predictive Analytics Pipeline")
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train', help='Execution mode')
    parser.add_argument('--data', type=str, default=None, help='Path to dataset file')
    parser.add_argument('--target', type=str, default=None, help='Name of the target column')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning during training')
    parser.add_argument('--days', type=int, default=30, help='Number of days to forecast (Time Series)')
    parser.add_argument('--output', type=str, default='data/processed/forecast_results.csv', help='Path to save predictions')
    parser.add_argument('--model_path', type=str, default='models/saved_model.pkl', help='Path to save/load model pkl')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        run_training(args.data, args.target, args.tune, args.model_path)
    else:
        run_prediction(args.data, args.days, args.output, args.model_path)
