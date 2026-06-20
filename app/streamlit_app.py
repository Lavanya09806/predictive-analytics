import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# Add root directory to path to enable src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import (
    load_data, detect_dataset_type, clean_data, 
    preprocess_and_scale, generate_default_timeseries, generate_default_regression
)
from src.feature_engineering import engineer_features
from src.train_model import (
    train_regression_model, train_traditional_time_series, 
    train_lstm_gru, save_pipeline, predict_keras_model, train_prophet_model
)
from src.evaluate import evaluate_model, compile_comparison, analyze_residuals
from src.predict import forecast_time_series, predict_regression

# Setup page config
st.set_page_config(
    page_title="Predictive Analytics Suite",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* Premium typography and colors */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 3rem !important;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B, #6A11CB, #2575FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.25rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border-color: rgba(106, 17, 203, 0.4);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #6A11CB;
    }
    
    /* Center align headers */
    .section-header {
        border-bottom: 2px solid #eaeaea;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
        color: #333333;
        font-weight: 600;
    }
    
    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        .section-header {
            border-bottom: 2px solid #333333;
            color: #ffffff;
        }
        .metric-value {
            color: #FF4B4B;
        }
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE INITIALIZATION -----------------
if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = None
if 'df_clean' not in st.session_state:
    st.session_state['df_clean'] = None
if 'dataset_type' not in st.session_state:
    st.session_state['dataset_type'] = None
if 'datetime_col' not in st.session_state:
    st.session_state['datetime_col'] = None
if 'target_col' not in st.session_state:
    st.session_state['target_col'] = None
if 'pipeline' not in st.session_state:
    st.session_state['pipeline'] = None
if 'df_compare' not in st.session_state:
    st.session_state['df_compare'] = None
if 'forecast_df' not in st.session_state:
    st.session_state['forecast_df'] = None

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/gradient/100/combo-chart.png", width=80)
st.sidebar.markdown("<h2 style='font-weight:800; margin-top:0;'>Predictive Engine</h2>", unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", [
    "🏠 Home",
    "📊 Dataset Overview",
    "📈 EDA Dashboard",
    "⚙️ Model Training",
    "🔮 Forecasting",
    "✅ Prediction Results"
])

# Utility to load default datasets
def load_default_data(choice):
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    if choice == "Stock Market AAPL (Time Series)":
        path = os.path.join(raw_dir, "default_stock_data.csv")
        if not os.path.exists(path):
            with st.spinner("Downloading stock data..."):
                generate_default_timeseries(path)
        df = pd.read_csv(path)
    else:  # Advertising spend (Regression)
        path = os.path.join(raw_dir, "default_regression_data.csv")
        if not os.path.exists(path):
            with st.spinner("Generating regression data..."):
                generate_default_regression(path)
        df = pd.read_csv(path)
        
    st.session_state['df_raw'] = df
    dataset_type, datetime_col = detect_dataset_type(df)
    st.session_state['dataset_type'] = dataset_type
    st.session_state['datetime_col'] = datetime_col
    st.session_state['df_clean'] = clean_data(df, datetime_col=datetime_col)
    
    # Reset downstream outputs
    st.session_state['pipeline'] = None
    st.session_state['df_compare'] = None
    st.session_state['forecast_df'] = None

# ----------------- 1. HOME PAGE -----------------
if page == "🏠 Home":
    st.markdown("<h1 class='main-title'>Predictive Analytics Using Historical Data</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Analyze data, train predictive models, and forecast future trends interactively</p>", unsafe_allow_html=True)
    
    # Banner image or cool graphic
    fig_banner = go.Figure()
    # Draw a futuristic glowing line
    x_banner = np.linspace(0, 10, 100)
    y_banner = np.sin(x_banner) + x_banner * 0.5 + np.random.normal(0, 0.1, 100)
    fig_banner.add_trace(go.Scatter(x=x_banner, y=y_banner, mode='lines+markers', 
                                   line=dict(color='#6A11CB', width=4),
                                   marker=dict(size=6, color='#FF4B4B'),
                                   name='Prediction Trend'))
    fig_banner.update_layout(
        height=250, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_banner, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("<h3 class='section-header'>Project Features</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='card'>
            <h4>📊 Automated Pipeline</h4>
            <p>Upload any tabular dataset. The platform automatically detects whether it is a <b>Time Series</b> or standard <b>Regression</b> problem.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class='card'>
            <h4>⚙️ Advanced Modelling</h4>
            <p>Compare classic methods (ARIMA/SARIMA/Prophet), standard ML Regressors (Random Forest/XGBoost), and Deep Learning (LSTM/GRU).</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class='card'>
            <h4>📈 Interactive Charts</h4>
            <p>Generate future forecasts with confidence intervals, run detailed residual analyses, and visualize feature importances.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### Getting Started")
    st.write("1. Navigate to **Dataset Overview** and upload your CSV/Excel file or choose one of our default datasets.")
    st.write("2. Explore your data using the **EDA Dashboard**.")
    st.write("3. Select your target variable and click train on the **Model Training** page.")
    st.write("4. View future predictions on the **Forecasting** page and analyze errors in **Prediction Results**.")

# ----------------- 2. DATASET OVERVIEW -----------------
elif page == "📊 Dataset Overview":
    st.markdown("<h1 class='main-title'>Dataset Overview</h1>", unsafe_allow_html=True)
    
    st.markdown("### Load Dataset")
    data_source = st.radio("Choose Data Source:", ["Use Default Dataset", "Upload CSV/Excel file"], horizontal=True)
    
    if data_source == "Use Default Dataset":
        default_choice = st.selectbox("Select Default Dataset:", [
            "Stock Market AAPL (Time Series)",
            "Advertising vs Sales (Regression)"
        ])
        
        # Check if we should initialize default data
        if st.session_state['df_raw'] is None or st.button("Load/Reset Selected Default Dataset"):
            load_default_data(default_choice)
            
    else:
        uploaded_file = st.file_uploader("Upload your dataset file (CSV, XLS, or XLSX):", type=['csv', 'xls', 'xlsx'])
        if uploaded_file is not None:
            try:
                # Save uploaded file
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                st.session_state['df_raw'] = df
                dataset_type, datetime_col = detect_dataset_type(df)
                st.session_state['dataset_type'] = dataset_type
                st.session_state['datetime_col'] = datetime_col
                st.session_state['df_clean'] = clean_data(df, datetime_col=datetime_col)
                
                # Reset downstream outputs
                st.session_state['pipeline'] = None
                st.session_state['df_compare'] = None
                st.session_state['forecast_df'] = None
                
                st.success("File uploaded and parsed successfully!")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                
    # Display details if dataset loaded
    df_raw = st.session_state['df_raw']
    df_clean = st.session_state['df_clean']
    
    if df_raw is not None:
        st.markdown("<h3 class='section-header'>Data Diagnostic Report</h3>", unsafe_allow_html=True)
        
        # Indicators
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Raw Rows", df_raw.shape[0])
        with col2:
            st.metric("Features", df_raw.shape[1])
        with col3:
            st.metric("Auto-Detected Type", st.session_state['dataset_type'].upper())
        with col4:
            st.metric("Datetime Column", str(st.session_state['datetime_col']))
            
        st.write("---")
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Raw Data Preview")
            st.dataframe(df_raw.head(10), use_container_width=True)
            
        with col_right:
            st.markdown("#### Preprocessed Data Preview")
            st.dataframe(df_clean.head(10), use_container_width=True)
            
        st.write("---")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### Summary Statistics")
            st.dataframe(df_clean.describe().T, use_container_width=True)
        with col_s2:
            st.markdown("#### Missing Values Report")
            missing_series = df_raw.isnull().sum()
            missing_df = pd.DataFrame({'Missing Count': missing_series, 'Percentage (%)': (missing_series / len(df_raw)) * 100})
            st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
            if missing_df['Missing Count'].sum() == 0:
                st.info("No missing values detected in the raw dataset.")
    else:
        st.info("Please load a dataset using the controls above to start.")

# ----------------- 3. EDA DASHBOARD -----------------
elif page == "📈 EDA Dashboard":
    st.markdown("<h1 class='main-title'>Exploratory Data Analysis (EDA)</h1>", unsafe_allow_html=True)
    
    df_clean = st.session_state['df_clean']
    is_ts = st.session_state['dataset_type'] == 'time_series'
    
    if df_clean is not None:
        eda_tab = st.selectbox("Select Visualizations Category:", [
            "Distribution (Univariate)",
            "Relationships (Bivariate)",
            "Trend & Seasonality (Time Series)"
        ])
        
        # Gather numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if eda_tab == "Distribution (Univariate)":
            st.markdown("### Univariate Analysis")
            col = st.selectbox("Select Numeric Feature to Analyze:", numeric_cols)
            
            # Sub-columns for side-by-side Plotly
            col1, col2 = st.columns(2)
            with col1:
                # Histogram with Density estimation (kde simulated via marginal)
                fig_hist = px.histogram(df_clean, x=col, marginal="box", 
                                        title=f"Histogram & Boxplot of {col}", 
                                        color_discrete_sequence=['#6A11CB'])
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with col2:
                # Density/Violin Plot
                fig_violin = px.violin(df_clean, y=col, box=True, points="all",
                                       title=f"Violin Plot with Individual Data Points: {col}",
                                       color_discrete_sequence=['#FF4B4B'])
                st.plotly_chart(fig_violin, use_container_width=True)
                
        elif eda_tab == "Relationships (Bivariate)":
            st.markdown("### Bivariate Analysis")
            
            col_x = st.selectbox("Select X-Axis Feature:", numeric_cols, index=0)
            col_y = st.selectbox("Select Y-Axis Feature:", numeric_cols, index=min(1, len(numeric_cols)-1))
            
            col_color = st.selectbox("Group by Categorical Variable (Optional):", ["None"] + categorical_cols)
            
            fig_scatter = px.scatter(
                df_clean, x=col_x, y=col_y, 
                color=None if col_color == "None" else col_color,
                trendline="ols" if col_color == "None" else None,
                title=f"Scatter Plot: {col_y} vs {col_x}",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.write("---")
            
            st.markdown("#### Correlation Heatmap")
            corr_matrix = df_clean[numeric_cols].corr()
            fig_heat = px.imshow(
                corr_matrix, text_auto=".2f", aspect="auto",
                title="Pearson Correlation Coefficient Matrix",
                color_continuous_scale="RdBu_r"
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            
        else: # Time-Series Specific
            if is_ts:
                st.markdown("### Time-Series Analysis")
                target = st.selectbox("Select Target Sequence:", numeric_cols)
                
                # Rolling statistics options
                roll_window = st.slider("Rolling Window Size (Days/Steps):", min_value=3, max_value=60, value=30)
                
                df_ts = df_clean.copy()
                df_ts['Rolling Mean'] = df_ts[target].rolling(window=roll_window).mean()
                df_ts['Rolling Std'] = df_ts[target].rolling(window=roll_window).std()
                
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=df_ts.index, y=df_ts[target], name='Actual Value', line=dict(color='#2575FC', width=1.5), alpha=0.5))
                fig_trend.add_trace(go.Scatter(x=df_ts.index, y=df_ts['Rolling Mean'], name=f'{roll_window}-Step Rolling Mean', line=dict(color='#FF4B4B', width=2.5)))
                fig_trend.add_trace(go.Scatter(x=df_ts.index, y=df_ts['Rolling Std'], name=f'{roll_window}-Step Rolling Std', line=dict(color='#6A11CB', width=1.5, dash='dash')))
                
                fig_trend.update_layout(
                    title=f"Trend & Rolling Statistics of {target}",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.5)')
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                
            else:
                st.warning("The current dataset is classified as Regression. Time-series analysis is only available for temporal datasets containing date columns.")
    else:
        st.info("Please load a dataset first.")

# ----------------- 4. MODEL TRAINING -----------------
elif page == "⚙️ Model Training":
    st.markdown("<h1 class='main-title'>Model Training & Comparison</h1>", unsafe_allow_html=True)
    
    df_clean = st.session_state['df_clean']
    is_ts = st.session_state['dataset_type'] == 'time_series'
    
    if df_clean is not None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        # 1. Target Column
        target_col = st.selectbox("Select Target Column to Predict:", numeric_cols, 
                                  index=numeric_cols.index('Close') if 'Close' in numeric_cols else 
                                        (numeric_cols.index('Sales') if 'Sales' in numeric_cols else len(numeric_cols)-1))
        
        st.session_state['target_col'] = target_col
        
        # 2. Excluded Features
        feature_candidates = [c for c in df_clean.columns if c != target_col]
        excluded_features = st.multiselect("Select features to exclude from model training (Optional):", feature_candidates)
        
        # Train configurations
        st.markdown("### Model Settings")
        
        tune_flag = st.checkbox("Perform Hyperparameter Tuning (using RandomizedSearchCV)", value=False,
                                help="Enabling hyperparameter tuning will search for optimal model arguments, but might take slightly longer to run.")
        
        # Check models to include
        st.markdown("#### Select Models to Train")
        model_selection = {}
        
        if is_ts:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Traditional Forecasters**")
                model_selection['SARIMA'] = st.checkbox("SARIMA", value=True)
                model_selection['Prophet'] = st.checkbox("Prophet (If installed)", value=True)
                
                st.write("**Deep Learning Models**")
                model_selection['LSTM'] = st.checkbox("LSTM (Tensorflow)", value=False)
                model_selection['GRU'] = st.checkbox("GRU (Tensorflow)", value=False)
            with col2:
                st.write("**Machine Learning Regressors**")
                model_selection['RandomForest'] = st.checkbox("Random Forest Regressor", value=True)
                model_selection['XGBoost'] = st.checkbox("XGBoost Regressor", value=True)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Linear Models**")
                model_selection['LinearRegression'] = st.checkbox("Linear Regression", value=True)
                model_selection['Ridge'] = st.checkbox("Ridge Regression", value=True)
                model_selection['Lasso'] = st.checkbox("Lasso Regression", value=True)
            with col2:
                st.write("**Tree-based ML Models**")
                model_selection['RandomForest'] = st.checkbox("Random Forest Regressor", value=True)
                model_selection['XGBoost'] = st.checkbox("XGBoost Regressor", value=True)
                
        # Train Button
        if st.button("Train Models"):
            # Exclude selected features
            df_train_set = df_clean.drop(columns=excluded_features)
            
            # Run Feature Engineering
            with st.spinner("Generating lag and rolling features..."):
                df_feat = engineer_features(df_train_set, target_col=target_col, is_time_series=is_ts)
                
            # Train Test split
            if is_ts:
                train_size = int(len(df_feat) * 0.8)
                df_train = df_feat.iloc[:train_size]
                df_test = df_feat.iloc[train_size:]
            else:
                df_train, df_test = train_test_split(df_feat, test_size=0.2, random_state=42)
                
            # Scale features
            X_train, y_train, encoders, scaler = preprocess_and_scale(
                df_train, target_col=target_col, scaling_method='standard'
            )
            X_test, y_test, _, _ = preprocess_and_scale(
                df_test, target_col=target_col, encoders=encoders, scaler=scaler
            )
            
            feature_cols = list(X_train.columns)
            
            results = {}
            trained_models = {}
            
            progress_bar = st.progress(0.0)
            total_selected_models = sum(model_selection.values())
            
            if total_selected_models == 0:
                st.warning("Please check at least one model to train.")
            else:
                step = 0
                
                # --- MODEL TRAINING LOOPS ---
                if is_ts:
                    # 1. SARIMA
                    if model_selection.get('SARIMA'):
                        with st.spinner("Training SARIMA..."):
                            try:
                                sarima_model = train_traditional_time_series(df_train[target_col], model_type='sarima')
                                trained_models['SARIMA'] = sarima_model
                                sarima_preds = sarima_model.forecast(steps=len(df_test))
                                results['SARIMA'] = evaluate_model(df_test[target_col], sarima_preds, is_time_series=True)
                            except Exception as e:
                                st.error(f"SARIMA failed: {e}")
                        step += 1
                        progress_bar.progress(step / total_selected_models)
                        
                    # 2. Prophet
                    if model_selection.get('Prophet'):
                        with st.spinner("Training Prophet..."):
                            try:
                                prophet_model = train_prophet_model(df_train, target_col)
                                if prophet_model is not None:
                                    trained_models['Prophet'] = prophet_model
                                    future_df = prophet_model.make_future_dataframe(periods=len(df_test), freq='D')
                                    forecast_df = prophet_model.predict(future_df)
                                    prophet_preds = forecast_df['yhat'].values[-len(df_test):]
                                    results['Prophet'] = evaluate_model(df_test[target_col].values, prophet_preds, is_time_series=True)
                                else:
                                    st.warning("Prophet not installed or not working, skipping.")
                            except Exception as e:
                                st.error(f"Prophet failed: {e}")
                        step += 1
                        progress_bar.progress(step / total_selected_models)
                        
                    # 3. Random Forest
                    if model_selection.get('RandomForest'):
                        with st.spinner("Training Random Forest..."):
                            try:
                                rf_model = train_regression_model(X_train, y_train, model_name='RandomForest', tune=tune_flag)
                                trained_models['RandomForest'] = rf_model
                                rf_preds = rf_model.predict(X_test)
                                results['RandomForest'] = evaluate_model(y_test, rf_preds, is_time_series=True)
                            except Exception as e:
                                st.error(f"Random Forest failed: {e}")
                        step += 1
                        progress_bar.progress(step / total_selected_models)
                        
                    # 4. XGBoost
                    if model_selection.get('XGBoost'):
                        with st.spinner("Training XGBoost..."):
                            try:
                                xgb_model = train_regression_model(X_train, y_train, model_name='XGBoost', tune=tune_flag)
                                trained_models['XGBoost'] = xgb_model
                                xgb_preds = xgb_model.predict(X_test)
                                results['XGBoost'] = evaluate_model(y_test, xgb_preds, is_time_series=True)
                            except Exception as e:
                                st.error(f"XGBoost failed: {e}")
                        step += 1
                        progress_bar.progress(step / total_selected_models)
                        
                    # 5. LSTM
                    if model_selection.get('LSTM'):
                        with st.spinner("Training LSTM..."):
                            try:
                                X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
                                lstm_data = train_lstm_gru(X_tr, y_tr, X_val, y_val, model_type='lstm')
                                if lstm_data is not None:
                                    trained_models['LSTM'] = lstm_data
                                    lstm_preds = predict_keras_model(lstm_data, X_test)
                                    results['LSTM'] = evaluate_model(y_test, lstm_preds, is_time_series=True)
                                else:
                                    st.warning("Tensorflow not installed, skipping LSTM.")
                            except Exception as e:
                                st.error(f"LSTM failed: {e}")
                        step += 1
                        progress_bar.progress(step / total_selected_models)
                        
                    # 6. GRU
                    if model_selection.get('GRU'):
                        with st.spinner("Training GRU..."):
                            try:
                                X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
                                gru_data = train_lstm_gru(X_tr, y_tr, X_val, y_val, model_type='gru')
                                if gru_data is not None:
                                    trained_models['GRU'] = gru_data
                                    gru_preds = predict_keras_model(gru_data, X_test)
                                    results['GRU'] = evaluate_model(y_test, gru_preds, is_time_series=True)
                                else:
                                    st.warning("Tensorflow not installed, skipping GRU.")
                            except Exception as e:
                                st.error(f"GRU failed: {e}")
                        step += 1
                        progress_bar.progress(step / total_selected_models)
                        
                else:
                    # Tabular Regression
                    reg_list = ['LinearRegression', 'Ridge', 'Lasso', 'RandomForest', 'XGBoost']
                    for r_model in reg_list:
                        if model_selection.get(r_model):
                            with st.spinner(f"Training {r_model}..."):
                                try:
                                    model = train_regression_model(X_train, y_train, model_name=r_model, tune=tune_flag)
                                    trained_models[r_model] = model
                                    preds = model.predict(X_test)
                                    results[r_model] = evaluate_model(y_test, preds, is_time_series=False)
                                except Exception as e:
                                    st.error(f"{r_model} failed: {e}")
                            step += 1
                            progress_bar.progress(step / total_selected_models)
                            
                # Compare
                if len(results) > 0:
                    df_compare = compile_comparison(results, is_time_series=is_ts)
                    st.session_state['df_compare'] = df_compare
                    
                    best_model_name = df_compare.index[0]
                    best_model = trained_models[best_model_name]
                    
                    pipeline_dict = {
                        'model_type': st.session_state['dataset_type'],
                        'model_name': best_model_name,
                        'model': best_model,
                        'scaler': scaler,
                        'encoders': encoders,
                        'target_col': target_col,
                        'datetime_col': st.session_state['datetime_col'],
                        'features': feature_cols,
                        'metrics': results[best_model_name]
                    }
                    st.session_state['pipeline'] = pipeline_dict
                    
                    # Save to model pkl
                    model_dir = "models"
                    os.makedirs(model_dir, exist_ok=True)
                    save_pipeline(pipeline_dict, os.path.join(model_dir, "saved_model.pkl"))
                    
                    st.success("All models trained successfully!")
                else:
                    st.error("No models succeeded training. Please verify your data and feature selection.")
                    
        # Display Comparisons
        df_compare = st.session_state['df_compare']
        pipeline_dict = st.session_state['pipeline']
        
        if df_compare is not None and pipeline_dict is not None:
            st.markdown("<h3 class='section-header'>Model Comparison Table</h3>", unsafe_allow_html=True)
            st.dataframe(df_compare.style.highlight_min(axis=0, subset=['RMSE', 'MAE'] if is_ts else ['RMSE', 'MAE', 'MSE'])
                                          .highlight_max(axis=0, subset=['R2'] if not is_ts else []), 
                         use_container_width=True)
            
            # Best model banner
            st.markdown(f"""
            <div class='card' style='background: linear-gradient(135deg, rgba(106, 17, 203, 0.1), rgba(37, 117, 252, 0.1)); border-color: #6A11CB;'>
                <h3>🏆 Best Model: <span style='color: #6A11CB;'>{pipeline_dict['model_name']}</span></h3>
                <p>Selected based on minimizing test set error metric (RMSE).</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature Importance
            best_model_name = pipeline_dict['model_name']
            best_model_obj = pipeline_dict['model']
            
            # Check if model has feature importance
            has_fi = False
            feat_imp = None
            
            if best_model_name in ['RandomForest', 'XGBoost']:
                feat_imp = best_model_obj.feature_importances_
                has_fi = True
                
            if has_fi and feat_imp is not None:
                st.markdown("<h3 class='section-header'>Feature Importance</h3>", unsafe_allow_html=True)
                features_list = pipeline_dict['features']
                
                df_fi = pd.DataFrame({'Feature': features_list, 'Importance': feat_imp})
                df_fi = df_fi.sort_values(by='Importance', ascending=True)
                
                fig_fi = px.bar(df_fi, x='Importance', y='Feature', orientation='h',
                               title=f"Relative Feature Importances ({best_model_name})",
                               color='Importance', color_continuous_scale='plasma')
                fig_fi.update_layout(height=max(400, len(features_list)*20))
                st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Please load a dataset in Dataset Overview to begin training.")

# ----------------- 5. FORECASTING -----------------
elif page == "🔮 Forecasting":
    st.markdown("<h1 class='main-title'>Future Trend Forecasting</h1>", unsafe_allow_html=True)
    
    pipeline = st.session_state['pipeline']
    df_clean = st.session_state['df_clean']
    is_ts = st.session_state['dataset_type'] == 'time_series'
    
    if pipeline is not None and df_clean is not None:
        target_col = pipeline['target_col']
        model_name = pipeline['model_name']
        
        if is_ts:
            st.markdown("### Time-Series Future Forecasting Settings")
            forecast_period = st.selectbox("Select Forecast Horizon:", [7, 30, 90], index=1)
            
            if st.button("Generate Future Forecast") or st.session_state['forecast_df'] is not None:
                # Recalculate if needed
                if st.session_state['forecast_df'] is None or len(st.session_state['forecast_df']) != forecast_period:
                    with st.spinner("Generating multi-step forecast..."):
                        forecast_df = forecast_time_series(pipeline, df_clean, periods=forecast_period, target_col=target_col)
                        st.session_state['forecast_df'] = forecast_df
                else:
                    forecast_df = st.session_state['forecast_df']
                
                # Plotly Chart
                fig_fc = go.Figure()
                
                # Plot last 120 days of historical data for visual context
                history_subset = df_clean.tail(120)
                fig_fc.add_trace(go.Scatter(x=history_subset.index, y=history_subset[target_col], 
                                            name="Historical Actual", line=dict(color='#2575FC', width=2)))
                
                # Plot forecasted values
                fig_fc.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Predicted_Value'], 
                                            name="Forecasted Trend", line=dict(color='#FF4B4B', width=2.5)))
                
                # Shaded confidence intervals
                fig_fc.add_trace(go.Scatter(
                    x=list(forecast_df.index) + list(forecast_df.index)[::-1],
                    y=list(forecast_df['Upper_CI']) + list(forecast_df['Lower_CI'])[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 75, 75, 0.15)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=True,
                    name="95% Confidence Interval"
                ))
                
                fig_fc.update_layout(
                    title=f"Future {forecast_period}-Day Trend Forecast using {model_name}",
                    xaxis_title="Date",
                    yaxis_title=f"{target_col} Value",
                    hovermode="x unified",
                    height=500
                )
                
                st.plotly_chart(fig_fc, use_container_width=True)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("#### Forecast Predictions Table")
                    st.dataframe(forecast_df[['Predicted_Value', 'Lower_CI', 'Upper_CI']], use_container_width=True)
                with col2:
                    st.markdown("#### Export Forecast Predictions")
                    csv_data = forecast_df.to_csv(index=True)
                    st.download_button(
                        label="📥 Download Forecast CSV",
                        data=csv_data,
                        file_name=f"{target_col}_forecast_{forecast_period}days_{model_name}.csv",
                        mime="text/csv"
                    )
        else:
            # Tabular Regression Predictions (Actual vs Predicted plot on full dataset)
            st.markdown("### Tabular Regression Predictions")
            
            with st.spinner("Generating predictions..."):
                X_eval, y_eval, _, _ = preprocess_and_scale(
                    df_clean, target_col=target_col, 
                    encoders=pipeline['encoders'], scaler=pipeline['scaler']
                )
                
                features_list = pipeline['features']
                preds = predict_regression(pipeline['model'], X_eval[features_list])
                
                df_res = df_clean.copy()
                df_res['Predicted_Value'] = preds
                
                # Scatter Actual vs Predicted
                fig_reg = px.scatter(df_res, x=target_col, y='Predicted_Value', 
                                     labels={'Close': 'Actual', 'Predicted_Value': 'Predicted'},
                                     title=f"Actual vs Predicted {target_col} ({model_name})",
                                     color_discrete_sequence=['#6A11CB'])
                # Diagonal perfect-fit line
                min_val = min(df_res[target_col].min(), preds.min())
                max_val = max(df_res[target_col].max(), preds.max())
                fig_reg.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                                  line=dict(color="#FF4B4B", width=2, dash="dash"))
                
                st.plotly_chart(fig_reg, use_container_width=True)
                
                st.dataframe(df_res[[target_col, 'Predicted_Value']].head(20), use_container_width=True)
                
                csv_data = df_res[[target_col, 'Predicted_Value']].to_csv(index=True)
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv_data,
                    file_name=f"{target_col}_predictions_{model_name}.csv",
                    mime="text/csv"
                )
    else:
        st.info("You must train a model on the Model Training page before forecasting future values.")

# ----------------- 6. PREDICTION RESULTS -----------------
elif page == "✅ Prediction Results":
    st.markdown("<h1 class='main-title'>Prediction Results & Residual Analysis</h1>", unsafe_allow_html=True)
    
    pipeline = st.session_state['pipeline']
    df_clean = st.session_state['df_clean']
    is_ts = st.session_state['dataset_type'] == 'time_series'
    
    if pipeline is not None and df_clean is not None:
        target_col = pipeline['target_col']
        model_name = pipeline['model_name']
        features_list = pipeline['features']
        
        # We perform predictions on full preprocessed data for residuals analysis
        with st.spinner("Analyzing errors..."):
            # Feature engineering
            df_feat = engineer_features(df_clean, target_col=target_col, is_time_series=is_ts)
            
            # Scale
            X_eval, y_eval, _, _ = preprocess_and_scale(
                df_feat, target_col=target_col, 
                encoders=pipeline['encoders'], scaler=pipeline['scaler']
            )
            
            # Predict
            if model_name in ['LSTM', 'GRU']:
                preds = predict_keras_model(pipeline['model'], X_eval[features_list])
            elif model_name in ['ARIMA', 'SARIMA']:
                # Statsmodels prediction
                # Generate predictions corresponding to the index
                preds = pipeline['model'].predict(start=df_feat.index[0], end=df_feat.index[-1]).values
            elif model_name == 'Prophet':
                # Prophet prediction
                prophet_df = pd.DataFrame({
                    'ds': df_feat.index,
                    'y': df_feat[target_col]
                }).reset_index(drop=True)
                forecast_df = pipeline['model'].predict(prophet_df)
                preds = forecast_df['yhat'].values
            else:
                preds = pipeline['model'].predict(X_eval[features_list])
                
            residuals, stats = analyze_residuals(y_eval, preds)
            
        st.markdown("### Residual Error Diagnostic Report")
        
        # Display residual stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Error", f"{stats['mean']:.4f}")
        with col2:
            st.metric("Error Std Dev", f"{stats['std']:.4f}")
        with col3:
            st.metric("Skewness", f"{stats['skew']:.4f}")
        with col4:
            st.metric("Kurtosis", f"{stats['kurtosis']:.4f}")
            
        st.write("---")
        
        col_l, col_r = st.columns(2)
        with col_l:
            # Residual histogram
            fig_hist = px.histogram(pd.DataFrame({'Residuals': residuals}), x='Residuals', 
                                    marginal="rug", title="Distribution of Residuals (Errors)",
                                    color_discrete_sequence=['#FF4B4B'])
            fig_hist.add_vline(x=0, line_dash="dash", line_color="#6A11CB", line_width=2)
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_r:
            # Residuals vs Fitted values
            fig_scat = px.scatter(x=preds, y=residuals, 
                                  labels={'x': 'Fitted (Predicted) Values', 'y': 'Residuals (Errors)'},
                                  title="Residuals vs Fitted Values",
                                  color_discrete_sequence=['#2575FC'])
            fig_scat.add_hline(y=0, line_dash="dash", line_color="#FF4B4B", line_width=2)
            st.plotly_chart(fig_scat, use_container_width=True)
            
        st.write("---")
        
        # Cumulative error tracking
        st.markdown("#### Cumulative Residual Error Time Series")
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=df_feat.index, y=np.cumsum(residuals), 
                                     name="Cumulative Residual", line=dict(color='#6A11CB', width=2)))
        fig_cum.update_layout(
            title="Cumulative Prediction Error over Time",
            xaxis_title="Index/Date",
            yaxis_title="Cumulative Error"
        )
        st.plotly_chart(fig_cum, use_container_width=True)
    else:
        st.info("You must train a model on the Model Training page before analyzing prediction results.")
