"""
Streamlit app for Electricity Demand Forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import config
import feature_engineering
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Electricity Demand Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained XGBoost model."""
    try:
        model = joblib.load(config.MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Model not found at {config.MODEL_PATH}. Please train the model first.")
        return None

@st.cache_data
def load_all_data():
    """Load and process all available data."""
    try:
        data = feature_engineering.get_processed_data(config.DATA_PATH)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_prediction(model, data):
    """Make predictions on data."""
    X = data.drop(config.TARGET_COL, axis=1) if config.TARGET_COL in data.columns else data
    predictions = model.predict(X)
    return predictions

def create_prediction_chart(results, title="Electricity Demand Predictions"):
    """Create an interactive prediction chart."""
    fig = go.Figure()
    
    if 'Actual_Demand' in results.columns:
        fig.add_trace(go.Scatter(
            x=results.index,
            y=results['Actual_Demand'],
            mode='lines',
            name='Actual Demand',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>Actual</b><br>Time: %{x}<br>Demand: %{y:.2f} MW<extra></extra>'
        ))
    
    fig.add_trace(go.Scatter(
        x=results.index,
        y=results['Predicted_Demand'],
        mode='lines',
        name='Predicted Demand',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        hovertemplate='<b>Predicted</b><br>Time: %{x}<br>Demand: %{y:.2f} MW<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date Time',
        yaxis_title='Demand (MW)',
        height=500,
        hovermode='x unified',
        template='plotly_white',
        font=dict(size=12)
    )
    
    return fig

def calculate_metrics(results):
    """Calculate prediction metrics."""
    if 'Actual_Demand' not in results.columns:
        return None
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    actual = results['Actual_Demand']
    predicted = results['Predicted_Demand']
    
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R² Score': r2,
        'MAPE': mape
    }

# ============================================================================
# MAIN APP
# ============================================================================

st.title("⚡ Electricity Demand Forecasting System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Configuration")
    
    # Load data and model
    data = load_all_data()
    model = load_model()
    
    if data is None or model is None:
        st.error("Error loading required files. Please ensure the model and data are available.")
        st.stop()
    
    st.success("✅ Model and Data Loaded Successfully")
    
    # Navigation
    page = st.radio(
        "Select Page:",
        ["📈 Dashboard", "🔮 Make Predictions", "📉 Analysis", "ℹ️ About"]
    )

# ============================================================================
# PAGE 1: DASHBOARD
# ============================================================================

if page == "📈 Dashboard":
    st.header("Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get test data predictions
    test_data = data.loc[config.TEST_START_DATE:]
    if not test_data.empty:
        predictions = get_prediction(model, test_data)
        test_data_copy = test_data.copy()
        test_data_copy['Predicted_Demand'] = predictions
        
        metrics = calculate_metrics(test_data_copy)
        
        if metrics:
            with col1:
                st.metric("MAE", f"{metrics['MAE']:.2f} MW")
            with col2:
                st.metric("RMSE", f"{metrics['RMSE']:.2f} MW")
            with col3:
                st.metric("R² Score", f"{metrics['R² Score']:.4f}")
            with col4:
                st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
    
    st.markdown("---")
    
    # Full prediction chart
    st.subheader("Full Historical Predictions vs Actual")
    
    predictions_all = get_prediction(model, data)
    results_all = pd.DataFrame(index=data.index)
    if config.TARGET_COL in data.columns:
        results_all['Actual_Demand'] = data[config.TARGET_COL]
    results_all['Predicted_Demand'] = predictions_all
    
    fig = create_prediction_chart(results_all)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent data
    st.subheader("Recent Predictions (Last 100 Hours)")
    recent_data = results_all.tail(100)
    st.dataframe(recent_data, use_container_width=True)

# ============================================================================
# PAGE 2: MAKE PREDICTIONS
# ============================================================================

elif page == "🔮 Make Predictions":
    st.header("Make New Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Select Prediction Range")
        
        # Date range selector
        available_dates = data.index
        min_date = min(available_dates).date()
        max_date = max(available_dates).date()
        
        start_date = st.date_input(
            "Start Date",
            value=max_date - timedelta(days=30),
            min_value=min_date,
            max_value=max_date
        )
        
        end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
        
        if start_date > end_date:
            st.error("Start date must be before end date")
        else:
            # Filter data for selected range
            mask = (data.index.date >= start_date) & (data.index.date <= end_date)
            selected_data = data.loc[mask]
            
            if selected_data.empty:
                st.warning("No data available for the selected date range")
            else:
                # Make predictions
                predictions = get_prediction(model, selected_data)
                results = pd.DataFrame(index=selected_data.index)
                if config.TARGET_COL in selected_data.columns:
                    results['Actual_Demand'] = selected_data[config.TARGET_COL]
                results['Predicted_Demand'] = predictions
                
                with col2:
                    st.subheader("Prediction Summary")
                    
                    st.write(f"**Data Points:** {len(results)}")
                    st.write(f"**Average Predicted Demand:** {results['Predicted_Demand'].mean():.2f} MW")
                    st.write(f"**Max Predicted Demand:** {results['Predicted_Demand'].max():.2f} MW")
                    st.write(f"**Min Predicted Demand:** {results['Predicted_Demand'].min():.2f} MW")
                    st.write(f"**Std Dev Predicted:** {results['Predicted_Demand'].std():.2f} MW")
                
                st.markdown("---")
                
                # Prediction chart
                fig = create_prediction_chart(results, f"Predictions: {start_date} to {end_date}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed data table
                st.subheader("Detailed Predictions")
                
                display_data = results.copy()
                if 'Actual_Demand' in display_data.columns:
                    display_data['Error'] = display_data['Actual_Demand'] - display_data['Predicted_Demand']
                    display_data['Error %'] = (display_data['Error'] / display_data['Actual_Demand'] * 100).round(2)
                
                st.dataframe(display_data, use_container_width=True)
                
                # Download predictions
                csv = display_data.to_csv()
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name=f"predictions_{start_date}_to_{end_date}.csv",
                    mime="text/csv"
                )

# ============================================================================
# PAGE 3: ANALYSIS
# ============================================================================

elif page == "📉 Analysis":
    st.header("Detailed Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Model Performance", "Feature Analysis", "Data Distribution"])
    
    with tab1:
        st.subheader("Model Performance Metrics")
        
        # Get test predictions
        test_data = data.loc[config.TEST_START_DATE:]
        predictions = get_prediction(model, test_data)
        test_data_copy = test_data.copy()
        test_data_copy['Predicted_Demand'] = predictions
        
        metrics = calculate_metrics(test_data_copy)
        
        if metrics:
            metric_df = pd.DataFrame(
                list(metrics.items()),
                columns=['Metric', 'Value']
            )
            st.dataframe(metric_df, use_container_width=True)
        
        # Residual analysis
        st.subheader("Residual Analysis")
        residuals = test_data[config.TARGET_COL] - predictions
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_residuals = px.histogram(
                residuals,
                nbins=50,
                title="Distribution of Residuals",
                labels={'value': 'Residual (MW)', 'count': 'Frequency'},
                template='plotly_white'
            )
            st.plotly_chart(fig_residuals, use_container_width=True)
        
        with col2:
            fig_scatter = px.scatter(
                x=test_data[config.TARGET_COL],
                y=residuals,
                title="Residuals vs Actual Demand",
                labels={'x': 'Actual Demand (MW)', 'y': 'Residual (MW)'},
                template='plotly_white',
                opacity=0.6
            )
            fig_scatter.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab2:
        st.subheader("Feature Importance")
        
        try:
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame(
                    {
                        'feature': [col for col in data.columns if col != config.TARGET_COL],
                        'importance': model.feature_importances_
                    }
                ).sort_values('importance', ascending=False).head(15)
                
                fig = px.bar(
                    feature_importance,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 15 Most Important Features',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Feature importance not available for this model type")
    
    with tab3:
        st.subheader("Data Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_demand = px.histogram(
                data[config.TARGET_COL],
                nbins=50,
                title="Distribution of Actual Demand",
                labels={'count': 'Frequency', 'value': 'Demand (MW)'},
                template='plotly_white'
            )
            st.plotly_chart(fig_demand, use_container_width=True)
        
        with col2:
            fig_temp = px.histogram(
                data['Temperature'],
                nbins=50,
                title="Distribution of Temperature",
                labels={'count': 'Frequency', 'value': 'Temperature (°C)'},
                template='plotly_white'
            )
            st.plotly_chart(fig_temp, use_container_width=True)

# ============================================================================
# PAGE 4: ABOUT
# ============================================================================

elif page == "ℹ️ About":
    st.header("About This Application")
    
    st.markdown("""
    ### 📊 Electricity Demand Forecasting System
    
    This application uses a machine learning model to predict electricity demand based on historical data
    and various environmental factors.
    
    #### 🤖 Model Details
    - **Model Type:** XGBoost Regressor
    - **Training Data:** Historical electricity demand and weather data
    - **Features:** Time-based, lag, and rolling window features
    - **Prediction Target:** Electricity Demand (MW)
    
    #### 📈 Features Used
    - **Time Features:** Hour, Day of Week, Month, Year, Day of Year, Quarter, Week of Year
    - **Lag Features:** 
        - Demand lag at 24 hours
        - Demand lag at 168 hours (1 week)
    - **Rolling Features:**
        - 24-hour rolling mean of demand
        - 24-hour rolling standard deviation of demand
    - **Weather Features:** Temperature, Humidity
    
    #### 🎯 Key Metrics
    - **MAE (Mean Absolute Error):** Average absolute prediction error
    - **RMSE (Root Mean Squared Error):** Penalty for larger errors
    - **R² Score:** Proportion of variance explained by the model
    - **MAPE (Mean Absolute Percentage Error):** Percentage-based error metric
    
    #### 📂 Project Structure
    """)
    
    st.code("""
    Electricity Demand Forecasting/
    ├── config.py              # Configuration and parameters
    ├── feature_engineering.py # Data preprocessing and feature creation
    ├── eda.py                 # Exploratory data analysis
    ├── train.py               # Model training script
    ├── predict.py             # Prediction script
    ├── app.py                 # Streamlit web application
    ├── requirements.txt       # Python dependencies
    ├── data/
    │   └── Electricity+Demand+Dataset.csv
    ├── models/
    │   └── electricity_xgb_prediction_model.pkl
    ├── outputs/
    │   ├── processed_electricity_demand.xlsx
    │   └── predictions.csv
    └── README.md
    """)
    
    st.markdown("""
    #### 🚀 Usage
    1. Navigate to the **Dashboard** to see overall model performance
    2. Use **Make Predictions** to select a date range and get forecasts
    3. Explore the **Analysis** tab for detailed metrics and visualizations
    4. Download predictions as CSV for further analysis
    
    #### 📝 Notes
    - All predictions are based on historical patterns and trained model
    - Actual demand may vary due to external factors not captured in the data
    - Model performance may decrease for dates far outside the training range
    
    #### 👨‍💻 Technologies Used
    - **Streamlit:** Interactive web application framework
    - **Pandas & NumPy:** Data manipulation and analysis
    - **XGBoost:** Gradient boosting machine learning model
    - **Plotly:** Interactive visualizations
    - **Scikit-learn:** Machine learning metrics and utilities
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
    <small>⚡ Electricity Demand Forecasting System | Built with Streamlit</small>
    </div>
    """,
    unsafe_allow_html=True
)
