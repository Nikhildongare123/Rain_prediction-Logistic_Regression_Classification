import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Rainfall Predictor Pro",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    .rain-box {
        background-color: #ff6b6b;
        color: white;
        border: 2px solid #ff0000;
    }
    .no-rain-box {
        background-color: #4ecdc4;
        color: white;
        border: 2px solid #00ff00;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .info-text {
        color: #666;
        font-size: 14px;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    try:
        with open('rain_prediction_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load data with caching
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data (1).csv')
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

# Feature names
FEATURES = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']
FEATURE_NAMES_DISPLAY = {
    'Temperature': '🌡️ Temperature (°C)',
    'Humidity': '💧 Humidity (%)',
    'Wind_Speed': '💨 Wind Speed (m/s)',
    'Cloud_Cover': '☁️ Cloud Cover (%)',
    'Pressure': '📊 Pressure (hPa)'
}

# Default values for different weather scenarios
DEFAULT_SCENARIOS = {
    "🌞 Clear Sky": {
        'Temperature': 28.0,
        'Humidity': 40.0,
        'Wind_Speed': 5.0,
        'Cloud_Cover': 10.0,
        'Pressure': 1015.0
    },
    "☁️ Cloudy": {
        'Temperature': 22.0,
        'Humidity': 65.0,
        'Wind_Speed': 8.0,
        'Cloud_Cover': 75.0,
        'Pressure': 1010.0
    },
    "🌧️ Rainy": {
        'Temperature': 18.0,
        'Humidity': 85.0,
        'Wind_Speed': 15.0,
        'Cloud_Cover': 90.0,
        'Pressure': 1005.0
    },
    "💨 Stormy": {
        'Temperature': 15.0,
        'Humidity': 90.0,
        'Wind_Speed': 25.0,
        'Cloud_Cover': 95.0,
        'Pressure': 995.0
    }
}

def make_prediction(model, features):
    """Make prediction with probability"""
    input_array = np.array([features]).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(input_array)[0]
        rain_idx = list(model.classes_).index('rain')
        rain_probability = probabilities[rain_idx]
        return prediction, rain_probability
    return prediction, None

def create_radar_chart(values):
    """Create radar chart for feature visualization"""
    categories = list(FEATURE_NAMES_DISPLAY.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current Conditions',
        line_color='#4CAF50',
        fillcolor='rgba(76, 175, 80, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.2]
            )),
        showlegend=True,
        height=400,
        title="Weather Feature Radar Chart"
    )
    return fig

def plot_feature_distribution(df, feature):
    """Create distribution plot for a feature"""
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('Distribution', 'Box Plot'),
                        specs=[[{'type': 'histogram'}, {'type': 'box'}]])
    
    # Histogram
    fig.add_trace(go.Histogram(x=df[feature], name='Distribution', 
                               marker_color='#4CAF50', opacity=0.7), row=1, col=1)
    
    # Box plot
    fig.add_trace(go.Box(y=df[feature], name='Box Plot', 
                         marker_color='#FF6B6B'), row=1, col=2)
    
    fig.update_layout(height=400, title_text=f"{FEATURE_NAMES_DISPLAY[feature]} Analysis")
    return fig

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1163/1163659.png", width=100)
        st.title("🌧️ Rainfall Predictor Pro")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigate",
            ["🎯 Single Prediction", "📊 Batch Prediction", "📈 Data Explorer", "📚 Model Insights"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This app uses **Logistic Regression** to predict rainfall based on:
        - Temperature
        - Humidity  
        - Wind Speed
        - Cloud Cover
        - Pressure
        
        **Accuracy:** Based on your training data
        """)
        
        st.markdown("---")
        st.markdown("### 📞 Support")
        st.markdown("For issues, contact: support@weatherpredictor.com")
    
    # Load model and data
    model = load_model()
    df = load_data()
    
    if model is None or df is None:
        st.error("Failed to load model or data. Please check your files.")
        return
    
    # Main content
    if page == "🎯 Single Prediction":
        st.title("🎯 Weather Prediction Engine")
        st.markdown("### Enter weather parameters to predict rainfall")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Quick scenario selector
            st.markdown("#### Quick Scenario Selector")
            selected_scenario = st.selectbox("Load preset weather scenario:", 
                                             list(DEFAULT_SCENARIOS.keys()))
            
            if selected_scenario:
                scenario = DEFAULT_SCENARIOS[selected_scenario]
            else:
                scenario = DEFAULT_SCENARIOS["🌞 Clear Sky"]
            
            # Input fields
            col_temp, col_hum = st.columns(2)
            with col_temp:
                temperature = st.slider(
                    FEATURE_NAMES_DISPLAY['Temperature'],
                    min_value=-10.0, max_value=50.0,
                    value=scenario['Temperature'], step=0.5,
                    help="Temperature in degrees Celsius"
                )
            with col_hum:
                humidity = st.slider(
                    FEATURE_NAMES_DISPLAY['Humidity'],
                    min_value=0.0, max_value=100.0,
                    value=scenario['Humidity'], step=1.0,
                    help="Relative humidity percentage"
                )
            
            col_wind, col_cloud = st.columns(2)
            with col_wind:
                wind_speed = st.slider(
                    FEATURE_NAMES_DISPLAY['Wind_Speed'],
                    min_value=0.0, max_value=30.0,
                    value=scenario['Wind_Speed'], step=0.5,
                    help="Wind speed in meters per second"
                )
            with col_cloud:
                cloud_cover = st.slider(
                    FEATURE_NAMES_DISPLAY['Cloud_Cover'],
                    min_value=0.0, max_value=100.0,
                    value=scenario['Cloud_Cover'], step=1.0,
                    help="Cloud cover percentage"
                )
            
            pressure = st.slider(
                FEATURE_NAMES_DISPLAY['Pressure'],
                min_value=980.0, max_value=1050.0,
                value=scenario['Pressure'], step=1.0,
                help="Atmospheric pressure in hPa"
            )
        
        with col2:
            st.markdown("#### 📊 Feature Radar")
            feature_values = [temperature, humidity, wind_speed, cloud_cover, pressure]
            radar_chart = create_radar_chart(feature_values)
            st.plotly_chart(radar_chart, use_container_width=True)
        
        # Make prediction
        if st.button("🔮 Predict Rainfall", use_container_width=True):
            features = [temperature, humidity, wind_speed, cloud_cover, pressure]
            prediction, probability = make_prediction(model, features)
            
            # Display prediction
            if prediction == 'rain':
                st.markdown(f"""
                <div class="prediction-box rain-box">
                    🌧️ RAIN PREDICTED 🌧️<br>
                    <span style="font-size: 18px;">Probability: {probability:.1%}</span>
                </div>
                """, unsafe_allow_html=True)
                st.warning("⚠️ Carry an umbrella! Rain is likely.")
            else:
                st.markdown(f"""
                <div class="prediction-box no-rain-box">
                    ☀️ NO RAIN PREDICTED ☀️<br>
                    <span style="font-size: 18px;">Probability: {1-probability:.1%}</span>
                </div>
                """, unsafe_allow_html=True)
                st.success("✅ Good weather! No rain expected.")
            
            # Risk assessment
            if probability:
                st.markdown("#### 📈 Risk Assessment")
                risk_level = "Low" if probability < 0.3 else "Medium" if probability < 0.6 else "High"
                risk_color = "green" if probability < 0.3 else "orange" if probability < 0.6 else "red"
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px;">
                    <b>Rain Probability:</b> {probability:.1%}<br>
                    <b>Risk Level:</b> <span style="color: {risk_color}; font-weight: bold;">{risk_level}</span><br>
                    <b>Recommendation:</b> {'Bring umbrella' if probability > 0.4 else 'No umbrella needed'}
                </div>
                """, unsafe_allow_html=True)
    
    elif page == "📊 Batch Prediction":
        st.title("📊 Batch Rainfall Prediction")
        st.markdown("### Upload CSV file for bulk predictions")
        
        st.info("""
        **CSV Format Requirements:**
        Your CSV file must contain these exact column names:
        - Temperature
        - Humidity
        - Wind_Speed
        - Cloud_Cover
        - Pressure
        
        **Example:**
        Temperature,Humidity,Wind_Speed,Cloud_Cover,Pressure
        25.5,65.0,10.2,50.0,1013.5
        18.0,85.0,15.5,90.0,1005.0
        """)
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                # Load and validate
                batch_df = pd.read_csv(uploaded_file)
                missing_cols = [col for col in FEATURES if col not in batch_df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing columns: {missing_cols}")
                else:
                    # Make predictions
                    X_batch = batch_df[FEATURES].values
                    predictions = model.predict(X_batch)
                    
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(X_batch)
                        rain_idx = list(model.classes_).index('rain')
                        batch_df['Rain_Probability'] = probabilities[:, rain_idx]
                    
                    batch_df['Prediction'] = predictions
                    
                    # Display summary
                    st.success(f"✅ Successfully processed {len(batch_df)} records!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", len(batch_df))
                    with col2:
                        rain_count = sum(batch_df['Prediction'] == 'rain')
                        st.metric("🌧️ Rain Predictions", rain_count)
                    with col3:
                        no_rain_count = len(batch_df) - rain_count
                        st.metric("☀️ No Rain Predictions", no_rain_count)
                    
                    # Show results
                    st.subheader("📋 Prediction Results")
                    st.dataframe(batch_df, use_container_width=True)
                    
                    # Download button
                    csv = batch_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name=f"rain_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    elif page == "📈 Data Explorer":
        st.title("📈 Exploratory Data Analysis")
        
        if df is not None:
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Features", len(FEATURES))
            with col3:
                rain_count = len(df[df['Rain'] == 'rain'])
                st.metric("🌧️ Rain Days", rain_count, delta=f"{rain_count/len(df)*100:.1f}%")
            with col4:
                no_rain_count = len(df[df['Rain'] == 'no rain'])
                st.metric("☀️ No Rain Days", no_rain_count)
            
            # Correlation heatmap
            st.subheader("🔥 Feature Correlation Heatmap")
            corr_matrix = df[FEATURES].corr()
            fig = px.imshow(corr_matrix, 
                           text_auto=True, 
                           aspect="auto",
                           color_continuous_scale='RdBu',
                           title="Correlation Between Features")
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature distributions
            st.subheader("📊 Feature Distributions by Weather Condition")
            selected_feature = st.selectbox("Select feature to analyze:", FEATURES)
            
            fig = make_subplots(rows=1, cols=2, 
                               subplot_titles=('Distribution - Rain', 'Distribution - No Rain'))
            
            rain_data = df[df['Rain'] == 'rain'][selected_feature]
            no_rain_data = df[df['Rain'] == 'no rain'][selected_feature]
            
            fig.add_trace(go.Histogram(x=rain_data, name='Rain', 
                                      marker_color='#FF6B6B', opacity=0.7), row=1, col=1)
            fig.add_trace(go.Histogram(x=no_rain_data, name='No Rain', 
                                      marker_color='#4ECDC4', opacity=0.7), row=1, col=2)
            
            fig.update_layout(height=400, title_text=f"{FEATURE_NAMES_DISPLAY[selected_feature]} Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical summary
            st.subheader("📊 Statistical Summary")
            st.dataframe(df[FEATURES].describe(), use_container_width=True)
            
            # Box plots
            st.subheader("📦 Box Plots by Condition")
            fig = make_subplots(rows=2, cols=3, subplot_titles=list(FEATURE_NAMES_DISPLAY.values()))
            
            for idx, feature in enumerate(FEATURES):
                row = idx // 3 + 1
                col = idx % 3 + 1
                
                rain_vals = df[df['Rain'] == 'rain'][feature]
                no_rain_vals = df[df['Rain'] == 'no rain'][feature]
                
                fig.add_trace(go.Box(y=rain_vals, name='Rain', marker_color='#FF6B6B'), row=row, col=col)
                fig.add_trace(go.Box(y=no_rain_vals, name='No Rain', marker_color='#4ECDC4'), row=row, col=col)
            
            fig.update_layout(height=800, showlegend=True, title_text="Feature Comparison: Rain vs No Rain")
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "📚 Model Insights":
        st.title("📚 Model Performance & Insights")
        
        if df is not None:
            # Model information
            st.subheader("🤖 Model Architecture")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"""
                **Model Details:**
                - **Type:** Logistic Regression
                - **Algorithm:** Linear Classifier
                - **Features:** {len(FEATURES)}
                - **Classes:** {', '.join(model.classes_)}
                """)
            
            with col2:
                if hasattr(model, 'coef_'):
                    st.info(f"""
                    **Coefficients:**
                    {', '.join([f'{FEATURES[i]}: {model.coef_[0][i]:.3f}' for i in range(len(FEATURES))])}
                    """)
            
            # Feature importance
            st.subheader("📊 Feature Importance")
            if hasattr(model, 'coef_'):
                importance_df = pd.DataFrame({
                    'Feature': FEATURES,
                    'Coefficient': np.abs(model.coef_[0]),
                    'Direction': ['Positive' if c > 0 else 'Negative' for c in model.coef_[0]]
                }).sort_values('Coefficient', ascending=True)
                
                fig = px.bar(importance_df, x='Coefficient', y='Feature', 
                            orientation='h', color='Direction',
                            color_discrete_map={'Positive': '#4CAF50', 'Negative': '#FF6B6B'},
                            title="Feature Impact on Prediction")
                st.plotly_chart(fig, use_container_width=True)
            
            # Dataset insights
            st.subheader("📊 Dataset Insights")
            
            # Class balance
            class_balance = df['Rain'].value_counts()
            fig = px.pie(values=class_balance.values, names=class_balance.index,
                        title="Class Distribution in Training Data",
                        color_discrete_sequence=['#4ECDC4', '#FF6B6B'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature averages comparison
            st.subheader("📈 Feature Averages: Rain vs No Rain")
            comparison_data = []
            for feature in FEATURES:
                comparison_data.append({
                    'Feature': FEATURE_NAMES_DISPLAY[feature],
                    'Average (Rain)': df[df['Rain'] == 'rain'][feature].mean(),
                    'Average (No Rain)': df[df['Rain'] == 'no rain'][feature].mean(),
                    'Difference': df[df['Rain'] == 'rain'][feature].mean() - df[df['Rain'] == 'no rain'][feature].mean()
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df.style.format({
                'Average (Rain)': '{:.2f}',
                'Average (No Rain)': '{:.2f}',
                'Difference': '{:.2f}'
            }), use_container_width=True)
            
            # Tips
            st.subheader("💡 Prediction Tips")
            st.markdown("""
            Based on the model analysis:
            
            1. **High Humidity** (>70%) significantly increases rain probability
            2. **High Cloud Cover** (>60%) is strongly correlated with rainfall
            3. **Low Pressure** (<1000 hPa) often indicates incoming rain
            4. **Wind Speed** above 10 m/s combined with high humidity suggests stormy conditions
            5. **Temperature** alone is not a strong predictor - consider all factors together
            
            🎯 **Best practice:** Use the radar chart to see how your current conditions compare to typical rain scenarios.
            """)

if __name__ == "__main__":
    main()
