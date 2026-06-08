import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Rain Prediction App",
    page_icon="🌧️",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    with open('rain_prediction_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load and prepare the data
@st.cache_data
def load_data():
    df = pd.read_csv('data (1).csv')
    return df

# Feature names (based on the CSV columns)
FEATURE_COLUMNS = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']

def main():
    st.title("🌧️ Weather Rain Prediction System")
    st.markdown("### Predict whether it will rain based on weather conditions")
    
    # Load model and data
    try:
        model = load_model()
        df = load_data()
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
        st.info("Please ensure 'rain_prediction_model.pkl' and 'data (1).csv' are in the same directory as this script.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📊 Batch Prediction", "📈 Data Analysis"])
    
    # Tab 1: Single Prediction
    with tab1:
        st.header("Make a Single Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.number_input(
                "🌡️ Temperature (°C)",
                min_value=-10.0,
                max_value=50.0,
                value=25.0,
                step=0.1,
                format="%.2f"
            )
            
            humidity = st.number_input(
                "💧 Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=65.0,
                step=0.1,
                format="%.2f"
            )
            
            wind_speed = st.number_input(
                "💨 Wind Speed (m/s)",
                min_value=0.0,
                max_value=30.0,
                value=10.0,
                step=0.1,
                format="%.2f"
            )
        
        with col2:
            cloud_cover = st.number_input(
                "☁️ Cloud Cover (%)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=0.1,
                format="%.2f"
            )
            
            pressure = st.number_input(
                "📊 Pressure (hPa)",
                min_value=900.0,
                max_value=1100.0,
                value=1013.0,
                step=0.1,
                format="%.2f"
            )
        
        # Predict button
        if st.button("🔍 Predict Rain", type="primary", use_container_width=True):
            # Prepare input data
            input_data = np.array([[temperature, humidity, wind_speed, cloud_cover, pressure]])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Get prediction probability if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_data)[0]
                rain_prob = proba[1] if model.classes_[1] == 'rain' else proba[0]
                no_rain_prob = 1 - rain_prob
            else:
                rain_prob = None
            
            # Display result
            st.markdown("---")
            col_result1, col_result2 = st.columns([1, 2])
            
            with col_result1:
                if prediction == 'rain':
                    st.error("🌧️ **Prediction: RAIN**")
                else:
                    st.success("☀️ **Prediction: NO RAIN**")
            
            with col_result2:
                if rain_prob is not None:
                    st.markdown(f"**Confidence:**")
                    st.progress(rain_prob)
                    st.caption(f"Rain probability: {rain_prob:.2%}")
                    st.caption(f"No rain probability: {no_rain_prob:.2%}")
            
            # Show weather condition summary
            st.markdown("---")
            st.markdown("### 📋 Weather Summary")
            
            col_sum1, col_sum2, col_sum3, col_sum4, col_sum5 = st.columns(5)
            
            with col_sum1:
                st.metric("Temperature", f"{temperature:.1f}°C")
            with col_sum2:
                st.metric("Humidity", f"{humidity:.1f}%")
            with col_sum3:
                st.metric("Wind Speed", f"{wind_speed:.1f} m/s")
            with col_sum4:
                st.metric("Cloud Cover", f"{cloud_cover:.1f}%")
            with col_sum5:
                st.metric("Pressure", f"{pressure:.1f} hPa")
    
    # Tab 2: Batch Prediction
    with tab2:
        st.header("Batch Prediction from CSV")
        st.markdown("Upload a CSV file with the following columns:")
        st.code("Temperature, Humidity, Wind_Speed, Cloud_Cover, Pressure")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                batch_df = pd.read_csv(uploaded_file)
                
                # Check if required columns exist
                missing_cols = [col for col in FEATURE_COLUMNS if col not in batch_df.columns]
                
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                    st.info(f"Required columns: {FEATURE_COLUMNS}")
                else:
                    # Make predictions
                    X_batch = batch_df[FEATURE_COLUMNS].values
                    predictions = model.predict(X_batch)
                    
                    # Add predictions to dataframe
                    batch_df['Prediction'] = predictions
                    
                    # Add probabilities if available
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_batch)
                        rain_idx = list(model.classes_).index('rain')
                        batch_df['Rain_Probability'] = proba[:, rain_idx]
                    
                    # Display results
                    st.success(f"✅ Predictions made for {len(batch_df)} rows!")
                    
                    # Show dataframe
                    st.dataframe(batch_df, use_container_width=True)
                    
                    # Download button
                    csv = batch_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name="rain_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Summary statistics
                    st.markdown("### 📊 Summary Statistics")
                    col_batch1, col_batch2 = st.columns(2)
                    
                    with col_batch1:
                        rain_count = sum(batch_df['Prediction'] == 'rain')
                        st.metric("🌧️ Rain Predictions", rain_count)
                        st.metric("☀️ No Rain Predictions", len(batch_df) - rain_count)
                    
                    with col_batch2:
                        if 'Rain_Probability' in batch_df.columns:
                            st.metric("Average Rain Probability", f"{batch_df['Rain_Probability'].mean():.2%}")
                            st.metric("Max Rain Probability", f"{batch_df['Rain_Probability'].max():.2%}")
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # Tab 3: Data Analysis
    with tab3:
        st.header("Data Analysis from Training Data")
        
        # Display basic information
        st.subheader("📊 Dataset Overview")
        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        
        with col_info1:
            st.metric("Total Records", len(df))
        with col_info2:
            st.metric("Features", len(FEATURE_COLUMNS))
        with col_info3:
            rain_count = len(df[df['Rain'] == 'rain'])
            st.metric("Rainy Days", rain_count)
        with col_info4:
            st.metric("No Rain Days", len(df) - rain_count)
        
        # Feature statistics
        st.subheader("📈 Feature Statistics")
        st.dataframe(df[FEATURE_COLUMNS].describe(), use_container_width=True)
        
        # Correlation with target (simplified)
        st.subheader("🔍 Feature Analysis")
        
        # Calculate average values for rain vs no rain
        rain_df = df[df['Rain'] == 'rain']
        no_rain_df = df[df['Rain'] == 'no rain']
        
        comparison_data = []
        for feature in FEATURE_COLUMNS:
            comparison_data.append({
                'Feature': feature,
                'Avg (Rain)': rain_df[feature].mean(),
                'Avg (No Rain)': no_rain_df[feature].mean(),
                'Difference': rain_df[feature].mean() - no_rain_df[feature].mean()
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Sample data
        st.subheader("📋 Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Model information
        st.subheader("🤖 Model Information")
        st.info(f"""
        - **Model Type:** Logistic Regression
        - **Features Used:** {', '.join(FEATURE_COLUMNS)}
        - **Target Classes:** {' → '.join(model.classes_)}
        """)

if __name__ == "__main__":
    main()
