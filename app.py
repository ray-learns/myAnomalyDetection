import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

# 1. PAGE SETUP
st.set_page_config(page_title="Custom Anomaly Detector", page_icon="🚨", layout="wide")

st.title("🚨 Anomaly Detection Tool")
st.markdown("""
Upload your data file in CSV format, select your features, and this tool will help to identify outliers in your data.
""")

# 2. FILE UPLOADER
uploaded_file = st.file_uploader("Upload your transaction or sensor data (CSV)", type=['csv'])



if uploaded_file is not None:
    try:
        # Load the user's data
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip() # Clean column names
        
        st.write("### Data Preview", df.head())

        # 3. FEATURE SELECTION
        # Identify numeric columns only for the ML model
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.error("The uploaded file needs at least two numeric columns for analysis.")
        else:
            st.sidebar.header("Model Settings")
            
            # Allow user to pick features
            feature_x = st.sidebar.selectbox("Select First Feature (X-axis)", numeric_cols, index=0)
            feature_y = st.sidebar.selectbox("Select Second Feature (Y-axis)", numeric_cols, index=min(1, len(numeric_cols)-1))
            
            contamination = st.sidebar.slider("Contamination Rate (% of expected outliers)", 0.01, 0.20, 0.05)

            # 4. ANOMALY DETECTION LOGIC
            model = IsolationForest(contamination=contamination, random_state=42)
            
            # We train on the two selected columns
            features = [feature_x, feature_y]
            df['anomaly_score'] = model.fit_predict(df[features])
            
            # Map results: -1 is Anomaly, 1 is Normal
            df['Status'] = df['anomaly_score'].map({1: 'Normal', -1: 'Suspicious'})

            # 5. VISUALIZATION
            
            
            fig = px.scatter(
                df, x=feature_x, y=feature_y,
                color='Status',
                symbol='Status',
                color_discrete_map={'Normal': '#00CC96', 'Suspicious': '#EF553B'},
                hover_data=df.columns.tolist(),
                title=f"Anomaly Detection: {feature_x} vs {feature_y}",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

            # 6. RESULTS TABLE
            st.write("---")
            st.subheader("Detected Anomalies")
            anomalies = df[df['Status'] == 'Suspicious']
            
            if not anomalies.empty:
                st.write(f"Found **{len(anomalies)}** suspicious data points.")
                st.dataframe(anomalies)
                
                # Allow user to download the results
                csv = anomalies.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Anomaly List",
                    data=csv,
                    file_name='detected_anomalies.csv',
                    mime='text/csv',
                )
            else:
                st.success("No anomalies detected with current settings.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

else:
    st.info("👆 Please upload a CSV file to begin the analysis.")

st.markdown("---")
st.caption("Machine Learning Anomaly Detection System | Powered by Isolation Forest")
