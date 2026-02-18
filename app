import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

# 1. PAGE SETUP
st.set_page_config(page_title="Anomaly Detector", page_icon="🚨", layout="wide")

st.title("🚨 Transaction Anomaly Detector")
st.markdown("""
This app uses **Machine Learning (Isolation Forest)** to detect unusual patterns in transaction data.
The model analyzes `Amount` and `Distance from Home` to find outliers.
""")

# 2. DATA LOADING
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('transactions_sample.csv')
        # Clean column names
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return None

df = load_data()

if df is not None:
    # 3. SIDEBAR PARAMETERS
    st.sidebar.header("Model Settings")
    # Contamination is the estimated percentage of anomalies in the data
    contamination = st.sidebar.slider("Contamination Rate (Estimated % of outliers)", 0.01, 0.50, 0.10)
    
    # 4. ANOMALY DETECTION LOGIC
    # We use amount and distance as features
    features = ['amount', 'dist_from_home']
    
    # Initialize and fit the model
    model = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly_score'] = model.fit_predict(df[features])
    
    # Isolation Forest returns -1 for anomalies and 1 for normal data
    # Let's map it to something more readable
    df['Status'] = df['anomaly_score'].map({1: 'Normal', -1: 'Suspicious'})

    # 5. VISUALIZATION
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Anomaly Map: Amount vs. Distance")
        fig = px.scatter(
            df, x='amount', y='dist_from_home',
            color='Status',
            symbol='Status',
            color_discrete_map={'Normal': '#00CC96', 'Suspicious': '#EF553B'},
            hover_data=['transaction_id', 'merchant_category'],
            title="Detected Anomalies (Red Points)",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Detection Summary")
        total_anomalies = len(df[df['Status'] == 'Suspicious'])
        st.metric("Suspicious Activities Found", total_anomalies)
        
        st.write("---")
        st.write("**Top Suspicious Transactions:**")
        st.dataframe(df[df['Status'] == 'Suspicious'][['transaction_id', 'amount', 'dist_from_home', 'merchant_category']])

    # 6. GROUND TRUTH COMPARISON (If 'is_fraud' column exists)
    if 'is_fraud' in df.columns:
        st.write("---")
        st.subheader("Model Performance vs. Ground Truth")
        st.write("Comparing the Machine Learning flags against the known 'is_fraud' labels.")
        
        # Create a comparison dataframe
        comparison = df[['transaction_id', 'is_fraud', 'Status']].copy()
        comparison['ML_Detected'] = comparison['Status'].apply(lambda x: 1 if x == 'Suspicious' else 0)
        
        st.table(comparison.head(10))

else:
    st.info("Please ensure 'transactions_sample.csv' is in your repository.")

st.markdown("---")
st.caption("Machine Learning Anomaly Detection System")
