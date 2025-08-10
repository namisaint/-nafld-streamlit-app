import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import plotly.express as px
import certifi
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO

# --- App Configuration ---
st.set_page_config(
    page_title="Dissertation Model Predictor",
    page_icon="🤖",
    layout="wide"
)

# Force Matplotlib to use Agg backend to prevent rendering issues in Streamlit
plt.style.use('default')
plt.switch_backend('Agg')

# --- MongoDB Connection ---
try:
    MONGODB_CONNECTION_STRING = st.secrets["mongo"]["connection_string"]
    DB_NAME = st.secrets["mongo"]["db_name"]
    COLLECTION_NAME = st.secrets["mongo"]["collection_name"]
except KeyError:
    st.error("MongoDB secrets are not configured. Please add your credentials to the Streamlit secrets.")
    st.stop()

@st.cache_resource
def get_mongo_client():
    try:
        client = MongoClient(MONGODB_CONNECTION_STRING, server_api=ServerApi('1'), tls=True, tlsCAFile=certifi.where())
        client.admin.command('ping')
        return client
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        return None

mongo_client = get_mongo_client()
if mongo_client:
    db = mongo_client[DB_NAME]
    predictions_collection = db[COLLECTION_NAME]
else:
    st.error("Could not connect to the database. The app will not be able to save or load predictions.")
    predictions_collection = None


@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error("Error loading model: " + str(e))
        return None

# Sidebar: Model file
with st.sidebar:
    st.header("Model")
    model_path = st.text_input("Model file path", value="rf_lifestyle_model (1).pkl")

model = load_model(model_path)

# MongoDB Connection Status in Sidebar
with st.sidebar:
    st.header("MongoDB Connection Status")
    try:
        # Check if connected
        if mongo_client is not None and db is not None:
            st.success("Connected to MongoDB")
        else:
            st.warning("Not connected to MongoDB")
    except Exception as e:
        st.error(f"Error: {e}")


# --- Prediction Logic and Display ---
st.header("Prediction Result")
st.markdown("Adjust the inputs in the sidebar to see the prediction update in real-time.")

if model is not None:
    try:
        # Get updated user inputs and encode them for prediction
        full = encode_inputs()
        X = pd.DataFrame([full], columns=MODEL_COLS)
        
        # Predict class probabilities
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        # Safe handling if there are fewer classes (binary classification)
        if len(probabilities) > 1:
            prediction_probability = probabilities[1] * 100  # Probability for class 1 (positive class)
        else:
            prediction_probability = probabilities[0] * 100  # Only one class, use the single probability

        # Define Risk Level based on the probability
        if prediction_probability < 33:
            risk_label = "Low"
            risk_color = "green"
        elif prediction_probability < 67:
            risk_label = "Medium"
            risk_color = "orange"
        else:
            risk_label = "High"
            risk_color = "red"

        # Displaying the result
        col_pred, col_report = st.columns([3, 1])
        with col_pred:
            st.markdown(f"### Predicted NAFLD Risk: **<span style='color:{risk_color}'>{prediction_probability:.2f}% ({risk_label})</span>**", unsafe_allow_html=True)
            st.progress(prediction_probability / 100)  # Progress bar showing the risk percentage
            st.markdown("The prediction is based on the features entered.")
        
        # Helper function for PDF report generation
        def create_pdf(inputs, prediction_prob, risk_label):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="NAFLD Risk Prediction Report", ln=1, align="C")
            pdf.ln(10)
            
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
            pdf.cell(200, 10, txt="---", ln=1)
            
            pdf.set_font("Arial", style='B', size=10)
            pdf.cell(200, 10, txt="Predicted Risk:", ln=1)
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, txt=f"Risk: {prediction_prob:.2f}% ({risk_label})", ln=1)
            pdf.cell(200, 10, txt="", ln=1)

            pdf.set_font("Arial", style='B', size=10)
            pdf.cell(200, 10, txt="Input Data:", ln=1)
            pdf.set_font("Arial", size=10)
            for key, value in inputs.items():
                pdf.cell(200, 5, txt=f"{key}: {value}", ln=1)
            
            return pdf

        with col_report:
            pdf = create_pdf(full, prediction_probability, risk_label)
            st.download_button(
                "Download Report",
                data=BytesIO(pdf.output(dest='S').encode("latin-1")),
                file_name=f"NAFLD_Risk_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please ensure that all 21 features have valid numerical inputs.")
