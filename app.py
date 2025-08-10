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


# --- Function to Encode Inputs ---
EXPECTED_FEATURES = [
    'Gender','Age in years','Race/Ethnicity','Family income ratio','Smoking status','Sleep Disorder Status',
    'Sleep duration (hours/day)','Work schedule duration (hours)','Physical activity (minutes/day)','BMI',
    'Alcohol consumption (days/week)','Alcohol drinks per day','Number of days drank in the past year',
    'Max number of drinks on any single day','Alcohol intake frequency (drinks/day)','Total calorie intake (kcal)',
    'Total protein intake (grams)','Total carbohydrate intake (grams)','Total sugar intake (grams)',
    'Total fiber intake (grams)','Total fat intake (grams)'
]

def encode_inputs():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"], index=0)
    age_years = st.sidebar.slider("Age in years", 0, 120, 40, 1)
    race = st.sidebar.selectbox("Race/Ethnicity", [
        "Mexican American", "Other Hispanic", "Non-Hispanic White",
        "Non-Hispanic Black", "Non-Hispanic Asian", "Other Race"
    ], index=0)
    family_income_ratio = st.sidebar.slider("Family income ratio", 0.0, 10.0, 2.0, 0.1)
    smoking_status = st.sidebar.selectbox("Smoking status", ["No", "Yes"], index=0)
    sleep_disorder = st.sidebar.selectbox("Sleep Disorder Status", ["No", "Yes"], index=0)
    sleep_duration_hours = st.sidebar.slider("Sleep duration (hours/day)", 0.0, 24.0, 8.0, 0.25)
    work_hours = st.sidebar.slider("Work schedule duration (hours)", 0, 24, 8, 1)
    physical_activity_mins = st.sidebar.slider("Physical activity (minutes/day)", 0, 1440, 30, 5)
    bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0, 0.1)
    alcohol_days_week = st.sidebar.slider("Alcohol consumption (days/week)", 0, 7, 0, 1)
    alcohol_drinks_per_day = st.sidebar.slider("Alcohol drinks per day", 0, 50, 0, 1)
    alcohol_days_past_year = st.sidebar.slider("Number of days drank in the past year", 0, 366, 0, 1)
    alcohol_max_any_day = st.sidebar.slider("Max number of drinks on any single day", 0, 50, 0, 1)
    alcohol_intake_freq = st.sidebar.slider("Alcohol intake frequency (drinks/day)", 0.0, 50.0, 0.0, 0.1)
    total_calories = st.sidebar.slider("Total calorie intake (kcal)", 0, 10000, 2000, 50)
    total_protein = st.sidebar.slider("Total protein intake (grams)", 0, 500, 60, 5)
    total_carbs = st.sidebar.slider("Total carbohydrate intake (grams)", 0, 1000, 250, 5)
    total_sugar = st.sidebar.slider("Total sugar intake (grams)", 0, 1000, 40, 5)
    total_fiber = st.sidebar.slider("Total fiber intake (grams)", 0, 500, 30, 1)
    total_fat = st.sidebar.slider("Total fat intake (grams)", 0, 500, 70, 1)

    # One-hot encode race categories
    race_one_hot = {}
    for r in ["Mexican American", "Other Hispanic", "Non-Hispanic White", "Non-Hispanic Black", "Non-Hispanic Asian", "Other Race"]:
        key = "RIDRETH3_" + r.replace(" ", "_")
        race_one_hot[key] = 1 if race == r else 0

    return {
        "RIAGENDR": 1 if gender == "Male" else 2,
        "RIDAGEYR": age_years,
        "INDFMPIR": float(family_income_ratio),
        "Is_Smoker_Cat": 1 if smoking_status == "Yes" else 0,
        "SLD012": 1 if sleep_disorder == "Yes" else 0,
        "SLQ050": float(sleep_duration_hours),
        "SLQ120": int(work_hours),
        "PAQ620": int(physical_activity_mins),
        "BMXBMI": float(bmi),
        "ALQ111": int(alcohol_days_week),
        "ALQ121": int(alcohol_drinks_per_day),
        "ALQ142": int(alcohol_days_past_year),
        "ALQ151": int(alcohol_max_any_day),
        "ALQ170": float(alcohol_intake_freq),
        "DR1TKCAL": int(total_calories),
        "DR1TPROT": int(total_protein),
        "DR1TCARB": int(total_carbs),
        "DR1TSUGR": int(total_sugar),
        "DR1TFIBE": int(total_fiber),
        "DR1TTFAT": int(total_fat),
        **race_one_hot
    }

# --- Prediction Logic and Display ---
st.header("Prediction Result")
st.markdown("Adjust the inputs in the sidebar to see the prediction update in real-time.")

if model is not None:
    try:
        # Get updated user inputs and encode them for prediction
        full = encode_inputs()
        X = pd.DataFrame([full], columns=model.feature_names_in_)

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

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please ensure that all 21 features have valid numerical inputs.")
