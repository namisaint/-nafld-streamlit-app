import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi
from io import BytesIO
from fpdf import FPDF

# --- Page Setup ---
st.set_page_config(page_title="Dissertation Model Predictor", page_icon="🤖", layout="wide")

import matplotlib.pyplot as plt

# --- MongoDB connection ---
try:
    MONGODB_CONNECTION_STRING = st.secrets["mongo"]["connection_string"]
    DB_NAME = st.secrets["mongo"]["db_name"]
    COLLECTION_NAME = st.secrets["mongo"]["collection_name"]
    client = MongoClient(MONGODB_CONNECTION_STRING, server_api=ServerApi('1'), tls=True, tlsCAFile=certifi.where())
    db = client[DB_NAME]
    predictions_collection = db[COLLECTION_NAME]
except:
    predictions_collection = None
    st.warning("MongoDB connection not configured.")

# --- Load model ---
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except:
        return None

with st.sidebar:
    st.header("Model")
    model_path = st.text_input("Model file path", value="rf_lifestyle_model (1).pkl")
model = load_model(model_path)

# --- Model features ---
try:
    MODEL_COLS = list(model.feature_names_in_)
except Exception:
    MODEL_COLS = [
        'RIAGENDR','RIDAGEYR','RIDRETH3','INDFMPIR','ALQ111','ALQ121','ALQ142',
        'ALQ151','ALQ170','Is_Smoker_Cat','SLQ050','SLQ120','SLD012','DR1TKCAL',
        'DR1TPROT','DR1TCARB','DR1TSUGR','DR1TFIBE','DR1TTFAT','PAQ620','BMXBMI'
    ]

# --- Helper function for encoding inputs ---
def encode_inputs():
    # Fetch inputs from session_state, or default
    gender = st.session_state.get('gender', 'Male')
    age = st.session_state.get('age', 40)
    race = st.session_state.get('race', "Mexican American")
    family_income_ratio = st.session_state.get('family_income_ratio', 2.0)
    smoking_status = st.session_state.get('smoking_status', 'No')
    sleep_disorder = st.session_state.get('sleep_disorder', 'No')
    sleep_hours = st.session_state.get('sleep_hours', 8.0)
    work_hours = st.session_state.get('work_hours', 8)
    physical_activity = st.session_state.get('physical_activity', 30)
    bmi = st.session_state.get('bmi', 25.0)
    alcohol_days = st.session_state.get('alcohol_days', 0)
    drinks_per_day = st.session_state.get('drinks_per_day', 0)
    days_past_year = st.session_state.get('days_past_year', 0)
    max_drinks = st.session_state.get('max_drinks', 0)
    alcohol_freq = st.session_state.get('alcohol_freq', 0.0)
    calories = st.session_state.get('calories', 2000)
    protein = st.session_state.get('protein', 60)
    carbs = st.session_state.get('carbs', 250)
    sugar = st.session_state.get('sugar', 40)
    fiber = st.session_state.get('fiber', 30)
    fat = st.session_state.get('fat', 70)

    # Encode race
    races = ["Mexican American", "Other Hispanic", "Non-Hispanic White", "Non-Hispanic Black", "Non-Hispanic Asian", "Other Race"]
    race_one_hot = {}
    for r in races:
        key = "RIDRETH3_" + r.replace(" ", "_")
        race_one_hot[key] = 1 if race == r else 0

    # Build feature dict
    out = {
    "RIAGENDR": 1 if gender == "Male" else 2,
    "RIDAGEYR": age,
    "INDFMPIR": float(family_income_ratio),
    "Is_Smoker_Cat": 1 if smoking_status == "Yes" else 0,
    "SLD012": 1 if sleep_disorder == "Yes" else 0,
    "SLQ050": float(sleep_hours),
    "SLQ120": int(work_hours),
    "PAQ620": int(physical_activity),
    "BMXBMI": float(bmi),
    "ALQ111": int(alcohol_days),
    "ALQ121": int(drinks_per_day),
    "ALQ142": int(days_past_year),
    "ALQ151": int(max_drinks),
    "ALQ170": float(alcohol_freq),
    "DR1TKCAL": int(calories),
    "DR1TPROT": int(protein),
    "DR1TCARB": int(carbs),
    "DR1TSUGR": int(sugar),
    "DR1TFIBE": int(fiber),
    "DR1TTFAT": int(fat),
}
# Add race dummy variables
races = ["Mexican American", "Other Hispanic", "Non-Hispanic White", "Non-Hispanic Black", "Non-Hispanic Asian", "Other Race"]
for r in races:
    key = "RIDRETH3_" + r.replace(" ", "_")
    out[key] = 1 if race == r else 0

return out
