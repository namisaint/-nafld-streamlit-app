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


# === Julius minimal helpers (no secrets/mongo changes) ===
import pandas as _pd
import numpy as _np
import streamlit as _st

EXPECTED_FEATURES = [
    'Gender','Age in years','Race/Ethnicity','Family income ratio','Smoking status','Sleep Disorder Status',
    'Sleep duration (hours/day)','Work schedule duration (hours)','Physical activity (minutes/day)','BMI',
    'Alcohol consumption (days/week)','Alcohol drinks per day','Number of days drank in the past year',
    'Max number of drinks on any single day','Alcohol intake frequency (drinks/day)','Total calorie intake (kcal)',
    'Total protein intake (grams)','Total carbohydrate intake (grams)','Total sugar intake (grams)',
    'Total fiber intake (grams)','Total fat intake (grams)'
]

def _build_X_from_values(values_dict):
    # Cast to numeric where possible; if categorical strings expected, leave as-is to let pipeline handle
    row = {}
    for k in EXPECTED_FEATURES:
        v = values_dict.get(k, None)
        row[k] = v
    X = _pd.DataFrame([row], columns = EXPECTED_FEATURES)
    # Coerce numerics where appropriate but ignore errors (pipeline may handle)
    numeric_like = [
        'Age in years','Family income ratio','Sleep duration (hours/day)','Work schedule duration (hours)',
        'Physical activity (minutes/day)','BMI','Alcohol consumption (days/week)','Alcohol drinks per day',
        'Number of days drank in the past year','Max number of drinks on any single day',
        'Alcohol intake frequency (drinks/day)','Total calorie intake (kcal)','Total protein intake (grams)',
        'Total carbohydrate intake (grams)','Total sugar intake (grams)','Total fiber intake (grams)','Total fat intake (grams)'
    ]
    for c in numeric_like:
        if c in X.columns:
            X[c] = _pd.to_numeric(X[c], errors = 'coerce')
    return X

def _positive_index(model_obj):
    try:
        classes = list(model_obj.classes_)
        if 1 in classes:
            return classes.index(1)
        if '1' in classes:
            return classes.index('1')
        if True in classes:
            return classes.index(True)
        # default to the larger label as positive if numeric
        try:
            nums = [float(x) for x in classes]
            return nums.index(max(nums))
        except Exception:
            return 1 if len(classes) > 1 else 0
    except Exception:
        return 1

def _predict_prob_safe(model_obj, X_df):
    # Prefer predict_proba
    try:
        pos = _positive_index(model_obj)
        return float(model_obj.predict_proba(X_df)[0][pos])
    except Exception:
        # Try pipelines exposing inner classifier
        try:
            clf = model_obj.named_steps.get('classifier', None)
            if clf is not None and hasattr(clf, 'predict_proba'):
                pos = _positive_index(clf)
                return float(clf.predict_proba(X_df)[0][pos])
        except Exception:
            pass
        # Decision function fallback
        try:
            val = float(model_obj.decision_function(X_df)[0])
            val_c = max(min((val + 5.0) / 10.0, 1.0), 0.0)
            return float(val_c)
        except Exception:
            # Binary predict fallback
            try:
                return float(_np.clip(float(model_obj.predict(X_df)[0]), 0.0, 1.0))
            except Exception:
                return 0.5

def render_risk_card(prob):
    try:
        p = float(prob)
    except Exception:
        p = 0.0
    if p < 0.34:
        label = 'Low'; color = '#22c55e'
    elif p < 0.67:
        label = 'Medium'; color = '#f59e0b'
    else:
        label = 'High'; color = '#ef4444'
    html = (
        '<div style="padding:16px;border-radius:10px;background:' + color + '1A;border:2px solid ' + color + ';margin:12px 0">' +
        '<div style="display:flex;justify-content:space-between;align-items:center;font-size:1.05rem;">' +
        '<div><b>Risk level:</b> ' + label + '</div>' +
        '<div><b>' + str(int(round(p*100))) + '%</b></div>' +
        '</div>' +
        '<div style="height:12px;background:#e5e7eb;border-radius:8px;margin-top:10px;">' +
        '<div style="width:' + str(int(round(p*100))) + '%;height:12px;background:' + color + ';border-radius:8px;"></div>' +
        '</div>' +
        '</div>'
    )
    _st.markdown(html, unsafe_allow_html = True)


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
# The connection string is now read securely from st.secrets.
try:
    MONGODB_CONNECTION_STRING = st.secrets["mongo"]["connection_string"]
    DB_NAME = st.secrets["mongo"]["db_name"]
    COLLECTION_NAME = st.secrets["mongo"]["collection_name"]
except KeyError:
    st.error("MongoDB secrets are not configured. Please add your credentials to the Streamlit secrets.")
    st.stop()

@st.cache_resource
def get_mongo_client():
    """
    Connects to the MongoDB Atlas cluster.
    """
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

# Try to read expected columns from model
try:
    MODEL_COLS = list(model.feature_names_in_)
except Exception:
    # Fallback to a hardcoded list if model.feature_names_in_ is not available
    MODEL_COLS = [
        'RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'INDFMPIR', 'ALQ111', 'ALQ121', 'ALQ142',
        'ALQ151', 'ALQ170', 'Is_Smoker_Cat', 'SLQ050', 'SLQ120', 'SLD012', 'DR1TKCAL',
        'DR1TPROT', 'DR1TCARB', 'DR1TSUGR', 'DR1TFIBE', 'DR1TTFAT', 'PAQ620', 'BMXBMI'
    ]

# Helpers for nicer output
def risk_label(p):
    if p < 0.33:
        return "Low", "green"
    if p < 0.66:
        return "Borderline", "orange"
    return "High", "red"

def save_to_mongo(payload, pred, proba):
    if predictions_collection is None:
        return
    try:
        predictions_collection.insert_one({
            "_created_at": datetime.utcnow(),
            "inputs": payload,
            "prediction": pred,
            "probability": proba
        })
        st.success("Saved to MongoDB")
    except Exception as e:
        st.error("Save failed: " + str(e))

# --- UI
st.title("🤖 NAFLD Lifestyle Risk Predictor")
st.subheader("User Data Input")
st.markdown("Enter values for the model's 21 features to get a prediction.")

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    age_years = st.slider("Age in years", 0, 120, 40, 1)
    race = st.selectbox("Race/Ethnicity", [
        "Mexican American", "Other Hispanic", "Non-Hispanic White",
        "Non-Hispanic Black", "Non-Hispanic Asian", "Other Race"
    ], index=0)
    family_income_ratio = st.slider("Family income ratio", 0.0, 10.0, 2.0, 0.1)
    st.info("Family income ratio: Household income divided by the federal poverty level for your household size. • 1.0 means at the poverty threshold. • 1.1 means 10 percent above the poverty threshold. • 2.0 means 200 percent (2x) the poverty threshold. Higher numbers equal higher income relative to poverty level.")
    smoking_status = st.selectbox("Smoking status", ["No", "Yes"], index=0)
with col2:
    sleep_disorder = st.selectbox("Sleep Disorder Status", ["No", "Yes"], index=0)
    sleep_duration_hours = st.slider("Sleep duration (hours/day)", 0.0, 24.0, 8.0, 0.25)
    work_hours = st.slider("Work schedule duration (hours)", 0, 24, 8, 1)
    physical_activity_mins = st.slider("Physical activity (minutes/day)", 0, 1440, 30, 5)
    bmi = st.slider("BMI", 10.0, 60.0, 25.0, 0.1)
with col3:
    alcohol_days_week = st.slider("Alcohol consumption (days/week)", 0, 7, 0, 1)
    alcohol_drinks_per_day = st.slider("Alcohol drinks per day", 0, 50, 0, 1)
    alcohol_days_past_year = st.slider("Number of days drank in the past year", 0, 366, 0, 1)
    alcohol_max_any_day = st.slider("Max number of drinks on any single day", 0, 50, 0, 1)
    alcohol_intake_freq = st.slider("Alcohol intake frequency (drinks/day)", 0.0, 50.0, 0.0, 0.1)

st.subheader("Nutritional Information")
col4, col5, col6 = st.columns(3)
with col4:
    total_calories = st.slider("Total calorie intake (kcal)", 0, 10000, 2000, 50)
    total_protein = st.slider("Total protein intake (grams)", 0, 500, 60, 5)
with col5:
    total_carbs = st.slider("Total carbohydrate intake (grams)", 0, 1000, 250, 5)
    total_sugar = st.slider("Total sugar intake (grams)", 0, 1000, 40, 5)
with col6:
    total_fiber = st.slider("Total fiber intake (grams)", 0, 500, 30, 1)
    total_fat = st.slider("Total fat intake (grams)", 0, 500, 70, 1)


# Build full encoded dict
def encode_inputs():
    races = ["Mexican American", "Other Hispanic", "Non-Hispanic White", "Non-Hispanic Black", "Non-Hispanic Asian", "Other Race"]
    race_one_hot = {}
    for r in races:
        key = "RIDRETH3_" + r.replace(" ", "_")
        race_one_hot[key] = 1 if race == r else 0
    out = {
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
        "DR1TTFAT": int(total_fat)
    }
    out.update(race_one_hot)
    return out

# --- Prediction Logic and Display ---
st.header("Prediction Result")
st.markdown("Adjust the inputs in the sidebar to see the prediction update in real-time.")

if model is not None:
    try:
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

        # Define Risk Level
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
