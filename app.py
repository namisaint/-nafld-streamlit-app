# app.py
import os
from io import BytesIO
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import certifi
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
from pymongo.mongo_client import MongoClient

# ---------------- App Configuration ----------------
st.set_page_config(page_title="NAFLD Lifestyle Model Predictor", page_icon="🧠", layout="wide")

# Matplotlib backend for Streamlit
plt.style.use("default")
plt.switch_backend("Agg")

# ---------------- MongoDB Connection ----------------
predictions_collection = None
with st.sidebar:
    st.header("Connections")
try:
    MONGODB_CONNECTION_STRING = st.secrets["mongo"]["connection_string"]
    DB_NAME = st.secrets["mongo"]["db_name"]
    COLLECTION_NAME = st.secrets["mongo"]["collection_name"]
    _client = MongoClient(MONGODB_CONNECTION_STRING, tls=True, tlsCAFile=certifi.where())
    _db = _client.get_database(DB_NAME)
    predictions_collection = _db[COLLECTION_NAME]
    st.sidebar.success("MongoDB Connected ✅")
except Exception as e:
    st.sidebar.error("MongoDB Connection Failed ❌")
    st.sidebar.caption(f"{e}")
    predictions_collection = None

# ---------------- Utilities ----------------
@st.cache_resource
def load_model(path: str):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Error loading model from '{path}': {e}")
        return None

def risk_bucket(p_float: float):
    """p_float in [0,1]"""
    if p_float < 0.33:
        return "Low", "green"
    if p_float < 0.66:
        return "Moderate", "orange"
    return "High", "red"

def save_to_mongo(payload: dict, pred: int, proba_pct: float):
    if predictions_collection is None:
        return
    try:
        predictions_collection.insert_one({
            "_created_at": datetime.utcnow(),
            "inputs": payload,
            "prediction": int(pred),
            "probability_pct": float(proba_pct)
        })
        st.success("Saved to MongoDB")
    except Exception as e:
        st.warning(f"Skip saving to MongoDB: {e}")

def get_expected_columns(_model) -> list:
    # Try direct attribute
    if hasattr(_model, "feature_names_in_"):
        return list(_model.feature_names_in_)
    # Try pipeline final step
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(_model, Pipeline) and hasattr(_model[-1], "feature_names_in_"):
            return list(_model[-1].feature_names_in_)
    except Exception:
        pass
    st.error(
        "Model is missing 'feature_names_in_'. "
        "Please export and load the trained Pipeline/estimator that preserves column names."
    )
    st.stop()

@st.cache_resource
def get_explainer(model_path: str, _model):
    # Cache per model path to avoid re-fitting on every rerun
    try:
        return shap.TreeExplainer(_model)
    except Exception as e:
        st.warning(f"SHAP explainer init failed: {e}")
        return None

# ---------------- Sidebar: Model Selection ----------------
with st.sidebar:
    st.header("Model")
    model_path = st.text_input("Model file path", value="rf_lifestyle_model.pkl", help="Path relative to this app.")
    st.caption("Tip: ensure this is the exact artifact used in training (same features/order).")

model = load_model(model_path)
if model is None:
    st.stop()

EXPECTED_COLS = get_expected_columns(model)

# ---------------- Input Form ----------------
st.title("NAFLD Risk Prediction (Lifestyle Model)")
st.caption("Random Forest-based prediction from lifestyle & sociodemographic inputs.")

st.subheader("Enter Your Data")

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    age_years = st.slider("Age in years", 0, 120, 40, 1)
    race = st.selectbox(
        "Race/Ethnicity",
        ["Mexican American", "Other Hispanic", "Non-Hispanic White", "Non-Hispanic Black", "Non-Hispanic Asian", "Other Race"],
        index=0
    )
    family_income_ratio = st.slider("Family income ratio", 0.0, 10.0, 2.0, 0.1)
    st.info(
        "Family income ratio = Household income / Poverty threshold for your household size.\n"
        "• 1.0 = at the poverty threshold • 2.0 = 200% of the threshold."
    )
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

# ---------------- Encoding & Alignment ----------------
def encode_inputs_to_expected() -> pd.DataFrame:
    """Build a single-row DataFrame that matches EXACTLY the model's EXPECTED_COLS."""
    # Detect whether model expects one-hot race columns like RIDRETH3_*
    expected_race_cols = [c for c in EXPECTED_COLS if c.startswith("RIDRETH3_")]
    payload = {}

    # Race encoding
    if expected_race_cols:
        # Map incoming selection to expected one-hot columns
        normalized_choice = race.lower().replace("-", " ").strip()
        for rc in expected_race_cols:
            token = rc.replace("RIDRETH3_", "").replace("_", " ").lower().strip()
            payload[rc] = 1 if token == normalized_choice else 0
    else:
        # Numeric code fallback — must match your training mapping exactly
        race_code_map = {
            "Mexican American": 1,
            "Other Hispanic": 2,
            "Non-Hispanic White": 3,
            "Non-Hispanic Black": 4,
            "Non-Hispanic Asian": 6,
            "Other Race": 7
        }
        payload["RIDRETH3"] = race_code_map[race]

    # Other inputs (must match training encodings)
    payload.update({
        "RIAGENDR": 1 if gender == "Male" else 2,
        "RIDAGEYR": int(age_years),
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
    })

    # Create DF and align to EXPECTED_COLS
    X = pd.DataFrame([payload])

    # Add missing expected cols as 0 (e.g., one-hot columns the form didn't create)
    missing = [c for c in EXPECTED_COLS if c not in X.columns]
    for c in missing:
        X[c] = 0

    # Drop extras and reorder to exact expected order
    X = X[EXPECTED_COLS]

    # Diagnostics — should be clean now
    extra = [c for c in X.columns if c not in EXPECTED_COLS]
    still_missing = [c for c in EXPECTED_COLS if c not in X.columns]
    if still_missing:
        st.error(f"Feature mismatch detected.\nMissing: {still_missing}\nExtra: {extra}")
        st.stop()

    with st.expander("🔎 Debug: Model & Feature Diagnostics", expanded=False):
        st.write("Model path:", model_path)
        st.write("Expected feature count:", len(EXPECTED_COLS))
        st.write("First 10 expected cols:", EXPECTED_COLS[:10])
        st.write("Input dtypes:", X.dtypes.apply(lambda s: s.name).to_dict())
        st.write("Input head:", X.head())

    return X

# ---------------- Prediction & Display ----------------
st.header("Prediction Result")

if st.button("Get Prediction"):
    try:
        X = encode_inputs_to_expected()

        # Predict probability
        y_proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if isinstance(proba, list) and len(proba) > 1:
                y_proba = float(proba[1][0])
            else:
                y_proba = float(proba[:, 1][0]) if proba.shape[1] > 1 else float(proba[:, 0][0])
        y_pred = int(model.predict(X)[0])

        pct = (y_proba * 100.0) if y_proba is not None else 0.0
        label, color = risk_bucket((pct / 100.0))

        col_pred, col_report = st.columns([3, 1])
        with col_pred:
            st.markdown(
                f"### Predicted NAFLD Risk: **<span style='color:{color}'>{pct:.2f}% ({label})</span>**",
                unsafe_allow_html=True
            )
            st.progress(pct / 100.0)
            st.caption("Prediction is based on your current inputs.")

        # -------- PDF Report --------
        def create_pdf(inputs_df: pd.DataFrame, prediction_prob_pct: float, risk_label_str: str):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="NAFLD Risk Prediction Report", ln=1, align="C")
            pdf.ln(8)
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 8, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
            pdf.cell(200, 8, txt="----------------------------------------", ln=1)
            pdf.set_font("Arial", style="B", size=10)
            pdf.cell(200, 8, txt="Predicted Risk:", ln=1)
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 8, txt=f"Risk: {prediction_prob_pct:.2f}% ({risk_label_str})", ln=1)
            pdf.ln(4)
            pdf.set_font("Arial", style="B", size=10)
            pdf.cell(200, 8, txt="Input Data:", ln=1)
            pdf.set_font("Arial", size=10)
            row = inputs_df.iloc[0]
            for key in inputs_df.columns:
                pdf.cell(200, 6, txt=f"{key}: {row[key]}", ln=1)
            return pdf

        with col_report:
            pdf = create_pdf(X, pct, label)
            st.download_button(
                "Download Report",
                data=BytesIO(pdf.output(dest="S").encode("latin-1")),
                file_name=f"NAFLD_Risk_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )

        # Save result (non-blocking for prediction)
        try:
            save_to_mongo(X.to_dict(orient="records")[0], y_pred, pct)
        except Exception as e:
            st.warning(f"Could not save to MongoDB: {e}")

        # -------- Advanced Analysis (SHAP) --------
        with st.expander("Show Advanced Analysis"):
            st.subheader("Model Explainability (SHAP)")
            explainer = get_explainer(model_path, model)
            if explainer is not None:
                try:
                    shap_vals = explainer.shap_values(X)
                    # For RandomForestClassifier, shap_values is often a list [class0, class1]
                    if isinstance(shap_vals, list) and len(shap_vals) > 1:
                        sv_row = np.array(shap_vals[1][0])
                    else:
                        sv_row = np.array(shap_vals[0])

                    # Build a simple bar chart for the single-row contributions
                    # Sort by absolute contribution
                    abs_order = np.argsort(np.abs(sv_row))[::-1]
                    sorted_vals = sv_row[abs_order]
                    sorted_feats = np.array(EXPECTED_COLS)[abs_order]

                    top_n = min(20, len(sorted_feats))  # show top 20 features
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(range(top_n), sorted_vals[:top_n])
                    ax.set_yticks(range(top_n))
                    ax.set_yticklabels(sorted_feats[:top_n])
                    ax.invert_yaxis()
                    ax.set_xlabel("SHAP value (impact on model output)")
                    ax.set_title("Top feature contributions (single input)")
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"SHAP plotting failed: {e}")
            else:
                st.info("SHAP explainer unavailable for this model.")

            st.markdown("---")
            st.subheader("Input Data Summary")
            st.dataframe(X.T)

            st.subheader("Saved Predictions (MongoDB)")
            if predictions_collection is not None:
                if st.button("Refresh Saved Predictions"):
                    try:
                        saved_predictions = list(predictions_collection.find())
                        if saved_predictions:
                            df_predictions = pd.DataFrame(saved_predictions).drop(columns=["_id"], errors="ignore")
                            st.dataframe(df_predictions)
                        else:
                            st.info("No saved predictions found.")
                    except Exception as e:
                        st.error(f"Error retrieving predictions: {e}")
            else:
                st.warning("Not connected to MongoDB.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Tip: Make sure your model artifact and the app use the exact same feature names and order.")
