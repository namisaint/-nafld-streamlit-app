# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime

# -------------------------
# APP CONFIG
# -------------------------
st.set_page_config(page_title="NAFLD Lifestyle Risk Predictor", page_icon="🤖", layout="wide")
plt.style.use("default")
plt.switch_backend("Agg")

st.title("🤖 NAFLD Lifestyle Risk Predictor")
st.caption("Dynamic predictions with feature alignment to your trained model.")

# -------------------------
# MODEL LOADER
# -------------------------
@st.cache_resource
def load_model(path: str):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model from '{path}': {e}")
        return None

with st.sidebar:
    st.header("Model")
    model_path = st.text_input("Model file path", value="rf_lifestyle_model.pkl")
    model = load_model(model_path)

if model is None:
    st.stop()

# Get expected columns exactly from the trained artifact if available
try:
    EXPECTED_COLS = list(model.feature_names_in_)
except Exception:
    # Fallback – update this list to match your training if needed
    EXPECTED_COLS = [
        'RIAGENDR','RIDAGEYR','RIDRETH3','INDFMPIR','ALQ111','ALQ121','ALQ142',
        'ALQ151','ALQ170','Is_Smoker_Cat','SLQ050','SLQ120','SLD012','DR1TKCAL',
        'DR1TPROT','DR1TCARB','DR1TSUGR','DR1TFIBE','DR1TTFAT','PAQ620','BMXBMI'
    ]
    st.warning("Using fallback EXPECTED_COLS. For best results, ensure your model exposes feature_names_in_.")

# -------------------------
# UI INPUTS
# -------------------------
st.subheader("Enter Your Data")

col1, col2, col3 = st.columns(3)

with col1:
    gender_ui = st.selectbox("Gender", ["Male", "Female"], index=0)
    age_years = st.slider("Age in years", 0, 120, 40, 1)
    race_ui = st.selectbox(
        "Race/Ethnicity",
        ["Mexican American", "Other Hispanic", "Non-Hispanic White",
         "Non-Hispanic Black", "Non-Hispanic Asian", "Other Race"],
        index=2
    )
    family_income_ratio = st.slider("Family income ratio", 0.0, 10.0, 2.0, 0.1)
    smoking_status_ui = st.selectbox("Smoking status", ["No", "Yes"], index=0)

with col2:
    sleep_disorder_ui = st.selectbox("Sleep Disorder Status", ["No", "Yes"], index=0)
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

st.subheader("Nutrition")
c4, c5, c6 = st.columns(3)
with c4:
    total_calories = st.slider("Total calorie intake (kcal)", 0, 10000, 2000, 50)
    total_protein = st.slider("Total protein intake (grams)", 0, 500, 60, 5)
with c5:
    total_carbs = st.slider("Total carbohydrate intake (grams)", 0, 1000, 250, 5)
    total_sugar = st.slider("Total sugar intake (grams)", 0, 1000, 40, 5)
with c6:
    total_fiber = st.slider("Total fiber intake (grams)", 0, 500, 30, 1)
    total_fat = st.slider("Total fat intake (grams)", 0, 500, 70, 1)

# -------------------------
# ENCODING HELPERS
# -------------------------

# NHANES codes for RIDRETH3 (you may adjust to your training mapping if different)
RIDRETH3_CODE_MAP = {
    "Mexican American": 1,
    "Other Hispanic": 2,
    "Non-Hispanic White": 3,
    "Non-Hispanic Black": 4,
    "Non-Hispanic Asian": 6,
    "Other Race": 7
}

# Possible one-hot column names if your model used get_dummies on the string label
# We create names that you likely had in training; adapt if your saved model uses different naming.
RACE_ONEHOT_PREFIX = "RIDRETH3_"
RACE_ONEHOT_NAMES = {
    "Mexican American": f"{RACE_ONEHOT_PREFIX}Mexican_American",
    "Other Hispanic": f"{RACE_ONEHOT_PREFIX}Other_Hispanic",
    "Non-Hispanic White": f"{RACE_ONEHOT_PREFIX}Non-Hispanic_White",
    "Non-Hispanic Black": f"{RACE_ONEHOT_PREFIX}Non-Hispanic_Black",
    "Non-Hispanic Asian": f"{RACE_ONEHOT_PREFIX}Non-Hispanic_Asian",
    "Other Race": f"{RACE_ONEHOT_PREFIX}Other_Race",
}

def expects_numeric_race(expected_cols):
    """Return True if model expects a single 'RIDRETH3' column (numeric)."""
    return "RIDRETH3" in expected_cols and not any(col.startswith(RACE_ONEHOT_PREFIX) for col in expected_cols)

def expects_onehot_race(expected_cols):
    """Return True if model expects one-hot columns like 'RIDRETH3_Non-Hispanic_White'."""
    return any(col.startswith(RACE_ONEHOT_PREFIX) for col in expected_cols)

def encode_row(expected_cols):
    """
    Build a single-row DataFrame whose columns exactly match 'expected_cols',
    with values taken from the UI inputs, using the right encoding.
    Any missing expected feature is filled with 0.
    """
    row = {col: 0 for col in expected_cols}  # start with safe defaults

    # Gender
    if "RIAGENDR" in expected_cols:
        row["RIAGENDR"] = 1 if gender_ui == "Male" else 2  # NHANES codes

    # Age
    if "RIDAGEYR" in expected_cols:
        row["RIDAGEYR"] = age_years

    # Race/Ethnicity
    if expects_numeric_race(expected_cols):
        row["RIDRETH3"] = RIDRETH3_CODE_MAP[race_ui]
    elif expects_onehot_race(expected_cols):
        # Zero already; set the active one-hot to 1 if present
        active = RACE_ONEHOT_NAMES[race_ui]
        if active in row:
            row[active] = 1

    # Family income ratio
    if "INDFMPIR" in expected_cols:
        row["INDFMPIR"] = float(family_income_ratio)

    # Smoking
    if "Is_Smoker_Cat" in expected_cols:
        row["Is_Smoker_Cat"] = 1 if smoking_status_ui == "Yes" else 0

    # Sleep
    if "SLD012" in expected_cols:
        row["SLD012"] = 1 if sleep_disorder_ui == "Yes" else 0
    if "SLQ050" in expected_cols:
        row["SLQ050"] = float(sleep_duration_hours)
    if "SLQ120" in expected_cols:
        row["SLQ120"] = int(work_hours)

    # Physical activity
    if "PAQ620" in expected_cols:
        row["PAQ620"] = int(physical_activity_mins)

    # BMI
    if "BMXBMI" in expected_cols:
        row["BMXBMI"] = float(bmi)

    # Alcohol (NHANES ALQ variables)
    if "ALQ111" in expected_cols:
        row["ALQ111"] = int(alcohol_days_week)
    if "ALQ121" in expected_cols:
        row["ALQ121"] = int(alcohol_drinks_per_day)
    if "ALQ142" in expected_cols:
        row["ALQ142"] = int(alcohol_days_past_year)
    if "ALQ151" in expected_cols:
        row["ALQ151"] = int(alcohol_max_any_day)
    if "ALQ170" in expected_cols:
        row["ALQ170"] = float(alcohol_intake_freq)

    # Diet
    if "DR1TKCAL" in expected_cols:
        row["DR1TKCAL"] = int(total_calories)
    if "DR1TPROT" in expected_cols:
        row["DR1TPROT"] = int(total_protein)
    if "DR1TCARB" in expected_cols:
        row["DR1TCARB"] = int(total_carbs)
    if "DR1TSUGR" in expected_cols:
        row["DR1TSUGR"] = int(total_sugar)
    if "DR1TFIBE" in expected_cols:
        row["DR1TFIBE"] = int(total_fiber)
    if "DR1TTFAT" in expected_cols:
        row["DR1TTFAT"] = int(total_fat)

    # Build DataFrame with exact column order
    X = pd.DataFrame([row], columns=expected_cols)
    return X

# -------------------------
# PREDICTION
# -------------------------
st.header("Prediction Result")
X = encode_row(EXPECTED_COLS)

pred = None
proba = None
error = None

try:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        # Pick the positive class index robustly
        classes = getattr(model, "classes_", [0, 1])
        pos_idx = 1
        try:
            if 1 in classes:
                pos_idx = list(classes).index(1)
            elif True in classes:
                pos_idx = list(classes).index(True)
            elif "1" in classes:
                pos_idx = list(classes).index("1")
            else:
                # fall back to the max numeric class
                nums = [float(c) for c in classes]
                pos_idx = int(np.argmax(nums))
        except Exception:
            pos_idx = 1 if len(classes) > 1 else 0

        proba = float(probs[0][pos_idx])
        pred = int(model.predict(X)[0]) if hasattr(model, "predict") else (1 if proba >= 0.5 else 0)
    else:
        # fallback if no predict_proba
        yhat = model.predict(X) if hasattr(model, "predict") else [0]
        pred = int(yhat[0])
        proba = float(pred)
except Exception as e:
    error = str(e)

if error:
    st.error(f"❌ Prediction error: {error}")
    st.stop()

risk_pct = proba * 100.0
risk_label = "High Risk" if risk_pct >= 70 else "Moderate Risk" if risk_pct >= 30 else "Low Risk"
risk_color = "red" if risk_pct >= 70 else "orange" if risk_pct >= 30 else "green"

col_pred, col_info = st.columns([3, 2])
with col_pred:
    st.markdown(
        f"### Predicted NAFLD Risk: "
        f"**<span style='color:{risk_color}'>{risk_pct:.2f}% ({risk_label})</span>**",
        unsafe_allow_html=True
    )
    st.progress(min(max(risk_pct/100.0, 0.0), 1.0))
    st.caption("Prediction updates live as you change inputs.")

with col_info:
    st.write("**Model Inputs Used (matching expected features):**")
    st.dataframe(X.T.rename(columns={0: "value"}))

# -------------------------
# OPTIONAL: SIMPLE SHAP (best-effort)
# Only runs if it's a tree-based model & explainer works
# -------------------------
with st.expander("Explain prediction (best-effort SHAP)"):
    st.caption("If your model is tree-based and saved with full pipeline, SHAP bar will appear.")
    try:
        import shap
        # Try TreeExplainer first (works for many tree models)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        # Handle binary case list
        vals = shap_vals[1] if isinstance(shap_vals, list) and len(shap_vals) > 1 else shap_vals
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.summary_plot(vals, X, plot_type="bar", show=False)
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.info(f"SHAP not available for this model: {e}")

# -------------------------
# PDF REPORT (optional)
# -------------------------
def make_pdf(inputs_df: pd.DataFrame, risk_pct: float, risk_label: str) -> bytes:
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "NAFLD Risk Prediction Report", ln=1, align="C")
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    pdf.ln(4)
    pdf.set_font("Arial", style="B", size=11)
    pdf.cell(0, 8, f"Predicted Risk: {risk_pct:.2f}% ({risk_label})", ln=1)
    pdf.ln(2)
    pdf.set_font("Arial", style="B", size=11)
    pdf.cell(0, 8, "Inputs:", ln=1)
    pdf.set_font("Arial", size=9)
    for k, v in inputs_df.iloc[0].items():
        pdf.cell(0, 6, f"{k}: {v}", ln=1)
    return pdf.output(dest="S").encode("latin-1")

with st.sidebar:
    st.markdown("---")
    if st.button("Generate PDF Report"):
        try:
            pdf_bytes = make_pdf(X, risk_pct, risk_label)
            st.download_button(
                "Download Report",
                data=BytesIO(pdf_bytes),
                file_name=f"NAFLD_Risk_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
