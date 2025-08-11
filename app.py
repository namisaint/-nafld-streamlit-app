# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.set_page_config(page_title="NAFLD Predictor", page_icon="🤖", layout="wide")
st.title("🤖 NAFLD Lifestyle Risk Predictor")

# -----------------------
# Constants: EXACT 21 FEATURES (order matters)
# -----------------------
EXPECTED_COLS = [
    'RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'INDFMPIR',
    'ALQ111', 'ALQ121', 'ALQ142', 'ALQ151', 'ALQ170',
    'Is_Smoker_Cat',
    'SLQ050', 'SLQ120', 'SLD012',
    'DR1TKCAL', 'DR1TPROT', 'DR1TCARB', 'DR1TSUGR', 'DR1TFIBE', 'DR1TTFAT',
    'PAQ620', 'BMXBMI'
]

# NHANES coding helpers
RIDRETH3_CODE_MAP = {
    "Mexican American": 1,
    "Other Hispanic": 2,
    "Non-Hispanic White": 3,
    "Non-Hispanic Black": 4,
    "Non-Hispanic Asian": 6,
    "Other Race": 7
}

# -----------------------
# Load model (cached)
# -----------------------
@st.cache_resource
def load_model(path: str = "rf_lifestyle_model.pkl"):
    return joblib.load(path)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Cannot load 'rf_lifestyle_model.pkl': {e}")
    st.stop()

# Optional sanity check against model.feature_names_in_
with st.sidebar:
    st.markdown("### Expected features (app)")
    st.write(EXPECTED_COLS)
    try:
        model_cols = list(model.feature_names_in_)
        st.markdown("### Features in model")
        st.write(model_cols)
        if model_cols != EXPECTED_COLS:
            st.warning("Model feature list differs from app's 21 features. Predictions may error or be wrong.")
    except Exception:
        st.info("Model does not expose feature_names_in_ (ok if you saved only the estimator).")

# -----------------------
# Optional: MongoDB via secrets
# -----------------------
HAS_MONGO = False
predictions_collection = None
mongo_status = "disabled"

if "mongo" in st.secrets:
    try:
        from pymongo.mongo_client import MongoClient
        from pymongo.server_api import ServerApi
        import certifi

        @st.cache_resource
        def get_mongo_client():
            conn = st.secrets["mongo"]["connection_string"]
            client = MongoClient(conn, server_api=ServerApi('1'), tls=True, tlsCAFile=certifi.where())
            client.admin.command("ping")
            return client

        _client = get_mongo_client()
        _db = _client[st.secrets["mongo"]["db_name"]]
        predictions_collection = _db[st.secrets["mongo"]["collection_name"]]
        HAS_MONGO = True
        mongo_status = "connected"
    except Exception as e:
        mongo_status = f"error: {e}"

with st.sidebar:
    st.subheader("MongoDB")
    st.success(f"MongoDB {mongo_status}") if HAS_MONGO else st.info(f"MongoDB {mongo_status}")

# -----------------------
# UI: Collect inputs for the 21 features
# -----------------------
st.header("Inputs")

c1, c2, c3 = st.columns(3)

with c1:
    gender_ui = st.selectbox("Gender", ["Male", "Female"], index=0)
    age_ui = st.slider("Age in years (RIDAGEYR)", 0, 120, 40, 1)
    race_ui = st.selectbox(
        "Race/Ethnicity (RIDRETH3)",
        list(RIDRETH3_CODE_MAP.keys()),
        index=2
    )
    income_ui = st.slider("Family income ratio (INDFMPIR)", 0.0, 10.0, 2.0, 0.1)
    smoker_ui = st.selectbox("Smoking status (Is_Smoker_Cat)", ["No", "Yes"], index=0)

with c2:
    sleep_hours_ui = st.slider("Sleep duration hours/day (SLQ050)", 0.0, 24.0, 8.0, 0.25)
    work_hours_ui = st.slider("Work schedule duration hours (SLQ120)", 0, 24, 8, 1)
    sleep_disorder_ui = st.selectbox("Sleep Disorder Status (SLD012)", ["No", "Yes"], index=0)
    pa_mins_ui = st.slider("Physical activity minutes/day (PAQ620)", 0, 1440, 30, 5)
    bmi_ui = st.slider("BMI (BMXBMI)", 10.0, 60.0, 25.0, 0.1)

with c3:
    alq111_ui = st.slider("ALQ111: Alcohol days/week", 0, 7, 0, 1)
    alq121_ui = st.slider("ALQ121: Alcohol drinks/day", 0, 50, 0, 1)
    alq142_ui = st.slider("ALQ142: Days drank in past year", 0, 366, 0, 1)
    alq151_ui = st.slider("ALQ151: Max drinks on any day", 0, 50, 0, 1)
    alq170_ui = st.slider("ALQ170: Intake freq (drinks/day)", 0.0, 50.0, 0.0, 0.1)

st.subheader("Nutrition")
n1, n2, n3 = st.columns(3)
with n1:
    kcal_ui = st.slider("DR1TKCAL: Total kcal", 0, 10000, 2000, 50)
    prot_ui = st.slider("DR1TPROT: Protein (g)", 0, 500, 60, 5)
with n2:
    carb_ui = st.slider("DR1TCARB: Carbs (g)", 0, 1000, 250, 5)
    sug_ui = st.slider("DR1TSUGR: Sugar (g)", 0, 1000, 40, 5)
with n3:
    fib_ui = st.slider("DR1TFIBE: Fiber (g)", 0, 500, 30, 1)
    fat_ui = st.slider("DR1TTFAT: Fat (g)", 0, 500, 70, 1)

# -----------------------
# Build the row EXACTLY in EXPECTED_COLS order
# -----------------------
row = {
    'RIAGENDR': 1 if gender_ui == "Male" else 2,
    'RIDAGEYR': int(age_ui),
    'RIDRETH3': int(RIDRETH3_CODE_MAP[race_ui]),
    'INDFMPIR': float(income_ui),

    'ALQ111': int(alq111_ui),
    'ALQ121': int(alq121_ui),
    'ALQ142': int(alq142_ui),
    'ALQ151': int(alq151_ui),
    'ALQ170': float(alq170_ui),

    'Is_Smoker_Cat': 1 if smoker_ui == "Yes" else 0,

    'SLQ050': float(sleep_hours_ui),
    'SLQ120': int(work_hours_ui),
    'SLD012': 1 if sleep_disorder_ui == "Yes" else 0,

    'DR1TKCAL': int(kcal_ui),
    'DR1TPROT': int(prot_ui),
    'DR1TCARB': int(carb_ui),
    'DR1TSUGR': int(sug_ui),
    'DR1TFIBE': int(fib_ui),
    'DR1TTFAT': int(fat_ui),

    'PAQ620': int(pa_mins_ui),
    'BMXBMI': float(bmi_ui),
}

# Force correct order/columns (and only these 21)
X = pd.DataFrame([row], columns=EXPECTED_COLS)

# -----------------------
# Predict
# -----------------------
st.header("Prediction")

def positive_class_index(classes):
    try:
        if 1 in classes: return list(classes).index(1)
        if True in classes: return list(classes).index(True)
        if "1" in classes: return list(classes).index("1")
        # fall back: largest numeric as positive
        nums = [float(c) for c in classes]
        return int(np.argmax(nums))
    except Exception:
        return 1 if len(classes) > 1 else 0

try:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        classes = getattr(model, "classes_", [0, 1])
        pos_idx = positive_class_index(classes)
        proba = float(probs[0][pos_idx])
        yhat = int(model.predict(X)[0]) if hasattr(model, "predict") else (1 if proba >= 0.5 else 0)
    else:
        y_arr = model.predict(X) if hasattr(model, "predict") else [0]
        yhat = int(y_arr[0])
        proba = float(yhat)
except Exception as e:
    st.error(f"❌ Prediction error: {e}")
    st.write("Inputs sent to model (debug):")
    st.dataframe(X.T.rename(columns={0: "value"}))
    st.stop()

risk_pct = proba * 100.0
risk_label = "High Risk" if risk_pct >= 70 else "Moderate Risk" if risk_pct >= 30 else "Low Risk"
risk_color = "red" if risk_pct >= 70 else "orange" if risk_pct >= 30 else "green"

colA, colB = st.columns([3, 2])
with colA:
    st.markdown(
        f"### Predicted NAFLD Risk: **<span style='color:{risk_color}'>{risk_pct:.2f}% ({risk_label})</span>**",
        unsafe_allow_html=True
    )
    st.progress(min(max(risk_pct/100.0, 0.0), 1.0))
with colB:
    st.write("**Inputs sent to the model (exact columns):**")
    st.dataframe(X.T.rename(columns={0: "value"}))

# -----------------------
# Save to MongoDB (optional)
# -----------------------
def save_prediction(doc):
    if not HAS_MONGO or predictions_collection is None:
        return False, "Mongo not configured"
    try:
        predictions_collection.insert_one(doc)
        return True, "Saved"
    except Exception as e:
        return False, str(e)

with st.sidebar:
    st.markdown("---")
    if st.button("Save this prediction"):
        doc = {
            "_created_at": datetime.utcnow(),
            "inputs": X.iloc[0].to_dict(),
            "label": int(yhat),
            "probability": float(proba),
        }
        ok, msg = save_prediction(doc)
        if ok:
            st.success("Saved to MongoDB")
        else:
            st.error(f"Could not save: {msg}")

    if HAS_MONGO and st.button("Show last 10 saved"):
        try:
            rows = list(predictions_collection.find().sort("_created_at", -1).limit(10))
            import pandas as _pd
            if rows:
                for r in rows:
                    r.pop("_id", None)
                st.dataframe(_pd.DataFrame(rows))
            else:
                st.info("No saved records yet.")
        except Exception as e:
            st.error(f"Read error: {e}")
