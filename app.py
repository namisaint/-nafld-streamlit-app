# app.py — NAFLD Predictor (Pipeline from MongoDB GridFS + lazy load)
import streamlit as st
import pandas as pd
import numpy as np
import joblib, io, certifi, gridfs
from datetime import datetime
from pymongo import MongoClient
from pymongo.server_api import ServerApi

st.set_page_config(page_title="NAFLD Predictor", page_icon="🤖", layout="wide")
st.title("🤖 NAFLD Lifestyle Risk Predictor")

# -----------------------------
# Lazy Mongo client (8s timeout)
# -----------------------------
@st.cache_resource(show_spinner=False)
def _mongo_client(uri: str):
    return MongoClient(
        uri,
        server_api=ServerApi('1'),
        tls=True,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=8000  # fail fast instead of hanging
    )

def load_model_from_gridfs():
    if "mongo" not in st.secrets:
        raise RuntimeError("Missing [mongo] in Streamlit secrets.")

    uri = st.secrets["mongo"]["connection_string"]
    model_db = st.secrets["mongo"].get("model_db_name", "NAFLD_Models")
    bucket   = st.secrets["mongo"].get("model_bucket", "fs")
    fname    = st.secrets["mongo"].get("model_filename", "rf_lifestyle_pipeline.pkl")

    client = _mongo_client(uri)
    client.admin.command("ping")  # triggers timeout if cannot connect

    db = client[model_db]
    fs = gridfs.GridFS(db, collection=bucket)
    files_coll = db[f"{bucket}.files"]

    doc = files_coll.find_one({"filename": fname}, sort=[("uploadDate", -1)])
    if not doc:
        raise FileNotFoundError(
            f"Model '{fname}' not found in GridFS bucket '{bucket}' (db='{model_db}')."
        )

    blob = fs.get(doc["_id"]).read()
    model = joblib.load(io.BytesIO(blob))
    info = {"db": model_db, "bucket": bucket, "filename": fname, "uploadDate": doc.get("uploadDate")}
    return model, info

# -----------------------------------
# Sidebar: connect & load model (lazy)
# -----------------------------------
MODEL_READY = st.session_state.get("model_ready", False)
MODEL_OBJ   = st.session_state.get("model_obj", None)
MODEL_INFO  = st.session_state.get("model_info", None)

with st.sidebar:
    st.subheader("Model")
    if not MODEL_READY:
        if st.button("Connect & load model", type="primary"):
            with st.spinner("Loading model from MongoDB…"):
                try:
                    model, info = load_model_from_gridfs()
                    st.session_state["model_ready"] = True
                    st.session_state["model_obj"] = model
                    st.session_state["model_info"] = info
                    st.success(f"Model loaded: {info['filename']} (db={info['db']}, bucket={info['bucket']})")
                except Exception as e:
                    st.error(f"Model load failed: {e}")
    else:
        st.success("Model loaded ✓")
        st.caption(st.session_state.get("model_info", {}))
        if st.button("Reset model"):
            for k in ["model_ready", "model_obj", "model_info"]:
                st.session_state.pop(k, None)
            st.experimental_rerun()

if not st.session_state.get("model_ready"):
    st.info("Click **Connect & load model** in the left sidebar to load the model from MongoDB.")
    st.stop()

model = st.session_state["model_obj"]

# -----------------------------------
# Optional: Mongo logging for saves
# -----------------------------------
HAS_MONGO_LOGS = False
predictions_collection = None
mongo_status = "disabled"
if "mongo" in st.secrets:
    try:
        uri = st.secrets["mongo"]["connection_string"]
        log_db_name = st.secrets["mongo"].get("db_name", "nafld_app")
        log_coll = st.secrets["mongo"].get("collection_name", "predictions")
        client_logs = _mongo_client(uri)
        client_logs.admin.command("ping")
        _db = client_logs[log_db_name]
        predictions_collection = _db[log_coll]
        HAS_MONGO_LOGS = True
        mongo_status = f"connected (db='{log_db_name}', coll='{log_coll}')"
    except Exception as e:
        mongo_status = f"error: {e}"

with st.sidebar:
    st.subheader("MongoDB logging")
    (st.success if HAS_MONGO_LOGS else st.info)(f"{mongo_status}")

# -----------------------------------
# Inputs (21 human-readable features)
# -----------------------------------
RAW_FEATURES = [
    'Gender','Age in years','Race/Ethnicity','Family income ratio',
    'Alcohol consumption (days/week)','Alcohol drinks per day',
    'Number of days drank in the past year','Max number of drinks on any single day',
    'Alcohol intake frequency (drinks/day)','Smoking status',
    'Sleep duration (hours/day)','Work schedule duration (hours)','Sleep Disorder Status',
    'Total calorie intake (kcal)','Total protein intake (grams)','Total carbohydrate intake (grams)',
    'Total sugar intake (grams)','Total fiber intake (grams)','Total fat intake (grams)',
    'Physical activity (minutes/day)','BMI'
]

st.header("Inputs")

c1, c2, c3 = st.columns(3)
with c1:
    gender = st.selectbox("Gender", ["Male","Female"], index=0)
    age = st.slider("Age in years", 0, 120, 40, 1)
    race = st.selectbox("Race/Ethnicity", [
        "Mexican American","Other Hispanic","Non-Hispanic White",
        "Non-Hispanic Black","Non-Hispanic Asian","Other Race"
    ], index=2)
    income = st.slider("Family income ratio", 0.0, 10.0, 2.0, 0.1)
    smoker = st.selectbox("Smoking status", ["No","Yes"], index=0)

with c2:
    sleep_hours = st.slider("Sleep duration (hours/day)", 0.0, 24.0, 8.0, 0.25)
    work_hours = st.slider("Work schedule duration (hours)", 0, 24, 8, 1)
    sleep_disorder = st.selectbox("Sleep Disorder Status", ["No","Yes"], index=0)
    pa_mins = st.slider("Physical activity (minutes/day)", 0, 1440, 30, 5)
    bmi = st.slider("BMI", 10.0, 60.0, 25.0, 0.1)

with c3:
    alq111 = st.slider("Alcohol consumption (days/week)", 0, 7, 0, 1)
    alq121 = st.slider("Alcohol drinks per day", 0, 50, 0, 1)
    alq142 = st.slider("Number of days drank in the past year", 0, 366, 0, 1)
    alq151 = st.slider("Max number of drinks on any single day", 0, 50, 0, 1)
    alq170 = st.slider("Alcohol intake frequency (drinks/day)", 0.0, 50.0, 0.0, 0.1)

st.subheader("Nutrition")
n1, n2, n3 = st.columns(3)
with n1:
    kcal = st.slider("Total calorie intake (kcal)", 0, 10000, 2000, 50)
    prot = st.slider("Total protein intake (grams)", 0, 500, 60, 5)
with n2:
    carb = st.slider("Total carbohydrate intake (grams)", 0, 1000, 250, 5)
    sug = st.slider("Total sugar intake (grams)", 0, 1000, 40, 5)
with n3:
    fib = st.slider("Total fiber intake (grams)", 0, 500, 30, 1)
    fat = st.slider("Total fat intake (grams)", 0, 500, 70, 1)

row = {
    'Gender': gender,
    'Age in years': age,
    'Race/Ethnicity': race,
    'Family income ratio': income,
    'Alcohol consumption (days/week)': alq111,
    'Alcohol drinks per day': alq121,
    'Number of days drank in the past year': alq142,
    'Max number of drinks on any single day': alq151,
    'Alcohol intake frequency (drinks/day)': alq170,
    'Smoking status': smoker,
    'Sleep duration (hours/day)': sleep_hours,
    'Work schedule duration (hours)': work_hours,
    'Sleep Disorder Status': sleep_disorder,
    'Total calorie intake (kcal)': kcal,
    'Total protein intake (grams)': prot,
    'Total carbohydrate intake (grams)': carb,
    'Total sugar intake (grams)': sug,
    'Total fiber intake (grams)': fib,
    'Total fat intake (grams)': fat,
    'Physical activity (minutes/day)': pa_mins,
    'BMI': bmi
}
X = pd.DataFrame([row], columns=RAW_FEATURES)

# -------------------
# Predict
# -------------------
st.header("Prediction")

def _pos_idx(classes):
    try:
        if 1 in classes: return list(classes).index(1)
        if True in classes: return list(classes).index(True)
        if "1" in classes: return list(classes).index("1")
        nums = [float(c) for c in classes]; return int(np.argmax(nums))
    except Exception:
        return 1 if classes is not None and len(classes) > 1 else 0

try:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        classes = getattr(model, "classes_", [0,1])
        p = float(probs[0][_pos_idx(classes)])
        yhat = int(model.predict(X)[0]) if hasattr(model,"predict") else int(p >= 0.5)
    else:
        yhat = int(model.predict(X)[0]); p = float(yhat)
except Exception as e:
    st.error(f"❌ Prediction error: {e}")
    st.write("Inputs sent:", X.T.rename(columns={0:'value'}))
    st.stop()

risk_pct = p * 100.0
label = "High Risk" if risk_pct >= 70 else "Moderate Risk" if risk_pct >= 30 else "Low Risk"

st.markdown(f"### Predicted NAFLD Risk: **{risk_pct:.2f}% ({label})**")
st.progress(min(max(risk_pct/100.0, 0.0), 1.0))
st.write("**Inputs sent to the model:**")
st.dataframe(X.T.rename(columns={0:'value'}))

# -------------------
# Optional: Save to Mongo logs
# -------------------
def save_prediction(doc):
    if not HAS_MONGO_LOGS or predictions_collection is None:
        return False, "Mongo logging not configured"
    try:
        predictions_collection.insert_one(doc); return True, "Saved"
    except Exception as e:
        return False, str(e)

with st.sidebar:
    st.markdown("---")
    if st.button("Save this prediction"):
        ok, msg = save_prediction({
            "_created_at": datetime.utcnow(),
            "inputs": X.iloc[0].to_dict(),
            "label": int(yhat),
            "probability": float(p)
        })
        (st.success if ok else st.error)(msg)
    if HAS_MONGO_LOGS and st.button("Show last 10 saved"):
        try:
            rows = list(predictions_collection.find().sort("_created_at",-1).limit(10))
            for r in rows: r.pop("_id", None)
            st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame([]))
        except Exception as e:
            st.error(f"Read error: {e}")
