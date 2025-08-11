import streamlit as st
import certifi, io, gridfs, joblib, pandas as pd
from pymongo import MongoClient
from pymongo.server_api import ServerApi

st.set_page_config(page_title="NAFLD Diag", page_icon="🛠️", layout="centered")
st.title("🛠️ NAFLD Deployment Diagnostics")

def ok(msg): st.success("✅ " + msg)
def info(msg): st.info("ℹ️ " + msg)
def warn(msg): st.warning("⚠️ " + msg)
def err(msg): st.error("❌ " + msg)

with st.status("Starting diagnostics…", expanded=True) as status:
    # 1) Secrets
    try:
        if "mongo" not in st.secrets:
            raise RuntimeError("Missing [mongo] in Streamlit secrets.")
        uri = st.secrets["mongo"]["connection_string"]
        model_db = st.secrets["mongo"].get("model_db_name", "NAFLD_Models")
        bucket   = st.secrets["mongo"].get("model_bucket", "fs")
        fname    = st.secrets["mongo"].get("model_filename", "rf_lifestyle_pipeline.pkl")
        log_db   = st.secrets["mongo"].get("db_name", "nafld_app")
        log_coll = st.secrets["mongo"].get("collection_name", "predictions")
        ok("Loaded secrets.toml")
        st.write({"model_db": model_db, "bucket": bucket, "filename": fname, "log_db": log_db, "log_coll": log_coll})
    except Exception as e:
        err(f"Secrets error: {e}")
        st.stop()

    # 2) Connect to Mongo
    try:
        client = MongoClient(uri, server_api=ServerApi('1'), tls=True, tlsCAFile=certifi.where())
        client.admin.command("ping")
        ok("MongoDB ping OK")
        st.write("Databases (subset):", client.list_database_names()[:10])
    except Exception as e:
        err(f"Mongo connection/auth failed: {e}")
        st.stop()

    # 3) GridFS lookup
    try:
        db = client[model_db]
        fs = gridfs.GridFS(db, collection=bucket)
        files_coll = db[f"{bucket}.files"]
        doc = files_coll.find_one({"filename": fname}, sort=[("uploadDate", -1)])
        if not doc:
            raise FileNotFoundError(f"GridFS doc not found for filename='{fname}' in bucket '{bucket}' db '{model_db}'")
        ok("Found model file in GridFS")
        st.write({"_id": str(doc["_id"]), "length": doc.get("length"), "uploadDate": doc.get("uploadDate")})
    except Exception as e:
        err(f"GridFS lookup error: {e}")
        st.stop()

    # 4) Download & deserialize
    try:
        blob = fs.get(doc["_id"]).read()
        model = joblib.load(io.BytesIO(blob))
        ok("Model loaded from GridFS and deserialized")
    except Exception as e:
        err(f"Model load error: {e}")
        st.stop()

    # 5) One simple predict to prove it works
    try:
        sample = pd.DataFrame([{
            'Gender': 'Male',
            'Age in years': 40,
            'Race/Ethnicity': 'Non-Hispanic White',
            'Family income ratio': 2.0,
            'Alcohol consumption (days/week)': 1,
            'Alcohol drinks per day': 1,
            'Number of days drank in the past year': 10,
            'Max number of drinks on any single day': 3,
            'Alcohol intake frequency (drinks/day)': 0.2,
            'Smoking status': 'No',
            'Sleep duration (hours/day)': 7.5,
            'Work schedule duration (hours)': 8,
            'Sleep Disorder Status': 'No',
            'Total calorie intake (kcal)': 2200,
            'Total protein intake (grams)': 70,
            'Total carbohydrate intake (grams)': 250,
            'Total sugar intake (grams)': 40,
            'Total fiber intake (grams)': 30,
            'Total fat intake (grams)': 70,
            'Physical activity (minutes/day)': 30,
            'BMI': 25.0
        }])
        if hasattr(model, "predict_proba"):
            p = float(model.predict_proba(sample)[0][1])
            ok(f"Predict OK (prob class 1): {p:.3f}")
        else:
            y = model.predict(sample)[0]
            ok(f"Predict OK (label): {y}")
        status.update(label="Diagnostics finished", state="complete")
    except Exception as e:
        err(f"Predict test failed: {e}")
        st.stop()

st.write("If everything above is green, your model is loading fine. You can now revert to the full app.py.")
