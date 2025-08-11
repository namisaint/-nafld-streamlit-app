# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# -----------------------
# Load trained model
# -----------------------
@st.cache_resource
def load_model():
    return joblib.load("rf_lifestyle_model.pkl")

model = load_model()

# -----------------------
# MongoDB connection (from secrets)
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
            client = MongoClient(
                conn,
                server_api=ServerApi('1'),
                tls=True,
                tlsCAFile=certifi.where()
            )
            client.admin.command("ping")
            return client

        client = get_mongo_client()
        db = client[st.secrets["mongo"]["db_name"]]
        predictions_collection = db[st.secrets["mongo"]["collection_name"]]
        HAS_MONGO = True
        mongo_status = "connected"
    except Exception as e:
        mongo_status = f"error: {e}"

with st.sidebar:
    st.subheader("MongoDB")
    if HAS_MONGO:
        st.success(f"MongoDB {mongo_status}")
    else:
        st.info(f"MongoDB {mongo_status}")

# -----------------------
# App title
# -----------------------
st.title("NAFLD Lifestyle Model Predictor")

# -----------------------
# Collect inputs
# -----------------------
st.header("Enter Patient Information")

Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.slider("Age in years", 18, 80, 40)
BMI = st.slider("BMI", 15.0, 45.0, 25.0)
PhysicalActivity = st.slider("Physical activity (minutes/day)", 0, 300, 30)
AlcoholPerWeek = st.slider("Alcohol consumption (days/week)", 0, 7, 2)

# -----------------------
# Prepare input for model
# -----------------------
# Make sure these match your model's expected feature order
input_df = pd.DataFrame([{
    "Gender": 1 if Gender == "Male" else 0,
    "Age in years": Age,
    "BMI": BMI,
    "Physical activity (minutes/day)": PhysicalActivity,
    "Alcohol consumption (days/week)": AlcoholPerWeek
}])

# -----------------------
# Predict
# -----------------------
y_pred = model.predict(input_df)[0]
y_prob = model.predict_proba(input_df)[0][1]

st.subheader("Prediction")
st.write(f"Predicted class: **{y_pred}**")
st.write(f"Probability of NAFLD: **{y_prob:.2%}**")

# -----------------------
# Save to MongoDB
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
            "inputs": input_df.iloc[0].to_dict(),
            "label": int(y_pred),
            "probability": float(y_prob),
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
