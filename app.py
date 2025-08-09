
import pandas as pd
import joblib
import streamlit as st
from datetime import datetime
from pymongo import MongoClient
import certifi

st.set_page_config(page_title = "NAFLD Lifestyle Risk Predictor", layout = "wide")
st.title("NAFLD Lifestyle Risk Predictor")
st.caption("Enter values for the model features to get a prediction. Use the sidebar to connect to MongoDB and choose the model file.")

@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error("Error loading model: " + str(e))
        return None

# Sidebar: Mongo
with st.sidebar:
    st.header("MongoDB")
    mongo_secrets = st.secrets.get("mongo", {})
    cs_default = mongo_secrets.get("connection_string", "")
    dbn_default = mongo_secrets.get("db_name", "nafld_db")
    cs = st.text_input("Connection String", value = cs_default, type = "password")
    dbn = st.text_input("Database Name", value = dbn_default)
    if st.button("Connect"):
        try:
            client = MongoClient(cs, tls = True, tlsCAFile = certifi.where())
            client.admin.command("ping")
            st.session_state["mongo_db"] = client[dbn]
            st.success("Connected to " + dbn)
        except Exception as e:
            st.error("Mongo connection failed: " + str(e))

# Sidebar: Model file
with st.sidebar:
    st.header("Model")
    model_path = st.text_input("Model file path", value = "rf_lifestyle_model (1).pkl")

model = load_model(model_path)

# Try to read expected columns from model
try:
    MODEL_COLS = list(model.feature_names_in_)
except Exception:
    MODEL_COLS = None

# UI
st.subheader("User Data Input")
st.markdown("Enter values for the model's 21 features to get a prediction.")

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"], index = 0)
    age_years = st.number_input("Age in years", 0, 120, 40, 1)
    race = st.selectbox("Race/Ethnicity", [
        "Mexican American","Other Hispanic","Non-Hispanic White",
        "Non-Hispanic Black","Non-Hispanic Asian","Other Race"
    ], index = 0)
    family_income_ratio = st.number_input("Family income ratio", 0.0, 10.0, 2.0, 0.1)
    st.info("Family income ratio: Household income divided by the federal poverty level for your household size. • 1.0 means at the poverty threshold. • 1.1 means 10 percent above the poverty threshold. • 2.0 means 200 percent (2x) the poverty threshold. Higher numbers equal higher income relative to poverty level.")
    smoking_status = st.selectbox("Smoking status", ["No", "Yes"], index = 0)
with col2:
    sleep_disorder = st.selectbox("Sleep Disorder Status", ["No", "Yes"], index = 0)
    sleep_duration_hours = st.number_input("Sleep duration (hours/day)", 0.0, 24.0, 8.0, 0.25)
    work_hours = st.number_input("Work schedule duration (hours)", 0, 24, 8, 1)
    physical_activity_mins = st.number_input("Physical activity (minutes/day)", 0, 1440, 30, 5)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0, 0.1)
with col3:
    alcohol_days_week = st.number_input("Alcohol consumption (days/week)", 0, 7, 0, 1)
    alcohol_drinks_per_day = st.number_input("Alcohol drinks per day", 0, 50, 0, 1)
    alcohol_days_past_year = st.number_input("Number of days drank in the past year", 0, 366, 0, 1)
    alcohol_max_any_day = st.number_input("Max number of drinks on any single day", 0, 50, 0, 1)
    alcohol_intake_freq = st.number_input("Alcohol intake frequency (drinks/day)", 0.0, 50.0, 0.0, 0.1)

st.subheader("Nutritional Information")
col4, col5, col6 = st.columns(3)
with col4:
    total_calories = st.number_input("Total calorie intake (kcal)", 0, 10000, 2000, 50)
    total_protein = st.number_input("Total protein intake (grams)", 0, 500, 60, 5)
with col5:
    total_carbs = st.number_input("Total carbohydrate intake (grams)", 0, 1000, 250, 5)
    total_sugar = st.number_input("Total sugar intake (grams)", 0, 1000, 40, 5)
with col6:
    total_fiber = st.number_input("Total fiber intake (grams)", 0, 500, 30, 1)
    total_fat = st.number_input("Total fat intake (grams)", 0, 500, 70, 1)

# Build full encoded dict

def encode_inputs():
    races = ["Mexican American","Other Hispanic","Non-Hispanic White","Non-Hispanic Black","Non-Hispanic Asian","Other Race"]
    race_one_hot = {}
    for r in races:
        key = "race_" + r.replace(" ", "_")
        race_one_hot[key] = 1 if race == r else 0
    out = {
        "gender_male": 1 if gender == "Male" else 0,
        "age_years": age_years,
        "family_income_ratio": float(family_income_ratio),
        "smoker_yes": 1 if smoking_status == "Yes" else 0,
        "sleep_disorder_yes": 1 if sleep_disorder == "Yes" else 0,
        "sleep_duration_hours": float(sleep_duration_hours),
        "work_hours": int(work_hours),
        "physical_activity_mins": int(physical_activity_mins),
        "bmi": float(bmi),
        "alcohol_days_week": int(alcohol_days_week),
        "alcohol_drinks_per_day": int(alcohol_drinks_per_day),
        "alcohol_days_past_year": int(alcohol_days_past_year),
        "alcohol_max_any_day": int(alcohol_max_any_day),
        "alcohol_intake_freq": float(alcohol_intake_freq),
        "total_calories": int(total_calories),
        "total_protein": int(total_protein),
        "total_carbs": int(total_carbs),
        "total_sugar": int(total_sugar),
        "total_fiber": int(total_fiber),
        "total_fat": int(total_fat)
    }
    out.update(race_one_hot)
    return out

# Force exactly 21 columns
# Change the final race_... entry below if your model used a different single race dummy
EXPECTED_21 = [
    "gender_male","age_years","family_income_ratio","smoker_yes","sleep_disorder_yes",
    "sleep_duration_hours","work_hours","physical_activity_mins","bmi",
    "alcohol_days_week","alcohol_drinks_per_day","alcohol_days_past_year",
    "alcohol_max_any_day","alcohol_intake_freq",
    "total_calories","total_protein","total_carbs","total_sugar","total_fiber","total_fat",
    "race_Mexican_American"
]

# If the model exposes feature_names_in_, prefer that
if MODEL_COLS is not None:
    EXPECTED_21 = list(MODEL_COLS)

submitted = st.button("Predict")

# Save to Mongo helper

def save_to_mongo(payload, pred, proba):
    if "mongo_db" not in st.session_state:
        return
    try:
        st.session_state["mongo_db"]["predictions"].insert_one({
            "_created_at": datetime.utcnow(),
            "inputs": payload,
            "prediction": pred,
            "probability": proba
        })
        st.success("Saved to MongoDB")
    except Exception as e:
        st.error("Save failed: " + str(e))

if submitted:
    if model is None:
        st.error("Model not loaded.")
    else:
        try:
            full = encode_inputs()
            row = {}
            for c in EXPECTED_21:
                row[c] = full.get(c, 0)
            X = pd.DataFrame([row], columns = EXPECTED_21)
            y_pred = model.predict(X)[0]
            try:
                y_proba = float(model.predict_proba(X)[0][1])
            except Exception:
                y_proba = None
            st.subheader("Prediction Result")
            st.write("Prediction: " + str(y_pred))
            if y_proba is not None:
                st.write("Probability of positive class: " + str(round(y_proba, 4)))
            save_to_mongo(row, str(y_pred), y_proba)
        except Exception as e:
            st.error("Prediction failed: " + str(e))

# Browse
st.subheader("Browse a MongoDB collection")
coll_name = st.text_input("Collection name", value = "predictions")
if st.button("Load Collection") and "mongo_db" in st.session_state:
    try:
        docs = list(st.session_state["mongo_db"][coll_name].find().limit(50))
        for d in docs:
            if "_id" in d:
                d["_id"] = str(d["_id"])
        if len(docs) == 0:
            st.info("No documents found.")
        else:
            st.dataframe(pd.DataFrame(docs))
    except Exception as e:
        st.error("Browse failed: " + str(e))
