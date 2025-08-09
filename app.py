
import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from datetime import datetime
from pymongo import MongoClient
import certifi

st.set_page_config(page_title = "NAFLD Lifestyle Risk Predictor", layout = "wide")
st.title("NAFLD Lifestyle Risk Predictor")
st.caption("Enter values for the model features to get a prediction. Use the sidebar to connect to MongoDB and choose the model file.")

@st.cache_resource
def load_model(model_path):
    try:
        model_obj = joblib.load(model_path)
        return model_obj
    except Exception as e:
        st.error("Error loading model from " + model_path + ": " + str(e))
        return None

# -------------------- MongoDB helpers --------------------

def connect_mongo_ui():
    with st.sidebar:
        st.header("MongoDB")
        mongo_secrets = st.secrets.get("mongo", {})
        cs_default = mongo_secrets.get("connection_string", "")
        dbn_default = mongo_secrets.get("db_name", "nafld_db")
        cs = st.text_input("Connection String", value = cs_default, type = "password", key = "mongo_uri_input")
        dbn = st.text_input("Database Name", value = dbn_default, key = "mongo_db_name_input")
        if st.button("Connect", key = "mongo_connect_btn"):
            try:
                client = MongoClient(cs, tls = True, tlsCAFile = certifi.where())
                client.admin.command("ping")
                st.session_state["mongo_client"] = client
                st.session_state["mongo_db"] = client[dbn]
                st.success("Connected to database: " + dbn)
            except Exception as e:
                st.error("Mongo connection failed: " + str(e))
        if "mongo_db" in st.session_state:
            st.subheader("Quick Test")
            q_coll = st.text_input("Test collection name", value = "predictions", key = "test_coll_name")
            if st.button("Insert test doc", key = "insert_test_doc_btn"):
                try:
                    res = st.session_state["mongo_db"][q_coll].insert_one({
                        "_created_at": datetime.utcnow(),
                        "hello": "world"
                    })
                    st.success("Inserted document with id " + str(res.inserted_id))
                except Exception as e:
                    st.error("Insert failed: " + str(e))

def save_prediction_to_mongo(input_payload, prediction_value, prediction_proba = None, collection_name = "predictions"):
    if "mongo_db" not in st.session_state:
        return
    doc = {
        "_created_at": datetime.utcnow(),
        "inputs": input_payload,
        "prediction": prediction_value,
        "probability": prediction_proba
    }
    try:
        st.session_state["mongo_db"][collection_name].insert_one(doc)
        st.success("Saved prediction to MongoDB in collection " + collection_name)
    except Exception as e:
        st.error("Save failed: " + str(e))

def browse_mongo_ui():
    if "mongo_db" not in st.session_state:
        return
    with st.expander("Browse a MongoDB collection"):
        browse_name = st.text_input("Collection name to view", value = "predictions")
        rows_to_fetch = st.number_input("Rows to fetch", min_value = 5, max_value = 1000, value = 50, step = 5)
        if st.button("Load Collection", key = "browse_load_btn"):
            try:
                docs = list(st.session_state["mongo_db"][browse_name].find().limit(int(rows_to_fetch)))
                for d in docs:
                    if "_id" in d:
                        d["_id"] = str(d["_id"])
                if len(docs) == 0:
                    st.info("No documents found.")
                else:
                    st.dataframe(pd.DataFrame(docs))
            except Exception as e:
                st.error("Browse failed: " + str(e))

# Call Mongo UI builder
connect_mongo_ui()

# -------------------- Sidebar: Model selection --------------------
with st.sidebar:
    st.header("Model")
    default_model = "rf_lifestyle_model (1).pkl"
    model_path = st.text_input("Model file path", value = default_model, key = "model_path_input")
    st.caption("Ensure the model file is in the repo next to app.py.")

model = load_model(model_path)

# Read model expected columns if available
try:
    EXPECTED_COLS = list(model.feature_names_in_)
except Exception:
    EXPECTED_COLS = None

# -------------------- Input UI --------------------
st.subheader("User Data Input")
st.markdown("Enter values for the model's 21 features to get a prediction.")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", options = ["Male", "Female"], index = 0)
    age_years = st.number_input("Age in years", min_value = 0, max_value = 120, value = 40, step = 1)
    race = st.selectbox("Race/Ethnicity", options = [
        "Mexican American",
        "Other Hispanic",
        "Non-Hispanic White",
        "Non-Hispanic Black",
        "Non-Hispanic Asian",
        "Other Race"
    ], index = 0)
    family_income_ratio = st.number_input("Family income ratio", min_value = 0.0, max_value = 10.0, value = 2.0, step = 0.1)
    st.info("Family income ratio: Household income divided by the federal poverty level for your household size. • 1.0 means at the poverty threshold. • 1.1 means 10 percent above the poverty threshold. • 2.0 means 200 percent (2x) the poverty threshold. Higher numbers equal higher income relative to poverty level.")
    smoking_status = st.selectbox("Smoking status", options = ["No", "Yes"], index = 0)

with col2:
    sleep_disorder = st.selectbox("Sleep Disorder Status", options = ["No", "Yes"], index = 0)
    sleep_duration_hours = st.number_input("Sleep duration (hours/day)", min_value = 0.0, max_value = 24.0, value = 8.0, step = 0.25)
    work_hours = st.number_input("Work schedule duration (hours)", min_value = 0, max_value = 24, value = 8, step = 1)
    physical_activity_mins = st.number_input("Physical activity (minutes/day)", min_value = 0, max_value = 1440, value = 30, step = 5)
    bmi = st.number_input("BMI", min_value = 10.0, max_value = 60.0, value = 25.0, step = 0.1)

with col3:
    alcohol_days_week = st.number_input("Alcohol consumption (days/week)", min_value = 0, max_value = 7, value = 0, step = 1)
    alcohol_drinks_per_day = st.number_input("Alcohol drinks per day", min_value = 0, max_value = 50, value = 0, step = 1)
    alcohol_days_past_year = st.number_input("Number of days drank in the past year", min_value = 0, max_value = 366, value = 0, step = 1)
    alcohol_max_any_day = st.number_input("Max number of drinks on any single day", min_value = 0, max_value = 50, value = 0, step = 1)
    alcohol_intake_freq = st.number_input("Alcohol intake frequency (drinks/day)", min_value = 0.0, max_value = 50.0, value = 0.0, step = 0.1)

st.subheader("Nutritional Information")
col4, col5, col6 = st.columns(3)
with col4:
    total_calories = st.number_input("Total calorie intake (kcal)", min_value = 0, max_value = 10000, value = 2000, step = 50)
    total_protein = st.number_input("Total protein intake (grams)", min_value = 0, max_value = 500, value = 60, step = 5)
with col5:
    total_carbs = st.number_input("Total carbohydrate intake (grams)", min_value = 0, max_value = 1000, value = 250, step = 5)
    total_sugar = st.number_input("Total sugar intake (grams)", min_value = 0, max_value = 1000, value = 40, step = 5)
with col6:
    total_fiber = st.number_input("Total fiber intake (grams)", min_value = 0, max_value = 500, value = 30, step = 1)
    total_fat = st.number_input("Total fat intake (grams)", min_value = 0, max_value = 500, value = 70, step = 1)

# Submit
submitted = st.button("Predict")

# Helper to encode inputs to feature dict

def encode_inputs_to_model_dict():
    # One-hot for race
    races = [
        "Mexican American",
        "Other Hispanic",
        "Non-Hispanic White",
        "Non-Hispanic Black",
        "Non-Hispanic Asian",
        "Other Race"
    ]
    race_one_hot = {}
    for r in races:
        key = "race_" + r.replace(" ", "_")
        race_one_hot[key] = 1 if race == r else 0

    gender_male = 1 if gender == "Male" else 0
    smoker_yes = 1 if smoking_status == "Yes" else 0
    sleep_disorder_yes = 1 if sleep_disorder == "Yes" else 0

    features = {
        "gender_male": gender_male,
        "age_years": age_years,
        "family_income_ratio": float(family_income_ratio),
        "smoker_yes": smoker_yes,
        "sleep_disorder_yes": sleep_disorder_yes,
        "sleep_duration_hours": float(sleep_duration_hours),
        "work_hours": int(work_hours),
        "alcohol_days_week": int(alcohol_days_week),
        "alcohol_drinks_per_day": int(alcohol_drinks_per_day),
        "alcohol_days_past_year": int(alcohol_days_past_year),
        "alcohol_max_any_day": int(alcohol_max_any_day),
        "alcohol_intake_freq": float(alcohol_intake_freq),
        "physical_activity_mins": int(physical_activity_mins),
        "total_calories": int(total_calories),
        "total_protein": int(total_protein),
        "total_carbs": int(total_carbs),
        "total_sugar": int(total_sugar),
        "total_fiber": int(total_fiber),
        "total_fat": int(total_fat),
        "bmi": float(bmi)
    }
    features.update(race_one_hot)
    return features

if submitted:
    if model is None:
        st.error("Model not loaded. Check your model file path in the sidebar.")
    else:
        try:
            # Build full dict (may include extras)
            user_inputs_full = encode_inputs_to_model_dict()

            # Strictly match model's expected feature set and order
            if EXPECTED_COLS is not None:
                user_inputs_filtered = {}
                for c in EXPECTED_COLS:
                    user_inputs_filtered[c] = user_inputs_full.get(c, 0)
                X = pd.DataFrame([user_inputs_filtered], columns = EXPECTED_COLS)
            else:
                X = pd.DataFrame([user_inputs_full])

            y_pred = model.predict(X)[0]
            try:
                y_proba = float(model.predict_proba(X)[0][1])
            except Exception:
                y_proba = None

            st.subheader("Prediction Result")
            st.write("Prediction: " + str(y_pred))
            if y_proba is not None:
                st.write("Probability of positive class: " + str(round(y_proba, 4)))

            payload_to_save = user_inputs_filtered if EXPECTED_COLS is not None else user_inputs_full
            save_prediction_to_mongo(payload_to_save, str(y_pred), y_proba, collection_name = "predictions")
        except Exception as e:
            st.error("Prediction failed: " + str(e))

# -------------------- Browse saved docs --------------------
browse_mongo_ui()
