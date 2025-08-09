
import os
from io import BytesIO
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
import streamlit as st
from pymongo import MongoClient
import certifi

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title = "NAFLD Lifestyle Risk Predictor", layout = "wide")
st.title("NAFLD Lifestyle Risk Predictor")
st.caption("Interactive risk assessment with explanations, reporting, and scenario comparison.")

if not os.path.exists('reports'):
    os.makedirs('reports', exist_ok = True)

@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error("Error loading model: " + str(e))
        return None

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

with st.sidebar:
    st.header("Model")
    model_path = st.text_input("Model file path", value = "rf_lifestyle_model (1).pkl")

model = load_model(model_path)

try:
    MODEL_COLS = list(model.feature_names_in_)
except Exception:
    MODEL_COLS = None

TH_LOW = 0.40
TH_HIGH = 0.60

st.subheader("User Data Input")
st.markdown("Enter values for the model's features to get a prediction.")

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"], index = 0)
    age_years = st.number_input("Age in years", 0, 120, 40, 1)
    race = st.selectbox("Race/Ethnicity", [
        "Mexican American","Other Hispanic","Non-Hispanic White",
        "Non-Hispanic Black","Non-Hispanic Asian","Other Race"
    ], index = 0)
    family_income_ratio = st.number_input("Family income ratio", 0.0, 10.0, 2.0, 0.1)
    st.info("Family income ratio: Household income divided by the federal poverty level for your household size. 1.0 means at the poverty threshold. 1.1 means 10 percent above the poverty threshold. 2.0 means 200 percent the poverty threshold. Higher numbers equal higher income relative to poverty level.")
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

# Encoding helpers

def encode_inputs(gender_val, age_val, race_val, income_val, smoker_val, sleep_dis_val, sleep_hours,
                  work_h, activity_mins, bmi_val, alc_days_week, alc_drinks_day, alc_days_year,
                  alc_max_day, alc_freq, kcal, prot, carbs, sugar, fiber, fat):
    races = ["Mexican American","Other Hispanic","Non-Hispanic White","Non-Hispanic Black","Non-Hispanic Asian","Other Race"]
    race_one_hot = {}
    for r in races:
        key = "race_" + r.replace(" ", "_")
        race_one_hot[key] = 1 if race_val == r else 0
    out = {
        "gender_male": 1 if gender_val == "Male" else 0,
        "age_years": age_val,
        "family_income_ratio": float(income_val),
        "smoker_yes": 1 if smoker_val == "Yes" else 0,
        "sleep_disorder_yes": 1 if sleep_dis_val == "Yes" else 0,
        "sleep_duration_hours": float(sleep_hours),
        "work_hours": int(work_h),
        "physical_activity_mins": int(activity_mins),
        "bmi": float(bmi_val),
        "alcohol_days_week": int(alc_days_week),
        "alcohol_drinks_per_day": int(alc_drinks_day),
        "alcohol_days_past_year": int(alc_days_year),
        "alcohol_max_any_day": int(alc_max_day),
        "alcohol_intake_freq": float(alc_freq),
        "total_calories": int(kcal),
        "total_protein": int(prot),
        "total_carbs": int(carbs),
        "total_sugar": int(sugar),
        "total_fiber": int(fiber),
        "total_fat": int(fat)
    }
    out.update(race_one_hot)
    return out

EXPECTED_21 = [
    "gender_male","age_years","family_income_ratio","smoker_yes","sleep_disorder_yes",
    "sleep_duration_hours","work_hours","physical_activity_mins","bmi",
    "alcohol_days_week","alcohol_drinks_per_day","alcohol_days_past_year",
    "alcohol_max_any_day","alcohol_intake_freq",
    "total_calories","total_protein","total_carbs","total_sugar","total_fiber","total_fat",
    "race_Mexican_American"
]
if MODEL_COLS is not None:
    EXPECTED_21 = list(MODEL_COLS)

# Prediction helpers

def predict_one(model_obj, row_dict, cols):
    X = pd.DataFrame([row_dict], columns = cols)
    y_pred = model_obj.predict(X)[0]
    try:
        y_proba = float(model_obj.predict_proba(X)[0][1])
    except Exception:
        y_proba = None
    return int(y_pred), y_proba


def risk_label(p):
    if p < TH_LOW:
        return "Low", "green"
    if p < TH_HIGH:
        return "Borderline", "orange"
    return "High", "red"

# Approximate contributions (simple reference perturbation)

def approx_contributions(model_obj, row_dict, cols, reference = None):
    X_row = pd.DataFrame([row_dict], columns = cols)
    if reference is None:
        ref = {c: 0 for c in cols}
        reference = pd.DataFrame([ref], columns = cols)
    try:
        p_row = float(model_obj.predict_proba(X_row)[0][1])
        p_ref = float(model_obj.predict_proba(reference)[0][1])
    except Exception:
        return []
    contribs = []
    for c in cols:
        ref_copy = reference.copy()
        ref_copy[c] = X_row[c].values[0]
        try:
            p_change = float(model_obj.predict_proba(ref_copy)[0][1])
        except Exception:
            p_change = p_ref
        contribs.append((c, p_change - p_ref))
    contribs_sorted = sorted(contribs, key = lambda x: abs(x[1]), reverse = True)
    return contribs_sorted

# Plot contributions

def plot_contribs(contribs, top_k = 8):
    top = contribs[:top_k]
    if len(top) == 0:
        st.info("No contribution data available for this model.")
        return None
    names = [t[0] for t in top]
    vals = [t[1] for t in top]
    colors = ["#2ca02c" if v < 0 else "#d62728" for v in vals]
    plt.figure(figsize = (8, 4))
    sns.barplot(x = vals, y = names, palette = colors)
    plt.axvline(0, color = "#999999", linewidth = 1)
    plt.xlabel("Estimated contribution to probability")
    plt.ylabel("")
    plt.title("Top factors moving risk up (red) or down (green)")
    plt.tight_layout()
    plt.show()
    return plt.gcf()

# HTML report

def render_html_report(pct, label, inputs_dict, contribs):
    rows = ""
    for k, v in inputs_dict.items():
        rows += "<tr><td>" + str(k) + "</td><td>" + str(v) + "</td></tr>"
    contrib_rows = ""
    for name, val in contribs[:10]:
        arrow = "⬆" if val > 0 else "⬇"
        contrib_rows += "<tr><td>" + name + "</td><td>" + arrow + " " + str(round(val * 100, 1)) + " pp</td></tr>"
    label_class = "low" if label == "Low" else ("mid" if label == "Borderline" else "high")
    html = "" +         "<html><head><meta charset='utf-8'><style>" +         "body { font-family: Arial, sans-serif; padding: 20px; }" +         ".label { font-weight: bold; }" +         ".low { color: #2ca02c; } .mid { color: #ff7f0e; } .high { color: #d62728; }" +         "table { border-collapse: collapse; width: 100%; } th, td { border: 1px solid #ddd; padding: 8px; } th { background: #f2f2f2; }" +         "</style></head><body>" +         "<h2>NAFLD Lifestyle Risk Report</h2>" +         "<p class='label'>Risk category: <span class='" + label_class + "'>" + label + "</span></p>" +         "<p>Estimated probability: " + str(pct) + "%</p>" +         "<h3>Inputs</h3><table><tr><th>Feature</th><th>Value</th></tr>" + rows + "</table>" +         "<h3>Top factors</h3><table><tr><th>Feature</th><th>Effect (percentage points)</th></tr>" + contrib_rows + "</table>" +         "<p style='margin-top: 20px; font-size: 12px; color: #555;'>This report is for educational purposes and is not a diagnosis.</p>" +         "</body></html>"
    return html

# Current inputs to dict

def current_inputs_dict():
    return encode_inputs(
        gender, age_years, race, family_income_ratio, smoking_status, sleep_disorder, sleep_duration_hours,
        work_hours, physical_activity_mins, bmi, alcohol_days_week, alcohol_drinks_per_day,
        alcohol_days_past_year, alcohol_max_any_day, alcohol_intake_freq,
        total_calories, total_protein, total_carbs, total_sugar, total_fiber, total_fat
    )

# Build row respecting order

def build_row_from_full(full_dict, cols):
    row = {}
    for c in cols:
        row[c] = full_dict.get(c, 0)
    return row

# Live what-if sliders for a few key features
st.subheader("What-if Analysis (live)")
wa_col1, wa_col2, wa_col3 = st.columns(3)
with wa_col1:
    wa_bmi = st.slider("What-if BMI", 10.0, 60.0, float(bmi), 0.1)
with wa_col2:
    wa_activity = st.slider("What-if Physical Activity (min/day)", 0, 240, int(physical_activity_mins), 5)
with wa_col3:
    wa_sleep = st.slider("What-if Sleep (hours/day)", 0.0, 12.0, float(sleep_duration_hours), 0.25)

# Compare scenarios side-by-side
st.subheader("Compare Two Scenarios")
comp_left, comp_right = st.columns(2)
with comp_left:
    st.markdown("**Scenario A adjustments**")
    a_bmi = st.slider("A: BMI", 10.0, 60.0, float(bmi), 0.1, key = "a_bmi")
    a_activity = st.slider("A: Activity (min/day)", 0, 240, int(physical_activity_mins), 5, key = "a_act")
    a_sleep = st.slider("A: Sleep (hours)", 0.0, 12.0, float(sleep_duration_hours), 0.25, key = "a_sleep")
with comp_right:
    st.markdown("**Scenario B adjustments**")
    b_bmi = st.slider("B: BMI", 10.0, 60.0, float(bmi), 0.1, key = "b_bmi")
    b_activity = st.slider("B: Activity (min/day)", 0, 240, int(physical_activity_mins), 5, key = "b_act")
    b_sleep = st.slider("B: Sleep (hours)", 0.0, 12.0, float(sleep_duration_hours), 0.25, key = "b_sleep")

# Predict button
left, right = st.columns([1, 1])
with left:
    submitted = st.button("Predict")

if submitted:
    if model is None:
        st.error("Model not loaded.")
    else:
        try:
            full = current_inputs_dict()
            row = build_row_from_full(full, EXPECTED_21)
            y_pred, y_proba = predict_one(model, row, EXPECTED_21)
            p = None
            label = ""
            if y_proba is not None:
                p = max(0.0, min(1.0, y_proba))
                pct = int(round(p * 100))
                label, color = risk_label(p)
                st.subheader("Prediction")
                st.markdown("Risk category: **" + label + "**")
                st.progress(p)
                st.write("Estimated probability: " + str(pct) + "%")
                st.caption("This is the model’s estimated chance of NAFLD given the inputs. It is not a diagnosis.")
            else:
                st.subheader("Prediction")
                st.write("Predicted class: " + ("Positive" if int(y_pred) == 1 else "Negative"))

            with st.expander("Advanced details (explanations, inputs, exports)"):
                contribs = approx_contributions(model, row, EXPECTED_21)
                fig = plot_contribs(contribs, top_k = 8)

                pct_for_report = int(round(p * 100)) if p is not None else (100 if int(y_pred) == 1 else 0)
                label_for_report = label if p is not None else ("Positive" if int(y_pred) == 1 else "Negative")
                html = render_html_report(pct_for_report, label_for_report, row, contribs)
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                html_path = os.path.join('reports', 'report_' + ts + '.html')
                with open(html_path, 'w', encoding = 'utf-8') as f:
                    f.write(html)
                st.success("Saved HTML report to " + html_path)

                # Optional: store to MongoDB
                db = st.session_state.get("mongo_db")
                if db is not None:
                    try:
                        rec = {
                            "timestamp_utc": ts,
                            "inputs": row,
                            "pred_class": int(y_pred),
                            "probability": float(p) if p is not None else None,
                            "label": label_for_report,
                            "html_path": html_path,
                            "contribs": [{"feature": n, "delta": float(v)} for n, v in contribs]
                        }
                        db.predictions.insert_one(rec)
                        st.info("Stored prediction to MongoDB.")
                    except Exception as e:
                        st.warning("MongoDB store failed: " + str(e))

            # Live what-if quick calc display
            if y_proba is not None:
                full_wa = dict(full)
                full_wa['bmi'] = float(wa_bmi)
                full_wa['physical_activity_mins'] = int(wa_activity)
                full_wa['sleep_duration_hours'] = float(wa_sleep)
                row_wa = build_row_from_full(full_wa, EXPECTED_21)
                try:
                    p_wa = float(model.predict_proba(pd.DataFrame([row_wa], columns = EXPECTED_21))[0][1])
                    pct_wa = int(round(max(0.0, min(1.0, p_wa)) * 100))
                    st.markdown("What-if estimate (BMI, activity, sleep): **" + str(pct_wa) + "%**")
                except Exception:
                    st.markdown("What-if estimate unavailable for this model.")

            # Scenario comparison
            if y_proba is not None:
                def scenario_prob(bmi_v, act_v, slp_v):

                def scenario_prob(bmi_v, act_v, slp_v):
                    f = dict(full)
                    f['bmi'] = float(bmi_v)
                    f['physical_activity_mins'] = int(act_v)
                    f['sleep_duration_hours'] = float(slp_v)
                    rw = build_row_from_full(f, EXPECTED_21)
                    try:
                        return float(model.predict_proba(pd.DataFrame([rw], columns = EXPECTED_21))[0][1])
                    except Exception:
                        return None

                p_a = scenario_prob(a_bmi, a_activity, a_sleep)
                p_b = scenario_prob(b_bmi, b_activity, b_sleep)
                if p_a is not None and p_b is not None:
                    st.write("Scenario A: " + str(int(round(p_a * 100))) + "%")
                    st.write("Scenario B: " + str(int(round(p_b * 100))) + "%")
                else:
                    st.write("Scenario comparison unavailable.")
        except Exception as e:
            st.error("Prediction failed: " + str(e))
