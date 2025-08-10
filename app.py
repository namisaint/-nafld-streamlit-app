import pandas as pd
import joblib
import streamlit as st
from datetime import datetime
from pymongo import MongoClient
import certifi
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO

st.set_page_config(page_title="NAFLD Lifestyle Risk Predictor", layout="wide")
st.title("🤖 NAFLD Lifestyle Risk Predictor")
st.caption("Enter values for the model features to get a prediction. Use the sidebar to connect to MongoDB and choose the model file.")

# Force Matplotlib to use Agg backend to prevent rendering issues in Streamlit
plt.style.use('default')
plt.switch_backend('Agg')

@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error("Error loading model: " + str(e))
        return None

# Sidebar: Mongo
with st.sidebar:
    st.header("🍃 MongoDB Connection")
    mongo_secrets = st.secrets.get("mongo", {})
    cs_default = mongo_secrets.get("connection_string", "")
    dbn_default = mongo_secrets.get("db_name", "nafld_db")
    cs = st.text_input("Connection String", value=cs_default, type="password", key="mongo_uri_input")
    dbn = st.text_input("Database Name", value=dbn_default, key="mongo_db_name_input")

    if st.button("Connect", key="mongo_connect_btn"):
        try:
            client = MongoClient(cs, tls=True, tlsCAFile=certifi.where())
            client.admin.command("ping")
            st.session_state["mongo_db"] = client[dbn]
            st.success("Connected to database: " + dbn)
        except Exception as e:
            st.error("Mongo connection failed: " + str(e))

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

# --- UI
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
        
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        prediction_probability = probabilities[1] * 100

        # Create a visual progress bar and color-coded label
        col_pred, col_report = st.columns([3, 1])
        with col_pred:
            risk_label = "High Risk" if prediction_probability >= 70 else "Moderate Risk" if prediction_probability >= 30 else "Low Risk"
            risk_color = "red" if prediction_probability >= 70 else "orange" if prediction_probability >= 30 else "green"
            st.markdown(f"### Predicted NAFLD Risk: **<span style='color:{risk_color}'>{prediction_probability:.2f}% ({risk_label})</span>**", unsafe_allow_html=True)
            st.progress(prediction_probability / 100)
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
                data=pdf.output(dest="S").encode("latin-1"),
                file_name=f"NAFLD_Risk_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )

        # Create an expandable section for advanced details
        with st.expander("Show Advanced Analysis"):
            # SHAP Analysis
            st.subheader("Model Explainability (SHAP)")
            st.markdown("The chart below shows how each feature contributed to the predicted risk. Red bars increase risk, while blue bars decrease it.")
            
            # Cache the SHAP explainer for performance
            @st.cache_resource
            def get_explainer(model):
                return shap.TreeExplainer(model)
            
            explainer = get_explainer(model)
            shap_values = explainer.shap_values(X)[1]
            st.pyplot(shap.summary_plot(shap_values, X, plot_type="bar", show=False))
            
            st.markdown("---")
            st.subheader("Input Data Summary")
            st.dataframe(pd.DataFrame([full]).T)

            # --- Saved Data Section ---
            st.subheader("Raw Saved Data")
            if st.button('Refresh Saved Predictions'):
                if 'mongo_db' in st.session_state:
                    try:
                        saved_predictions = list(st.session_state["mongo_db"]["predictions"].find())
                        if saved_predictions:
                            df_predictions = pd.DataFrame(saved_predictions)
                            # Remove MongoDB's default _id column for display
                            if '_id' in df_predictions.columns:
                                df_predictions = df_predictions.drop(columns=['_id'])
                            st.dataframe(df_predictions)
                        else:
                            st.info("No saved predictions found.")
                    except Exception as e:
                        st.error(f"Error retrieving predictions: {e}")
                else:
                    st.error("Cannot retrieve predictions. Not connected to MongoDB.")


    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please ensure that all 21 features have valid numerical inputs.")

