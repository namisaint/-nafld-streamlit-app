import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import plotly.express as px
import certifi
import shap
from fpdf import FPDF
from io import BytesIO

# --- App Configuration ---
st.set_page_config(
    page_title="Dissertation Model Predictor",
    page_icon="🤖",
    layout="wide"
)

# --- MongoDB Connection ---
# The connection string is now read securely from st.secrets.
try:
    MONGODB_CONNECTION_STRING = st.secrets["mongo"]["connection_string"]
    DB_NAME = st.secrets["mongo"]["db_name"]
    COLLECTION_NAME = st.secrets["mongo"]["collection_name"]
except KeyError:
    st.error("MongoDB secrets are not configured. Please add your credentials to the Streamlit secrets.")
    st.stop()

@st.cache_resource
def get_mongo_client():
    """
    Connects to the MongoDB Atlas cluster.
    """
    try:
        client = MongoClient(MONGODB_CONNECTION_STRING, server_api=ServerApi('1'), tls=True, tlsCAFile=certifi.where())
        client.admin.command('ping')
        return client
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        return None

mongo_client = get_mongo_client()
if mongo_client:
    db = mongo_client[DB_NAME]
    predictions_collection = db[COLLECTION_NAME]
else:
    st.error("Could not connect to the database. The app will not be able to save or load predictions.")
    predictions_collection = None


# --- Main App Title and Introduction ---
st.title("🤖 NAFLD Lifestyle Risk Predictor")
st.markdown("""
This application allows you to interact with the machine learning model developed for my dissertation.
Use the sidebar to input new data and get a prediction from the model.
""")
st.divider()

# --- Model Loading and Caching ---
@st.cache_resource
def load_model(file_path):
    """
    Loads the machine learning model from a pickled file.
    """
    if not os.path.exists(file_path):
        st.error(f"Error: Model file not found at '{file_path}'")
        return None
    
    try:
        model = joblib.load(file_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_model('rf_lifestyle_model (1).pkl')
if model is None:
    st.stop()

feature_names = [
    'RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'INDFMPIR', 'ALQ111', 'ALQ121', 'ALQ142',
    'ALQ151', 'ALQ170', 'Is_Smoker_Cat', 'SLQ050', 'SLQ120', 'SLD012', 'DR1TKCAL',
    'DR1TPROT', 'DR1TCARB', 'DR1TSUGR', 'DR1TFIBE', 'DR1TTFAT', 'PAQ620', 'BMXBMI'
]

# Dictionary to map technical feature names to human-readable labels
feature_labels = {
    'RIAGENDR': 'Gender', 'RIDAGEYR': 'Age in years', 'RIDRETH3': 'Race/Ethnicity',
    'INDFMPIR': 'Family income ratio', 'ALQ111': 'Alcohol consumption (days/week)',
    'ALQ121': 'Alcohol drinks/day', 'ALQ142': 'Number of days drank in the past year',
    'ALQ151': 'Number of drinks per day', 'ALQ170': 'Alcohol intake frequency',
    'Is_Smoker_Cat': 'Smoking status', 'SLQ050': 'Sleep duration (hours/day)',
    'SLQ120': 'Work schedule duration (hours)', 'SLD012': 'Sleep disorder status',
    'DR1TKCAL': 'Total calorie intake (kcal)', 'DR1TPROT': 'Total protein intake (g)',
    'DR1TCARB': 'Total carbohydrate intake (g)', 'DR1TSUGR': 'Total sugar intake (g)',
    'DR1TFIBE': 'Total fiber intake (g)', 'DR1TTFAT': 'Total fat intake (g)',
    'PAQ620': 'Physical activity (minutes/day)', 'BMXBMI': 'BMI'
}

gender_options = {'Male': 1, 'Female': 2}
smoking_options = {'No': 0, 'Yes': 1}
race_options = {
    'Mexican American': 1, 'Other Hispanic': 2, 'Non-Hispanic White': 3,
    'Non-Hispanic Black': 4, 'Other Race - Including Multi-Racial': 6
}
sleep_disorder_options = {'No': 0, 'Yes': 1}

user_inputs = {}
st.sidebar.header("User Data Input")
st.sidebar.subheader("Demographic & Lifestyle")
user_inputs['RIAGENDR'] = st.sidebar.selectbox('Gender', options=list(gender_options.keys()))
user_inputs['RIDAGEYR'] = st.sidebar.slider('Age in years', 18, 100, 30)
user_inputs['RIDRETH3'] = st.sidebar.selectbox('Race/Ethnicity', options=list(race_options.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown("**Family Income Details**")
st.sidebar.markdown("The app calculates your family's income relative to the poverty line. A value of **1.0** means you are at the poverty line.")
user_inputs['household_income'] = st.sidebar.number_input('Annual household income (£)', min_value=0, step=1000, value=30000, help="Enter your total household income.")
user_inputs['family_size'] = st.sidebar.number_input('Number of people in household', min_value=1, step=1, value=1, help="Enter the number of people in your household.")
POVERTY_LINE_PER_PERSON = 15000  
poverty_line_for_family = POVERTY_LINE_PER_PERSON * user_inputs['family_size']
if user_inputs['family_size'] > 0:
    user_inputs['INDFMPIR'] = user_inputs['household_income'] / poverty_line_for_family
else:
    user_inputs['INDFMPIR'] = 0.0
st.sidebar.caption(f"Your calculated income ratio is: **{user_inputs['INDFMPIR']:.2f}**")

user_inputs['Is_Smoker_Cat'] = st.sidebar.radio('Smoking status', options=list(smoking_options.keys()))
user_inputs['SLD012'] = st.sidebar.selectbox('Sleep Disorder Status', options=list(sleep_disorder_options.keys()))
user_inputs['SLQ050'] = st.sidebar.number_input('Sleep duration (hours/day)', step=0.5, value=8.0)
user_inputs['SLQ120'] = st.sidebar.number_input('Work schedule duration (hours)', step=1, value=8)

st.sidebar.divider()
st.sidebar.subheader("Alcohol & Physical Activity")
user_inputs['ALQ111'] = st.sidebar.number_input('Alcohol consumption (days/week)', min_value=0, max_value=7, step=1, value=0)
user_inputs['ALQ121'] = st.sidebar.number_input('Alcohol drinks per day', min_value=0, max_value=50, step=1, value=0,
    help="A standard drink is 14g of pure alcohol (e.g., 12oz beer, 5oz wine, 1.5oz spirits).")
user_inputs['ALQ142'] = st.sidebar.number_input('Number of days drank in the past year', min_value=0, max_value=365, step=1, value=0)
user_inputs['ALQ151'] = st.sidebar.number_input('Max number of drinks on any single day', min_value=0, max_value=50, step=1, value=0)
user_inputs['ALQ170'] = st.sidebar.number_input('Alcohol intake frequency (drinks/day)', min_value=0, step=1, value=0)
user_inputs['PAQ620'] = st.sidebar.number_input('Physical activity (minutes/day)', min_value=0, step=15, value=30)

st.sidebar.divider()
st.sidebar.subheader("Nutritional Information")
user_inputs['DR1TKCAL'] = st.sidebar.number_input('Total calorie intake (kcal)', min_value=0, step=100, value=2000,
    help="Estimate your daily total calories.")
user_inputs['DR1TPROT'] = st.sidebar.number_input('Total protein intake (grams)', min_value=0, step=1, value=60)
user_inputs['DR1TCARB'] = st.sidebar.number_input('Total carbohydrate intake (grams)', min_value=0, step=1, value=250)
user_inputs['DR1TSUGR'] = st.sidebar.number_input('Total sugar intake (grams)', min_value=0, step=1, value=40)
user_inputs['DR1TFIBE'] = st.sidebar.number_input('Total fiber intake (grams)', min_value=0, step=1, value=30)
user_inputs['DR1TTFAT'] = st.sidebar.number_input('Total fat intake (grams)', min_value=0, step=1, value=70)
user_inputs['BMXBMI'] = st.sidebar.number_input('BMI', step=0.1, value=25.0, help="Body Mass Index")

final_inputs = {
    'RIAGENDR': gender_options[user_inputs['RIAGENDR']],
    'RIDRETH3': race_options[user_inputs['RIDRETH3']],
    'Is_Smoker_Cat': smoking_options[user_inputs['Is_Smoker_Cat']],
    'SLD012': sleep_disorder_options[user_inputs['SLD012']],
    'INDFMPIR': user_inputs['INDFMPIR'],
}
for feature in feature_names:
    if feature not in final_inputs:
        final_inputs[feature] = user_inputs[feature]

input_data = pd.DataFrame([final_inputs], columns=feature_names)

# --- Prediction Logic and Display ---
st.header("Prediction Result")
st.markdown("Adjust the inputs in the sidebar to see the prediction update in real-time.")

if model is not None:
    try:
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        prediction_probability = probabilities[0][1] * 100

        # Create a visual progress bar and color-coded label
        col_pred, col_report = st.columns([3, 1])
        with col_pred:
            risk_label = "High Risk" if prediction_probability >= 70 else "Moderate Risk" if prediction_probability >= 30 else "Low Risk"
            risk_color = "red" if prediction_probability >= 70 else "orange" if prediction_probability >= 30 else "green"
            st.markdown(f"### Predicted NAFLD Risk: **<span style='color:{risk_color}'>{prediction_probability:.2f}% ({risk_label})</span>**", unsafe_allow_html=True)
            st.progress(prediction_probability / 100)
            st.markdown("The prediction is based on the features entered.")
        with col_report:
            pdf = create_pdf(final_inputs, prediction_probability, risk_label)
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
            shap_values = explainer.shap_values(input_data)

            # Create a SHAP bar chart
            st.pyplot(shap.summary_plot(shap_values[1], input_data, plot_type="bar", show=False))
            
            st.markdown("---")
            st.subheader("Input Data Summary")
            st.dataframe(pd.DataFrame([final_inputs], columns=feature_names).T)
            
            st.subheader("Raw Saved Data")
            if st.button('Refresh Saved Predictions'):
                if predictions_collection is not None:
                    try:
                        saved_predictions = list(predictions_collection.find())
                        if saved_predictions:
                            df_predictions = pd.DataFrame(saved_predictions)
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

# --- Helper function for PDF report generation ---
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

