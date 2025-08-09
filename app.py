import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- App Configuration ---
st.set_page_config(
    page_title="Dissertation Model Predictor",
    page_icon="🤖",
    layout="wide"
)

# --- Main App Title and Introduction ---
st.title("🤖 NAFLD Lifestyle Risk Predictor")
st.markdown("""
This application allows you to interact with the machine learning model developed for my dissertation.
Use the sidebar to input new data and get a prediction from the model.
""")
st.divider()

# --- Model Loading and Caching ---
# @st.cache_resource is perfect for loading models, as it ensures the model is loaded only once.
@st.cache_resource
def load_model(file_path):
    """
    Loads the machine learning model from a pickled file.
    
    Args:
        file_path (str): The path to the .pkl file.
    
    Returns:
        The loaded machine learning model.
    """
    # Check if the file exists before trying to load it
    if not os.path.exists(file_path):
        st.error(f"Error: Model file not found at '{file_path}'")
        return None
    
    try:
        model = joblib.load(file_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# The model file is assumed to be in the same directory as your app.py script.
# This line has been updated to match the filename you uploaded to GitHub.
model = load_model('rf_lifestyle_model (1).pkl')

# If the model failed to load, display an error and stop the app.
if model is None:
    st.stop()

# List of the 21 input features expected by the model.
feature_names = [
    'RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'INDFMPIR', 'ALQ111', 'ALQ121', 'ALQ142',
    'ALQ151', 'ALQ170', 'Is_Smoker_Cat', 'SLQ050', 'SLQ120', 'SLD012', 'DR1TKCAL',
    'DR1TPROT', 'DR1TCARB', 'DR1TSUGR', 'DR1TFIBE', 'DR1TTFAT', 'PAQ620', 'BMXBMI'
]

# Dictionary to map technical feature names to human-readable labels
# This makes the user interface much clearer.
feature_labels = {
    'RIAGENDR': 'Gender',
    'RIDAGEYR': 'Age in years',
    'RIDRETH3': 'Race/Ethnicity',
    'INDFMPIR': 'Family income ratio',
    'ALQ111': 'Alcohol consumption (days/week)',
    'ALQ121': 'Alcohol drinks/day',
    'ALQ142': 'Number of days drank in the past year',
    'ALQ151': 'Number of drinks per day',
    'ALQ170': 'Alcohol intake frequency',
    'Is_Smoker_Cat': 'Smoking status',
    'SLQ050': 'Sleep duration (hours/day)',
    'SLQ120': 'Work schedule duration (hours)',
    'SLD012': 'Sleep disorder status',
    'DR1TKCAL': 'Total calorie intake (kcal)',
    'DR1TPROT': 'Total protein intake (g)',
    'DR1TCARB': 'Total carbohydrate intake (g)',
    'DR1TSUGR': 'Total sugar intake (g)',
    'DR1TFIBE': 'Total fiber intake (g)',
    'DR1TTFAT': 'Total fat intake (g)',
    'PAQ620': 'Physical activity (minutes/day)',
    'BMXBMI': 'BMI'
}

# --- Sidebar for User Input ---
st.sidebar.header("User Data Input")
st.sidebar.markdown("Enter values for the model's 21 features to get a prediction.")


# Dictionary to map user-friendly labels to numerical values for the model
gender_options = {'Male': 1, 'Female': 2}
smoking_options = {'No': 0, 'Yes': 1}
race_options = {
    'Mexican American': 1,
    'Other Hispanic': 2,
    'Non-Hispanic White': 3,
    'Non-Hispanic Black': 4,
    'Other Race - Including Multi-Racial': 6
}
sleep_disorder_options = {'No': 0, 'Yes': 1}
income_options = {
    'Less than the poverty line': 0.5,
    'At the poverty line': 1.0,
    '1.5 times the poverty line': 1.5,
    '2 times the poverty line': 2.0,
    '3 times the poverty line': 3.0,
    'More than 3 times the poverty line': 4.0
}


user_inputs = {}

st.sidebar.subheader("Demographic & Lifestyle")

# Use selectbox and radio for features with a clear list of options
user_inputs['RIAGENDR'] = st.sidebar.selectbox('Gender', options=list(gender_options.keys()))
user_inputs['RIDAGEYR'] = st.sidebar.slider('Age in years', 18, 100, 30)
user_inputs['RIDRETH3'] = st.sidebar.selectbox('Race/Ethnicity', options=list(race_options.keys()))
user_inputs['INDFMPIR'] = st.sidebar.selectbox(
    'Family income ratio',
    options=list(income_options.keys()),
    help="Select the option that best describes your household's income relative to the poverty line."
)
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

# Map the selected string options back to the numerical values the model expects
final_inputs = {
    'RIAGENDR': gender_options[user_inputs['RIAGENDR']],
    'RIDRETH3': race_options[user_inputs['RIDRETH3']],
    'Is_Smoker_Cat': smoking_options[user_inputs['Is_Smoker_Cat']],
    'SLD012': sleep_disorder_options[user_inputs['SLD012']],
    'INDFMPIR': income_options[user_inputs['INDFMPIR']],
}

# Add the rest of the numerical inputs to the final dictionary
for feature in feature_names:
    if feature not in ['RIAGENDR', 'RIDRETH3', 'Is_Smoker_Cat', 'SLD012', 'INDFMPIR']:
        final_inputs[feature] = user_inputs[feature]

# Combine final user inputs into a single DataFrame for prediction, ensuring correct order
input_data = pd.DataFrame([final_inputs], columns=feature_names)

# --- Main Content Area: Prediction ---
st.header("Prediction Result")
st.markdown("Click the button below to get a prediction from the model.")

if st.button('Get Prediction'):
    try:
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        prediction_probability = probabilities[0][1] * 100
        
        # New conditional logic to provide context for the prediction percentage
        if prediction_probability >= 70:
            st.error(f"### Predicted NAFLD Risk: {prediction_probability:.2f}% ⚠️")
            st.markdown("The model predicts a **high risk** of NAFLD based on the features you've entered. A higher percentage suggests a higher likelihood.")
        elif prediction_probability >= 30:
            st.warning(f"### Predicted NAFLD Risk: {prediction_probability:.2f}%")
            st.markdown("The model predicts a **moderate risk** of NAFLD based on the features you've entered. A higher percentage suggests a higher likelihood.")
        else:
            st.success(f"### Predicted NAFLD Risk: {prediction_probability:.2f}% ✅")
            st.markdown("The model predicts a **low risk** of NAFLD based on the features you've entered.")

        st.markdown("---")
        st.subheader("Input Data Summary")
        st.dataframe(input_data)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please ensure that all 21 features have valid numerical inputs.")

st.divider()
st.info("Remember to save this file as `app.py` and commit it to your GitHub repository to update your live app.")
