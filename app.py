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
st.sidebar.markdown("Use the input boxes or the `+` and `-` buttons to change the values below.")


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

user_inputs = {}

st.sidebar.subheader("Demographic & Lifestyle")

# Use selectbox and radio for features with a clear list of options
user_inputs['RIAGENDR'] = st.sidebar.selectbox('Gender', options=list(gender_options.keys()))
user_inputs['RIDRETH3'] = st.sidebar.selectbox('Race/Ethnicity', options=list(race_options.keys()))
user_inputs['Is_Smoker_Cat'] = st.sidebar.radio('Smoking status', options=list(smoking_options.keys()))
user_inputs['SLD012'] = st.sidebar.selectbox('Sleep Disorder Status', options=list(sleep_disorder_options.keys()))

st.sidebar.divider()
st.sidebar.subheader("Health Metrics")

# Use number_input for the remaining numerical features
for feature in feature_names:
    if feature not in ['RIAGENDR', 'RIDRETH3', 'Is_Smoker_Cat', 'SLD012']:
        label = feature_labels.get(feature, feature)
        
        # Add a tooltip for the family income ratio to make it more intuitive
        if feature == 'INDFMPIR':
            help_text = "A value of 1.0 represents the poverty line. A value of 2.0 is twice the poverty line, and so on."
            user_inputs[feature] = st.sidebar.number_input(
                f'Input for {label}',
                step=0.1,
                value=0.0,
                help=help_text
            )
        else:
            user_inputs[feature] = st.sidebar.number_input(
                f'Input for {label}',
                step=0.1,
                value=0.0
            )

# Map the selected string options back to the numerical values the model expects
final_inputs = {
    'RIAGENDR': gender_options[user_inputs['RIAGENDR']],
    'RIDRETH3': race_options[user_inputs['RIDRETH3']],
    'Is_Smoker_Cat': smoking_options[user_inputs['Is_Smoker_Cat']],
    'SLD012': sleep_disorder_options[user_inputs['SLD012']],
}

# Add the rest of the numerical inputs to the final dictionary
for feature in feature_names:
    if feature not in ['RIAGENDR', 'RIDRETH3', 'Is_Smoker_Cat', 'SLD012']:
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
        
        st.success(f"### Predicted NAFLD Risk: {prediction_probability:.2f}%")
        st.info("The prediction is based on the features entered. The higher the percentage, the higher the predicted risk.")
        
        st.markdown("---")
        st.subheader("Input Data Summary")
        st.dataframe(input_data)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please ensure that all 21 features have valid numerical inputs.")

st.divider()
st.info("Remember to save this file as `app.py` and commit it to your GitHub repository to update your live app.")
