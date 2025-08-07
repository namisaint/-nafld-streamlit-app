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
# This list is from your original app.py file.
feature_names = [
    'RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'INDFMPIR', 'ALQ111', 'ALQ121', 'ALQ142',
    'ALQ151', 'ALQ170', 'Is_Smoker_Cat', 'SLQ050', 'SLQ120', 'SLD012', 'DR1TKCAL',
    'DR1TPROT', 'DR1TCARB', 'DR1TSUGR', 'DR1TFIBE', 'DR1TTFAT', 'PAQ620', 'BMXBMI'
]


# --- Sidebar for User Input ---
st.sidebar.header("User Data Input")
st.sidebar.markdown("Enter values for the model's 21 features to get a prediction.")

# Dictionary to hold the user inputs
user_inputs = {}

# Loop to create an input widget for each feature in the sidebar
for feature in feature_names:
    user_inputs[feature] = st.sidebar.number_input(
        f'Input for {feature}',
        step=0.1,
        value=0.0  # Set a default value
    )

# Combine user inputs into a single DataFrame for prediction.
# The shape and column names MUST match the data the model was trained on.
input_data = pd.DataFrame([user_inputs], columns=feature_names)


# --- Main Content Area: Prediction ---
st.header("Prediction Result")
st.markdown("Click the button below to get a prediction from the model.")

if st.button('Get Prediction'):
    try:
        # Make the prediction
        # The model expects a 2D array, so we pass input_data.
        prediction = model.predict(input_data)
        
        # Get the probability of the positive class.
        probabilities = model.predict_proba(input_data)
        
        # The probability of the positive class (e.g., NAFLD risk)
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
