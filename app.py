import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- App Configuration ---
st.set_page_config(
    page_title="Dissertation Model Predictor",
    page_icon="�",
    layout="wide"
)

# --- Main App Title and Introduction ---
st.title("🤖 Dissertation Lifestyle Model Predictor")
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

# --- Sidebar for User Input ---
st.sidebar.header("User Data Input (21 Features)")
st.sidebar.markdown("Enter values for the model's 21 features to get a prediction.")

# List of all 21 feature names. You MUST replace these with your actual feature names.
# This makes it easy to add or remove features in one place.
feature_names = [f"Feature_{i}" for i in range(1, 22)]

# Dictionary to hold the user inputs
user_inputs = {}

# Loop to create a slider for each feature in the sidebar
for feature in feature_names:
    user_inputs[feature] = st.sidebar.slider(
        f'Input for {feature}',
        0.0, 100.0, 50.0  # Min, max, and default values. Customize as needed.
    )

# Combine user inputs into a single DataFrame for prediction.
# The shape and column names must match the data the model was trained on.
input_data = pd.DataFrame([user_inputs], columns=feature_names)


# --- Main Content Area: Prediction ---
st.header("Prediction Result")
st.markdown("Click the button below to get a prediction from the model.")

if st.button('Get Prediction'):
    try:
        # Make the prediction
        prediction = model.predict(input_data)
        
        # Get the probability of each class
        probabilities = model.predict_proba(input_data)
        
        # Based on the model type (e.g., RandomForestClassifier), interpret the result.
        # The classes_ attribute holds the names of the classes.
        prediction_label = model.classes_[prediction[0]]
        confidence = probabilities[0][prediction[0]] * 100
        
        st.success(f"### Predicted Lifestyle: **{prediction_label}** 🌟")
        st.info(f"The model is **{confidence:.2f}%** confident in this prediction.")
        
        st.markdown("---")
        st.subheader("Input Data Summary")
        st.dataframe(input_data)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.divider()
st.info("Remember to install the necessary library by running `pip install joblib`. You'll also need to have scikit-learn installed, which you likely do already. Once you have saved this code, run it from your terminal using `streamlit run <filename>.py` to see the app in action.")

�
