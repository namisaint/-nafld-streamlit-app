import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load your trained model (update with your exact filename if needed)
model = joblib.load("rf_lifestyle_model.pkl")

# List of 21 input features expected by the model
feature_names = [
    'RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'INDFMPIR', 'ALQ111', 'ALQ121', 'ALQ142',
    'ALQ151', 'ALQ170', 'Is_Smoker_Cat', 'SLQ050', 'SLQ120', 'SLD012', 'DR1TKCAL',
    'DR1TPROT', 'DR1TCARB', 'DR1TSUGR', 'DR1TFIBE', 'DR1TTFAT', 'PAQ620', 'BMXBMI'
]

# Page title
st.title("🧠 NAFLD Lifestyle Risk Predictor")

# Collect user input
st.markdown("### 🔍 Enter your information:")

user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", step=0.1)
    user_input.append(value)

# Predict button
if st.button("Predict NAFLD Risk"):
    input_data = np.array(user_input).reshape(1, -1)
    prediction = model.predict_proba(input_data)[0][1]  # Probability of positive class
    st.success(f"Predicted NAFLD Risk: {prediction * 100:.2f}%")
