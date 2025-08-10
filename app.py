
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title = "NAFLD Lifestyle Risk Predictor", layout = "wide")

@st.cache_resource
def get_mongo_client():
    try:
        from pymongo import MongoClient
        import certifi
        uri = st.secrets["mongo"]["connection_string"]
        client = MongoClient(uri, tlsCAFile = certifi.where(), serverSelectionTimeoutMS = 20000, connectTimeoutMS = 20000)
        client.admin.command("ping")
        return client
    except Exception as e:
        st.warning("Mongo connection not available: " + str(e))
        return None

@st.cache_resource
def load_model_cached(path):
    return joblib.load(path)

@st.cache_resource
def build_explainer_cached(model, background_df):
    try:
        return shap.Explainer(model.predict_proba, background_df)
    except Exception:
        try:
            return shap.Explainer(model)
        except Exception as e:
            st.warning("Could not build SHAP explainer: " + str(e))
            return None

def render_risk_card(prob):
    try:
        p = float(prob)
    except Exception:
        p = 0.0
    if p < 0.34:
        label = 'Low'; color = '#22c55e'
    elif p < 0.67:
        label = 'Medium'; color = '#f59e0b'
    else:
        label = 'High'; color = '#ef4444'
    html = (
        '<div style="padding:12px;border-radius:8px;background:' + color + '1A;border:1px solid ' + color + '">' +
        '<b>Risk:</b> ' + label + ' (' + str(int(round(p*100))) + '%)' +
        '<div style="height:10px;background:#e5e7eb;border-radius:6px;margin-top:8px;">' +
        '<div style="width:' + str(int(round(p*100))) + '%;height:10px;background:' + color + ';border-radius:6px;"></div>' +
        '</div>' +
        '</div>'
    )
    st.markdown(html, unsafe_allow_html = True)

st.title("NAFLD Lifestyle Risk Predictor")
st.caption("Enter lifestyle factors only. No clinical labs.")

with st.sidebar:
    st.header("Connections")
    client = get_mongo_client()
    if client is not None:
        st.success("Connected to MongoDB")
    else:
        st.info("Not connected to MongoDB (proceeding without DB)")

    st.header("Model")
    model_path = st.text_input("Path to model .joblib file", value = "model.joblib")
    load_btn = st.button("Load model")

model = None
if load_btn:
    try:
        model = load_model_cached(model_path)
        st.success("Model loaded: " + str(type(model)))
    except Exception as e:
        st.error("Failed to load model: " + str(e))

# Lifestyle feature names (adjust to your trained model columns)
feature_names = [
    'age',
    'sex',
    'smoking_status',
    'alcohol_units_per_week',
    'physical_activity_minutes_per_week',
    'diet_score',
    'sleep_hours',
    'waist_circumference_cm'
]

col1, col2 = st.columns(2)
with col1:
    age = st.number_input('Age', min_value = 10, max_value = 100, value = 45, step = 1)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    smoking_status = st.selectbox('Smoking status', ['Never', 'Former', 'Current'])
    alcohol_units_per_week = st.number_input('Alcohol units per week', min_value = 0, max_value = 100, value = 4, step = 1)
with col2:
    physical_activity_minutes_per_week = st.number_input('Physical activity minutes per week', min_value = 0, max_value = 10000, value = 150, step = 10)
    diet_score = st.slider('Diet score (0 worst to 10 best)', min_value = 0, max_value = 10, value = 6)
    sleep_hours = st.slider('Sleep hours per night', min_value = 3, max_value = 12, value = 7)
    waist_circumference_cm = st.number_input('Waist circumference (cm)', min_value = 40, max_value = 200, value = 92, step = 1)

# Encode categoricals simply; adjust to your model's preprocessing
sex_map = {'Male': 1, 'Female': 0}
smoke_map = {'Never': 0, 'Former': 1, 'Current': 2}

row_display = {
    'age': int(age),
    'sex': sex,
    'smoking_status': smoking_status,
    'alcohol_units_per_week': int(alcohol_units_per_week),
    'physical_activity_minutes_per_week': int(physical_activity_minutes_per_week),
    'diet_score': int(diet_score),
    'sleep_hours': int(sleep_hours),
    'waist_circumference_cm': int(waist_circumference_cm)
}

row_model = {
    'age': int(age),
    'sex': sex_map.get(sex, 0),
    'smoking_status': smoke_map.get(smoking_status, 0),
    'alcohol_units_per_week': float(alcohol_units_per_week),
    'physical_activity_minutes_per_week': float(physical_activity_minutes_per_week),
    'diet_score': float(diet_score),
    'sleep_hours': float(sleep_hours),
    'waist_circumference_cm': float(waist_circumference_cm)
}

X = pd.DataFrame([row_model], columns = [
    'age', 'sex', 'smoking_status', 'alcohol_units_per_week',
    'physical_activity_minutes_per_week', 'diet_score', 'sleep_hours',
    'waist_circumference_cm'
])

prob = None
if load_btn:
    try:
        if hasattr(model, 'predict_proba'):
            prob = float(model.predict_proba(X)[0][1])
        else:
            if hasattr(model, 'decision_function'):
                val = float(model.decision_function(X)[0])
                prob = 1.0 / (1.0 + np.exp(-val))
            else:
                pred = float(model.predict(X)[0])
                prob = max(0.0, min(1.0, pred))
    except Exception as e:
        st.error('Prediction failed: ' + str(e))

if prob is not None:
    st.subheader('Prediction')
    render_risk_card(prob)
    st.write('Estimated probability: ' + str(int(round(prob*100))) + '%')

if prob is not None and load_btn:
    with st.expander('Why this risk? (SHAP)'):
        try:
            bg = pd.concat([X for _ in range(20)], ignore_index = True)
            for c in bg.columns:
                try:
                    bg[c] = bg[c] + np.random.normal(0, 1e-6, size = len(bg))
                except Exception:
                    pass
            explainer = build_explainer_cached(model, bg)
            if explainer is not None:
                shap_values = explainer(X)
                try:
                    shap.plots.waterfall(shap_values[0], show = False)
                    plt.tight_layout()
                    plt.show()
                    st.caption('Waterfall plot showing how each feature pushes the prediction.')
                except Exception:
                    try:
                        shap.plots.bar(shap_values, max_display = 10, show = False)
                        plt.tight_layout()
                        plt.show()
                    except Exception as e2:
                        st.info('Could not render SHAP plots: ' + str(e2))
            else:
                st.info('No explainer available.')
        except Exception as e:
            st.info('SHAP explanation unavailable: ' + str(e))

if prob is not None:
    try:
        html = '<html><head><meta charset=utf-8><title>NAFLD Report</title></head><body>'
        html += '<h2>NAFLD Lifestyle Risk Report</h2>'
        html += '<p>Generated: ' + datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC') + '</p>'
        html += '<h3>Inputs</h3><pre>' + pd.DataFrame([row_display]).to_csv(index = False) + '</pre>'
        html += '<h3>Predicted Risk</h3><p>' + str(int(round(prob*100))) + '%</p>'
        html += '</body></html>'
        fname = 'nafld_lifestyle_report.html'
        with open(fname, 'w', encoding = 'utf-8') as f:
            f.write(html)
        with open(fname, 'rb') as f:
            st.download_button('Download HTML report', f, file_name = fname, mime = 'text/html')
    except Exception as e:
        st.info('Report not available: ' + str(e))
