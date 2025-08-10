import os
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title = "NAFLD Lifestyle Risk Predictor", layout = "wide")
st.title("NAFLD Lifestyle Risk Predictor")
st.caption("Interactive risk assessment with explanations and reporting.")

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
    st.header("Model")
    model_path = st.text_input("Model file path", value = "rf_lifestyle_model (1).pkl")

model = load_model(model_path)

EXPECTED_21 = [
    'age','sex_male','bmi','waist_circumference_cm','hip_circumference_cm','systolic_bp',
    'diastolic_bp','triglycerides_mg_dl','hdl_mg_dl','ldl_mg_dl','alt_u_l','ast_u_l',
    'gammagt_u_l','fasting_glucose_mg_dl','hba1c_percent','alcohol_units_per_week',
    'smoker_current','physical_activity_mins','sleep_duration_hours','diet_score','family_history'
]

st.subheader("User Data Input")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value = 18, max_value = 100, value = 45)
    sex_male = st.selectbox("Sex", ["Female","Male"], index = 1) == "Male"
    bmi = st.number_input("BMI", min_value = 10.0, max_value = 60.0, value = 28.0)
    waist = st.number_input("Waist (cm)", min_value = 40.0, max_value = 200.0, value = 95.0)
    hip = st.number_input("Hip (cm)", min_value = 40.0, max_value = 200.0, value = 102.0)
with col2:
    sbp = st.number_input("Systolic BP", min_value = 80, max_value = 220, value = 125)
    dbp = st.number_input("Diastolic BP", min_value = 40, max_value = 140, value = 80)
    tg = st.number_input("Triglycerides (mg/dL)", min_value = 30, max_value = 1500, value = 160)
    hdl = st.number_input("HDL (mg/dL)", min_value = 10, max_value = 150, value = 45)
    ldl = st.number_input("LDL (mg/dL)", min_value = 30, max_value = 300, value = 120)
with col3:
    alt = st.number_input("ALT (U/L)", min_value = 1, max_value = 2000, value = 35)
    ast = st.number_input("AST (U/L)", min_value = 1, max_value = 2000, value = 30)
    ggt = st.number_input("GGT (U/L)", min_value = 1, max_value = 2000, value = 40)
    fpg = st.number_input("Fasting glucose (mg/dL)", min_value = 40, max_value = 600, value = 98)
    hba1c = st.number_input("HbA1c (%)", min_value = 3.0, max_value = 20.0, value = 5.6)

col4, col5, col6 = st.columns(3)
with col4:
    alcohol = st.number_input("Alcohol units/week", min_value = 0, max_value = 200, value = 2)
with col5:
    smoker = st.selectbox("Smoker", ["No","Yes"], index = 0) == "Yes"
with col6:
    activity = st.number_input("Physical activity (mins/day)", min_value = 0, max_value = 600, value = 30)

sleep = st.number_input("Sleep duration (hours)", min_value = 3.0, max_value = 14.0, value = 7.0)
diet = st.slider("Diet score (higher is better)", min_value = 0, max_value = 100, value = 50)
family = st.selectbox("Family history of NAFLD", ["No","Yes"], index = 0) == "Yes"
row = [
    int(age), 1 if sex_male else 0, float(bmi), float(waist), float(hip), int(sbp), int(dbp),
    int(tg), int(hdl), int(ldl), int(alt), int(ast), int(ggt), int(fpg), float(hba1c),
    int(alcohol), 1 if smoker else 0, int(activity), float(sleep), int(diet), 1 if family else 0
]

y_proba = None
y_pred = None
if model is not None:
    try:
        df_in = pd.DataFrame([row], columns = EXPECTED_21)
        if hasattr(model, 'predict_proba'):
            y_proba = float(model.predict_proba(df_in)[0][1])
        y_pred = int(model.predict(df_in)[0])
    except Exception as e:
        st.error("Prediction failed: " + str(e))
def risk_label(p):
    if p is None:
        return "Unknown"
    if p < 0.4:
        return "Low"
    if p < 0.6:
        return "Borderline"
    return "High"

st.subheader("Risk")
if y_proba is not None:
    pct = int(round(y_proba * 100))
    label = risk_label(y_proba)
    st.write("Predicted probability: " + str(pct) + "% (" + label + ")")
    fig, ax = plt.subplots(figsize = (6, 0.6))
    ax.barh([0], [pct], color = '#2b8cbe')
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel('Risk %')
    ax.set_title('Risk')
    plt.tight_layout()
    plt.show()
else:
    if y_pred is not None:
        st.write("Predicted class: " + ("Positive" if y_pred == 1 else "Negative"))
def approx_contributions(clf, x_row, names):
    try:
        if hasattr(clf, 'feature_importances_'):
            imps = clf.feature_importances_
            vals = np.array(x_row)
            scl = (vals - np.mean(vals))
            raw = imps * scl
            pairs = list(zip(names, raw))
            pairs.sort(key = lambda t: abs(t[1]), reverse = True)
            return pairs
    except Exception:
        pass
    return [(names[i], 0.0) for i in range(len(names))]

if y_pred is not None:
    contribs = approx_contributions(model, row, EXPECTED_21)
    topk = contribs[:8]
    st.subheader("Top feature influences")
    try:
        names = [n for n, v in topk]
        vals = [float(v) for n, v in topk]
        fig2, ax2 = plt.subplots(figsize = (6, 3))
        sns.barplot(x = vals, y = names, palette = 'viridis', ax = ax2)
        ax2.set_title('Approx contributions (not SHAP)')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        st.info("Could not draw contributions: " + str(e))
def render_html_report(prob_pct, label, input_row, contrib_pairs):
    try:
        items = []
        for i in range(len(EXPECTED_21)):
            items.append('<li>' + EXPECTED_21[i] + ': ' + str(input_row[i]) + '</li>')
        feats = []
        for name, val in contrib_pairs[:10]:
            feats.append('<li>' + name + ': ' + str(round(float(val), 4)) + '</li>')
        html = """
        <html>
        <head><meta charset="utf-8"><title>NAFLD Report</title></head>
        <body>
        <h2>NAFLD Lifestyle Risk Report</h2>
        <p><b>Predicted probability:</b> """ + str(prob_pct) + " (% (" + label + "))</p>"""
        html = html + "<h3>Inputs</h3><ul>" + "".join(items) + "</ul>"
        html = html + "<h3>Top features</h3><ul>" + "".join(feats) + "</ul>"
        html = html + "</body></html>"
        return html
    except Exception as e:
        return "<html><body>Error building report: " + str(e) + "</body></html>"
if st.button("Save HTML report"):
    prob_pct = int(round(y_proba * 100)) if y_proba is not None else (100 if (y_pred is not None and y_pred == 1) else 0)
    label = risk_label(y_proba) if y_proba is not None else ("Positive" if (y_pred is not None and y_pred == 1) else "Negative")
    contribs_all = approx_contributions(model, row, EXPECTED_21) if model is not None else []
    html = render_html_report(prob_pct, label, row, contribs_all)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join('reports', 'report_' + ts + '.html')
    with open(out_path, 'w', encoding = 'utf-8') as f:
        f.write(html)
    st.success("Saved HTML report to " + out_path)
