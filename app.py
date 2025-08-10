
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title = "NAFLD Lifestyle Risk Predictor", layout = "wide")

# -----------------------
# Mongo connection (uses st.secrets)
# -----------------------
@st.cache_resource
def get_mongo_client():
    try:
        from pymongo import MongoClient
        import certifi
        uri = st.secrets["mongo"]["connection_string"]
        client = MongoClient(uri, tlsCAFile = certifi.where(), serverSelectionTimeoutMS = 20000, connectTimeoutMS = 20000)
        # ping
        client.admin.command("ping")
        return client
    except Exception as e:
        st.warning("Mongo connection not available: " + str(e))
        return None

# -----------------------
# Model + SHAP
# -----------------------
@st.cache_resource
def load_model_cached(path):
    return joblib.load(path)

@st.cache_resource
def build_explainer_cached(model, background_df):
    try:
        return shap.Explainer(model.predict_proba, background_df)
    except Exception:
        # fallback for tree models
        try:
            return shap.Explainer(model)
        except Exception as e:
            st.warning("Could not build SHAP explainer: " + str(e))
            return None

# Risk card UI

def render_risk_card(prob):
    try:
        p = float(prob)
    except Exception:
        p = 0.0
    if p < 0.34:
        label = "Low"; color = "#22c55e"
    elif p < 0.67:
        label = "Medium"; color = "#f59e0b"
    else:
        label = "High"; color = "#ef4444"
    st.markdown(
        "<div style=padding:12px;border-radius:8px;background:" + color + "1A;border:1px solid " + color + ">" +
        "<b>Risk:</b> " + label + " (" + str(int(round(p*100))) + "%)" +
        "<div style=height:10px;background:#e5e7eb;border-radius:6px;margin-top:8px;>" +
        "<div style=width:" + str(int(round(p*100))) + "%;height:10px;background:" + color + ";border-radius:6px;"></div>" +
        "</div></div>", unsafe_allow_html = True)

# -----------------------
# Sidebar: connections and model
# -----------------------
st.title("NAFLD Lifestyle Risk Predictor")
st.caption("Enter values for the model features to get a prediction. Uses Streamlit secrets for MongoDB.")

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

# -----------------------
# Feature inputs (example schema - replace with your real features)
# -----------------------
# Update this list to match your training columns
feature_names = ["age", "bmi", "alt", "ast", "hdl", "tg"]

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value = 10, max_value = 100, value = 45, step = 1)
    bmi = st.number_input("BMI", min_value = 10.0, max_value = 60.0, value = 28.0, step = 0.1)
with col2:
    alt = st.number_input("ALT", min_value = 0, max_value = 200, value = 30, step = 1)
    ast = st.number_input("AST", min_value = 0, max_value = 200, value = 28, step = 1)
with col3:
    hdl = st.number_input("HDL", min_value = 10, max_value = 120, value = 45, step = 1)
    tg = st.number_input("Triglycerides", min_value = 30, max_value = 500, value = 150, step = 5)

row = {
    "age": int(age),
    "bmi": float(bmi),
    "alt": float(alt),
    "ast": float(ast),
    "hdl": float(hdl),
    "tg": float(tg)
}
X = pd.DataFrame([row], columns = feature_names)

# -----------------------
# Predict
# -----------------------
prob = None
if model is not None:
    try:
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[0][1])
        else:
            # fallback: decision_function or predict
            if hasattr(model, "decision_function"):
                from sklearn.preprocessing import MinMaxScaler
                val = float(model.decision_function(X)[0])
                scaler = MinMaxScaler(feature_range = (0.0, 1.0))
                prob = float(scaler.fit_transform(np.array([[val],[0.0],[1.0]])).flatten()[0])
            else:
                pred = float(model.predict(X)[0])
                prob = max(0.0, min(1.0, pred))
    except Exception as e:
        st.error("Prediction failed: " + str(e))

# -----------------------
# Display result
# -----------------------
if prob is not None:
    st.subheader("Prediction")
    render_risk_card(prob)
    st.write("Estimated probability: " + str(int(round(prob*100))) + "%")

# -----------------------
# SHAP explanation
# -----------------------
if prob is not None and model is not None:
    with st.expander("Why this risk? (SHAP)"):
        try:
            # Build a small background from jittered copies of X
            bg = pd.concat([X for _ in range(20)], ignore_index = True)
            for c in bg.columns:
                try:
                    bg[c] = bg[c] + np.random.normal(0, 1e-6, size = len(bg))
                except Exception:
                    pass
            explainer = build_explainer_cached(model, bg)
            if explainer is not None:
                shap_values = explainer(X)
                st.write("Top contributing features:")
                try:
                    shap.plots.waterfall(shap_values[0], show = False)
                    plt.tight_layout()
                    plt.show()
                    st.caption("Waterfall plot showing how each feature pushes the prediction.")
                except Exception:
                    try:
                        shap.plots.bar(shap_values, max_display = 10, show = False)
                        plt.tight_layout()
                        plt.show()
                    except Exception as e2:
                        st.info("Could not render SHAP plots: " + str(e2))
            else:
                st.info("No explainer available.")
        except Exception as e:
            st.info("SHAP explanation unavailable: " + str(e))

# -----------------------
# What-if analysis
# -----------------------
if model is not None:
    st.subheader("What-if analysis")
    wcol1, wcol2, wcol3 = st.columns(3)
    with wcol1:
        age_w = st.slider("Age (what-if)", 10, 100, int(age))
        bmi_w = st.slider("BMI (what-if)", 10, 60, int(round(bmi)))
    with wcol2:
        alt_w = st.slider("ALT (what-if)", 0, 200, int(round(alt)))
        ast_w = st.slider("AST (what-if)", 0, 200, int(round(ast)))
    with wcol3:
        hdl_w = st.slider("HDL (what-if)", 10, 120, int(round(hdl)))
        tg_w = st.slider("Triglycerides (what-if)", 30, 500, int(round(tg)))

    X_whatif = pd.DataFrame([{ "age": age_w, "bmi": float(bmi_w), "alt": float(alt_w), "ast": float(ast_w), "hdl": float(hdl_w), "tg": float(tg_w) }], columns = feature_names)

    try:
        if hasattr(model, "predict_proba"):
            prob_w = float(model.predict_proba(X_whatif)[0][1])
        else:
            pred = float(model.predict(X_whatif)[0])
            prob_w = max(0.0, min(1.0, pred))
        st.write("What-if risk:")
        render_risk_card(prob_w)
        delta_pct = int(round((prob_w - (prob if prob is not None else 0.0)) * 100))
        st.metric(label = "Delta vs current", value = str(int(round(prob_w*100))) + "%", delta = ("+" if delta_pct >= 0 else "") + str(delta_pct) + "%")
    except Exception as e:
        st.info("What-if prediction not available: " + str(e))

# -----------------------
# Download HTML report
# -----------------------
if prob is not None:
    try:
        html = "<html><head><meta charset=utf-8><title>NAFLD Report</title></head><body>"
        html += "<h2>NAFLD Lifestyle Risk Report</h2>"
        html += "<p>Generated: " + datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC") + "</p>"
        html += "<h3>Inputs</h3><pre>" + pd.DataFrame([row]).to_csv(index = False) + "</pre>"
        html += "<h3>Predicted Risk</h3><p>" + str(int(round(prob*100))) + "%</p>"
        html += "</body></html>"
        fname = "nafld_report.html"
        with open(fname, "w", encoding = "utf-8") as f:
            f.write(html)
        with open(fname, "rb") as f:
            st.download_button("Download HTML report", f, file_name = fname, mime = "text/html")
    except Exception as e:
        st.info("Report not available: " + str(e))
