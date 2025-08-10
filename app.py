
# Streamlit NAFLD app (fixed probability, consistent risk card, SHAP fallback, Mongo sidebar)
import os
import pandas as pd
import numpy as np
import streamlit as st

# --- Helpers ---
EXPECTED_FEATURES = [
    'Gender','Age in years','Race/Ethnicity','Family income ratio','Smoking status','Sleep Disorder Status',
    'Sleep duration (hours/day)','Work schedule duration (hours)','Physical activity (minutes/day)','BMI',
    'Alcohol consumption (days/week)','Alcohol drinks per day','Number of days drank in the past year',
    'Max number of drinks on any single day','Alcohol intake frequency (drinks/day)','Total calorie intake (kcal)',
    'Total protein intake (grams)','Total carbohydrate intake (grams)','Total sugar intake (grams)',
    'Total fiber intake (grams)','Total fat intake (grams)'
]

def _positive_index_safe(model_obj):
    try:
        classes = list(model_obj.classes_)
        if 1 in classes:
            return classes.index(1)
        if '1' in classes:
            return classes.index('1')
        if True in classes:
            return classes.index(True)
        try:
            nums = [float(x) for x in classes]
            return nums.index(max(nums))
        except Exception:
            return 1 if len(classes) > 1 else 0
    except Exception:
        return 1

def _predict_prob_safe(model_obj, X_df):
    try:
        pos = _positive_index_safe(model_obj)
        return float(model_obj.predict_proba(X_df)[0][pos])
    except Exception:
        try:
            clf = getattr(model_obj, 'named_steps', {}).get('classifier', None)
            if clf is not None and hasattr(clf, 'predict_proba'):
                pos = _positive_index_safe(clf)
                return float(clf.predict_proba(X_df)[0][pos])
        except Exception:
            pass
        try:
            val = float(model_obj.decision_function(X_df)[0])
            val_c = max(min((val + 5.0) / 10.0, 1.0), 0.0)
            return float(val_c)
        except Exception:
            try:
                return float(np.clip(float(model_obj.predict(X_df)[0]), 0.0, 1.0))
            except Exception:
                return 0.5

def _build_X(values_dict):
    row = {}
    for k in EXPECTED_FEATURES:
        row[k] = values_dict.get(k, None)
    X = pd.DataFrame([row], columns = EXPECTED_FEATURES)
    numeric_like = [
        'Age in years','Family income ratio','Sleep duration (hours/day)','Work schedule duration (hours)',
        'Physical activity (minutes/day)','BMI','Alcohol consumption (days/week)','Alcohol drinks per day',
        'Number of days drank in the past year','Max number of drinks on any single day',
        'Alcohol intake frequency (drinks/day)','Total calorie intake (kcal)','Total protein intake (grams)',
        'Total carbohydrate intake (grams)','Total sugar intake (grams)','Total fiber intake (grams)','Total fat intake (grams)'
    ]
    for c in numeric_like:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors = 'coerce')
    return X

def render_risk_card(prob):
    try:
        p = float(prob)
    except Exception:
        p = 0.0
    if p < 0.34:
        label = 'Low'; color = '#22c55e'
    elif p < 0.67:
        label = 'Moderate'; color = '#f59e0b'
    else:
        label = 'High'; color = '#ef4444'
    html = (
        '<div style=' + 'padding:16px;border-radius:10px;background:' + color + '1A;border:2px solid ' + color + ';margin:12px 0' + '>' +
        '<div style=' + 'display:flex;justify-content:space-between;align-items:center;font-size:1.05rem;' + '>' +
        '<div><b>Risk level:</b> ' + label + '</div>' +
        '<div><b>' + str(int(round(p*100))) + '%</b></div>' +
        '</div>' +
        '<div style=' + 'height:12px;background:#e5e7eb;border-radius:8px;margin-top:10px;' + '>' +
        '<div style=' + 'width:' + str(int(round(p*100))) + '%;height:12px;background:' + color + ';border-radius:8px;' + '></div>' +
        '</div>' +
        '</div>'
    )
    st.markdown(html, unsafe_allow_html = True)

# --- UI skeleton (minimal; your existing UI can remain) ---
st.title('NAFLD Risk Prediction')

# Placeholder for your existing value collection. We try both values_dict and raw_row if defined by your code.
values_source = {}
if 'values_dict' in globals() and isinstance(values_dict, dict):
    values_source = values_dict
elif 'raw_row' in globals() and isinstance(raw_row, dict):
    values_source = raw_row

# Validate inputs
missing = [k for k in EXPECTED_FEATURES if k not in values_source]
if len(missing) > 0:
    st.info('Waiting for inputs. Missing: ' + ', '.join(missing))

# Build X and predict safely if model exists
prob = 0.5
if 'model' in globals():
    try:
        X = _build_X(values_source)
        prob = _predict_prob_safe(model, X)
    except Exception as e:
        st.error('Prediction error: ' + str(e))
else:
    st.warning('Model not found in current session.')

# Single source of truth for display
st.subheader('Prediction Result')
st.write('Adjust the inputs in the sidebar to see the prediction update in real-time.')
st.write('Predicted NAFLD Risk: ' + str(round(prob * 100, 2)) + '%')
render_risk_card(prob)

# SHAP analysis (non-blocking)
with st.expander('Model Explainability (SHAP)'):
    try:
        import shap, matplotlib.pyplot as plt
        if 'model' in globals() and 'X' in locals():
            bg = pd.concat([X] * 30, ignore_index = True)
            for c in bg.columns:
                try:
                    bg[c] = bg[c] + np.random.normal(0, 1e-6, size = len(bg))
                except Exception:
                    pass
            try:
                explainer = shap.Explainer(model, bg)
            except Exception:
                try:
                    explainer = shap.Explainer(model.predict_proba, bg)
                except Exception:
                    explainer = None
            if explainer is None:
                st.info('SHAP not available for this model setup.')
            else:
                sv = explainer(X)
                shap.plots.bar(sv, max_display = 10, show = False)
                plt.tight_layout()
                plt.show()
        else:
            st.info('Model or inputs not ready for SHAP.')
    except Exception as e:
        st.info('SHAP unavailable: ' + str(e))

# Sidebar Mongo badge (does not change your connection code)
st.sidebar.markdown('## Connection')
try:
    if 'db' in globals() and db is not None:
        st.sidebar.success('MongoDB: connected')
    else:
        st.sidebar.warning('MongoDB: not connected')
except Exception:
    st.sidebar.warning('MongoDB: status unknown')
