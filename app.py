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
import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO

# === Julius minimal helpers (no secrets/mongo changes) ===
import pandas as _pd
import numpy as _np
import streamlit as _st

EXPECTED_FEATURES = [
    'Gender','Age in years','Race/Ethnicity','Family income ratio','Smoking status','Sleep Disorder Status',
    'Sleep duration (hours/day)','Work schedule duration (hours)','Physical activity (minutes/day)','BMI',
    'Alcohol consumption (days/week)','Alcohol drinks per day','Number of days drank in the past year',
    'Max number of drinks on any single single day','Alcohol intake frequency (drinks/day)',
    'Total calorie intake (kcal)','Total protein intake (grams)','Total carbohydrate intake (grams)',
    'Total sugar intake (grams)','Total fiber intake (grams)','Total fat intake (grams)'
]

def _build_X_from_values(values_dict):
    row = {}
    for k in EXPECTED_FEATURES:
        v = values_dict.get(k, None)
        row[k] = v
    X = _pd.DataFrame([row], columns=EXPECTED_FEATURES)
    numeric_like = [
        'Age in years','Family income ratio','Sleep duration (hours/day)','Work schedule duration (hours)',
        'Physical activity (minutes/day)','BMI','Alcohol consumption (days/week)','Alcohol drinks per day',
        'Number of days drank in the past year','Max number of drinks on any single single day',
        'Alcohol intake frequency (drinks/day)','Total calorie intake (kcal)','Total protein intake (grams)',
        'Total carbohydrate intake (grams)','Total sugar intake (grams)','Total fiber intake (grams)','Total fat intake (grams)'
    ]
    for c in numeric_like:
        if c in X.columns:
            X[c] = _pd.to_numeric(X[c], errors='coerce')
    return X

def _positive_index(model_obj):
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
        pos = _positive_index(model_obj)
        return float(model_obj.predict_proba(X_df)[0][pos])
    except Exception:
        try:
            clf = model_obj.named_steps.get('classifier', None)
            if clf is not None and hasattr(clf, 'predict_proba'):
                pos = _positive_index(clf)
                return float(clf.predict_proba(X_df)[0][pos])
        except Exception:
            pass
        try:
            val = float(model_obj.decision_function(X_df)[0])
            val_c = max(min((val + 5.0) / 10.0, 1.0), 0.0)
            return float(val_c)
        except Exception:
            try:
                return float(_np.clip(float(model_obj.predict(X_df)[0]), 0.0, 1.0))
            except Exception:
                return 0.5

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
        '<div style="padding:16px;border-radius:10px;background:' + color + '1A;border:2px solid ' + color + ';margin:12px 0">' +
        '<div style="display:flex;justify-content:space-between;align-items:center;font-size:1.05rem;">' +
        '<div><b>Risk level:</b> ' + label + '</div>' +
        '<div><b
