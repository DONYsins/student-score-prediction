import json
from pathlib import Path

import streamlit as st
import joblib
import pandas as pd

ART_DIR = Path("artifacts")
MODEL_PATH = ART_DIR / "model.joblib"
SCHEMA_PATH = ART_DIR / "schema.json"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


st.set_page_config(page_title="Student Exam Score Predictor", layout="centered")
st.title("Student Exam Score Predictor")
st.subheader("Enter student details to predict the expected exam score (HGB model).")
st.divider()    
model = load_model()
schema = load_schema()

num_features = schema["num_features"]
cat_features = schema["cat_features"]
feature_cols = schema["feature_cols"]

st.subheader("Inputs")

inputs = {}

# Numeric inputs
for col in num_features:
    stats = schema["num_stats"][col]
    default = stats["mean"]
    inputs[col] = st.number_input(
        label=col.replace("_", " ").title(),
        min_value=float(round((stats["min"]) * 0.8)),
        max_value=float(round((stats["max"]) * 1.2)),
        value=float(default),
        step=1.0,
    )

# Categorical inputs
for col in cat_features:
    choices = schema["cat_choices"].get(col, [])
    label = col.replace("_", " ").title()
    if choices:
        choice_map = {c.title(): c for c in choices}
        selected_display = st.selectbox(label, options=list(choice_map.keys()))
        inputs[col] = choice_map[selected_display]
    else:
        inputs[col] = st.text_input(label, value="")

st.divider()

if st.button("Predict Exam Score"):
    with st.spinner("Predicting exam score..."):
        # Build dataframe in correct order excluding exam_score
        row = {c: inputs.get(c, None) for c in feature_cols}
        X_input = pd.DataFrame([row])

        pred = model.predict(X_input)[0]

    st.success(f"Predicted Exam Score: {pred:.2f}")
    st.caption("Note: Prediction is produced by a trained HistGradientBoosting model with consistent preprocessing.")
