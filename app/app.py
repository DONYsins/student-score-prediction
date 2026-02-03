import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ------------------
# Paths & constants
# ------------------
ART_DIR = Path("artifacts")
MODEL_PATH = ART_DIR / "model.joblib"
SCHEMA_PATH = ART_DIR / "schema.json"


# ------------------
# Cached loaders
# ------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------
# UI helpers
# ------------------
def render_numeric_inputs(num_features, num_stats):
    inputs = {}
    for col in num_features:
        stats = num_stats[col]
        max_val = (
            100.0 if col == "class_attendance"
            else float(round(stats["max"] * 1.2))
        )
        inputs[col] = st.number_input(
            label=col.replace("_", " ").title(),
            min_value=float(round(stats["min"] * 0.8)),
            max_value=max_val,
            value=float(stats["mean"]),
            step=1.0,
        )
    return inputs


def render_categorical_inputs(cat_features, cat_choices):
    inputs = {}
    for col in cat_features:
        label = col.replace("_", " ").title()
        choices = cat_choices.get(col, [])

        if choices:
            choice_map = {c.title(): c for c in choices}
            selected = st.selectbox(label, options=choice_map.keys())
            inputs[col] = choice_map[selected]
        else:
            inputs[col] = st.text_input(label)

    return inputs


# ------------------
# App layout
# ------------------
st.set_page_config(
    page_title="Student Exam Score Predictor",
    layout="centered",
)

st.title("🎓 Student Exam Score Predictor")
st.subheader(
    "Enter student details to predict the expected exam score (HGB model)."
)
st.divider()

model = load_model()
schema = load_schema()

num_features = schema["num_features"]
cat_features = schema["cat_features"]
feature_cols = schema["feature_cols"]

st.subheader("Inputs")

inputs = {}
inputs.update(
    render_numeric_inputs(num_features, schema["num_stats"])
)
inputs.update(
    render_categorical_inputs(cat_features, schema["cat_choices"])
)

st.divider()

# ------------------
# Prediction
# ------------------
if st.button("Predict Exam Score"):
    with st.spinner("Predicting exam score..."):
        row = {c: inputs.get(c) for c in feature_cols}
        X_input = pd.DataFrame([row])
        pred = model.predict(X_input)[0]

    st.success(f"Predicted Exam Score: {pred:.2f}")
    st.caption(
        "Prediction generated using a trained HistGradientBoosting model"
    )
