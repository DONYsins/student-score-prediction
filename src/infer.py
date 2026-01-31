import json
from pathlib import Path

import joblib
import pandas as pd

ART_DIR = Path("artifacts")
MODEL_PATH = ART_DIR / "model.joblib"
SCHEMA_PATH = ART_DIR / "schema.json"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return model, schema


def predict_one(input_dict: dict) -> float:
    """
    input_dict: keys should match schema['feature_cols'] (excluding exam_score)
    """
    model, schema = load_artifacts()

    # Ensure correct columns/order
    feature_cols = schema["feature_cols"]
    row = {c: input_dict.get(c, None) for c in feature_cols}
    X = pd.DataFrame([row])

    pred = model.predict(X)[0]
    return float(pred)
