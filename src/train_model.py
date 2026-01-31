import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True)

TRAIN_PATH = "train.csv" 


def build_pipeline(num_features, cat_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_features),
        ],
        remainder="drop",
    )

    hgb = HistGradientBoostingRegressor(
        max_depth=8,
        learning_rate=0.08,
        max_iter=400,
        random_state=42,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", hgb)])


def main():
    train = pd.read_csv(TRAIN_PATH)

    # Separate features/target
    X = train.drop(columns=["id","exam_score"])
    y = train["exam_score"]

    # Identify feature types
    feature_cols = [c for c in X.columns]
    X = X[feature_cols]

    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    model = build_pipeline(num_features, cat_features)

    # Train on full training set
    model.fit(X, y)

    # Save model artifact
    joblib.dump(model, ART_DIR / "model.joblib")

    # Build a lightweight schema for UI (categories + numeric ranges)
    schema = {
        "num_features": num_features,
        "cat_features": cat_features,
        "feature_cols": feature_cols,
        "num_stats": {},
        "cat_choices": {},
    }

    for col in num_features:
        schema["num_stats"][col] = {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "mean": float(X[col].mean()),
        }

    for col in cat_features:
        # keep unique values for dropdowns
        vals = X[col].dropna().astype(str).unique().tolist()
        schema["cat_choices"][col] = vals

    with open(ART_DIR / "schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print("Saved artifacts:")
    print(" - artifacts/model.joblib")
    print(" - artifacts/schema.json")


if __name__ == "__main__":
    main()
