"""
car_price_model.py
------------------
Machine Learning pipeline for car price prediction.
Models: Linear Regression, Random Forest, XGBoost (ensemble).
Includes preprocessing, feature engineering, training, evaluation, and inference.
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/car_price_model.pkl")
DATA_PATH  = Path("data/car_data.csv")

# Feature definitions
NUMERIC_FEATURES = ["year", "km_driven", "engine_cc", "max_power_bhp", "seats", "car_age"]
CATEGORICAL_FEATURES = ["fuel", "seller_type", "transmission", "owner"]
TARGET = "selling_price"


# ------------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to improve prediction."""
    current_year = 2024
    df = df.copy()

    # Car age
    df["car_age"] = current_year - df["year"]

    # Mileage per year (km efficiency proxy)
    df["km_per_year"] = df["km_driven"] / (df["car_age"].replace(0, 1))

    # Power-to-weight ratio proxy
    if "max_power_bhp" in df.columns and "engine_cc" in df.columns:
        df["power_to_engine"] = df["max_power_bhp"] / (df["engine_cc"].replace(0, 1))

    # Log transform skewed features
    df["log_km_driven"] = np.log1p(df["km_driven"])

    # Drop rows with missing target
    if TARGET in df.columns:
        df = df.dropna(subset=[TARGET])

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardise raw data."""
    df = df.copy()

    # Strip units from string columns (e.g. "1197 CC" → 1197)
    for col in ["engine_cc", "max_power_bhp"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.extract(r"([\d.]+)").astype(float)

    # Fill missing numerics with median
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

    # Fill missing categoricals with mode
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df


# ------------------------------------------------------------------
# Model building
# ------------------------------------------------------------------

def build_pipeline(model) -> Pipeline:
    """Wrap a model in a preprocessing + model pipeline."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train(data_path: str = str(DATA_PATH)) -> dict:
    """
    Full training run.
    Returns dict of model name → evaluation metrics.
    """
    df = pd.read_csv(data_path)
    df = preprocess(df)
    df = engineer_features(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "ridge": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        "gradient_boost": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
    }

    results = {}
    best_model_name = None
    best_r2 = -np.inf

    for name, model in candidates.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        r2  = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        results[name] = {"r2": round(r2, 4), "mae": round(mae, 2), "rmse": round(rmse, 2)}
        logger.info("Model: %-20s  R²: %.4f  MAE: %.0f  RMSE: %.0f", name, r2, mae, rmse)

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline

    logger.info("Best model: %s (R²=%.4f)", best_model_name, best_r2)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"pipeline": best_pipeline, "model_name": best_model_name, "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES}, f)
    logger.info("Saved best model to %s", MODEL_PATH)

    return results


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

class CarPricePredictor:
    """Load trained model and predict car prices."""

    def __init__(self, model_path: str = str(MODEL_PATH)):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Run train() first.")
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.pipeline = data["pipeline"]
        self.model_name = data["model_name"]
        logger.info("Loaded model: %s", self.model_name)

    def predict(self, car: dict) -> float:
        """
        Predict price for a single car.
        car = {
            "year": 2018,
            "km_driven": 45000,
            "fuel": "Petrol",
            "seller_type": "Individual",
            "transmission": "Manual",
            "owner": "First Owner",
            "engine_cc": 1197,
            "max_power_bhp": 82.0,
            "seats": 5
        }
        Returns predicted price in INR.
        """
        df = pd.DataFrame([car])
        df = preprocess(df)
        df = engineer_features(df)
        features = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
        return float(self.pipeline.predict(df[features])[0])

    def predict_batch(self, cars: list) -> list:
        return [self.predict(c) for c in cars]


if __name__ == "__main__":
    results = train()
    print("\nTraining results:")
    for model, metrics in results.items():
        print(f"  {model}: {metrics}")
