#!/usr/bin/env python3
"""
Reproducible XGBoost Training Pipeline & MLflow Experiment Tracking for ParkEase.

SCIENTIFIC NOTICE:
This pipeline trains and evaluates on a clearly labelled SYNTHETIC DEMONSTRATION DATASET.
Metrics generated (MAE, RMSE, R²) demonstrate pipeline stability, reproducibility,
and MLflow artifact tracking, but do NOT establish real-world predictive parking accuracy.
"""

import os

# Configure MLflow local file store compatibility
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import hashlib
import argparse
import platform
import numpy as np
import pandas as pd
import joblib

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import mlflow
import mlflow.sklearn

NUMERIC_FEATURES = ["Duration (min)", "Congestion Level"]
CATEGORICAL_FEATURES = ["Vehicle Type", "Electric", "Weather"]
ALL_INPUT_FEATURES = [
    "Duration (min)",
    "Vehicle Type",
    "Electric",
    "Weather",
    "Congestion Level"
]
TARGET_FEATURE = "Parking Fee (INR)"

ALLOWED_VEHICLES = ["car", "2-wheeler", "auto", "suv/van"]
ALLOWED_EV = ["electric", "non-electric"]
ALLOWED_WEATHER = ["good", "average", "bad"]

FORBIDDEN_OVERWRITE_PATHS = ["price_prediction_model.pkl", "./price_prediction_model.pkl"]


def compute_file_sha256(filepath):
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_and_validate_dataset(csv_path):
    """
    Load dataset and strictly validate schema, types, missing values, and value ranges.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # 1. Check required columns
    expected_cols = set(ALL_INPUT_FEATURES + [TARGET_FEATURE])
    if not expected_cols.issubset(set(df.columns)):
        missing = expected_cols - set(df.columns)
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # 2. Check missing values
    null_counts = df[ALL_INPUT_FEATURES + [TARGET_FEATURE]].isnull().sum()
    if null_counts.sum() > 0:
        raise ValueError(f"Dataset contains null values: {null_counts.to_dict()}")

    # 3. Check categorical values
    inv_veh = set(df["Vehicle Type"].unique()) - set(ALLOWED_VEHICLES)
    if inv_veh:
        raise ValueError(f"Invalid Vehicle Type values found: {inv_veh}")

    inv_ev = set(df["Electric"].unique()) - set(ALLOWED_EV)
    if inv_ev:
        raise ValueError(f"Invalid Electric values found: {inv_ev}")

    inv_w = set(df["Weather"].unique()) - set(ALLOWED_WEATHER)
    if inv_w:
        raise ValueError(f"Invalid Weather values found: {inv_w}")

    # 4. Check numeric ranges
    if (df["Duration (min)"] < 0).any():
        raise ValueError("Negative duration values found in dataset")

    if (df["Congestion Level"] < 0.0).any() or (df["Congestion Level"] > 1.0).any():
        raise ValueError("Congestion Level values out of bounds [0.0, 1.0]")

    if (df["Parking Fee (INR)"] < 0).any():
        raise ValueError("Negative Parking Fee (INR) values found in dataset")

    return df


def build_pipeline(model_params, random_state=42):
    """
    Build scikit-learn Pipeline with ColumnTransformer and XGBRegressor.
    Preprocessing is fitted exclusively on training data during pipeline.fit().
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    regressor = xgb.XGBRegressor(
        **model_params,
        random_state=random_state
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    return pipeline


def train_and_track(
    data_path="ml/data/parking_synthetic_demo.csv",
    model_output="models/xgboost_parking_v1.pkl",
    test_size=0.20,
    seed=42,
    force=False,
    experiment_name="ParkEase-Pricing-Engine",
    tracking_uri="file:./mlruns"
):
    """
    Execute end-to-end reproducible training with MLflow tracking and baseline comparison.
    """
    # Safety Check: Never overwrite legacy model
    norm_output = os.path.normpath(model_output)
    for forbidden in FORBIDDEN_OVERWRITE_PATHS:
        if os.path.normpath(forbidden) == norm_output:
            raise PermissionError(f"CRITICAL: Refusing to overwrite legacy protected model: {model_output}")

    # Check candidate conflict
    if os.path.exists(model_output) and not force:
        raise FileExistsError(
            f"Candidate model artifact already exists at '{model_output}'. "
            "Use --force to overwrite or choose a different --model-output path."
        )

    # 1. Load and Validate Data
    print(f"[INFO] Loading and validating dataset from: {data_path}")
    df = load_and_validate_dataset(data_path)
    dataset_sha256 = compute_file_sha256(data_path)
    print(f"[INFO] Dataset SHA-256: {dataset_sha256}")
    print(f"[INFO] Total records: {len(df)}")

    X = df[ALL_INPUT_FEATURES]
    y = df[TARGET_FEATURE]

    # 2. Reproducible Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed
    )
    print(f"[INFO] Train samples: {len(X_train)} | Test samples: {len(X_test)} (test_size={test_size})")

    # 3. XGBoost Hyperparameter Specification
    xgb_params = {
        "n_estimators": 150,
        "max_depth": 5,
        "learning_rate": 0.08,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
        "n_jobs": 1
    }

    # 4. Construct Pipeline & Fit Preprocessing + Model strictly on Training Set
    pipeline = build_pipeline(xgb_params, random_state=seed)
    print("[INFO] Fitting XGBoost pipeline on training partition...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate on Held-Out Test Partition
    y_pred = pipeline.predict(X_test)
    y_train_pred = pipeline.predict(X_train)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    train_r2 = float(r2_score(y_train, y_train_pred))

    print(f"[METRIC] XGBoost Test MAE:  {mae:.4f} INR")
    print(f"[METRIC] XGBoost Test RMSE: {rmse:.4f} INR")
    print(f"[METRIC] XGBoost Test R²:   {r2:.4f}")
    print(f"[METRIC] XGBoost Train R²:  {train_r2:.4f}")

    # 6. Train and Evaluate DummyRegressor Baseline (Mean)
    dummy_model = DummyRegressor(strategy="mean")
    dummy_model.fit(X_train, y_train)
    dummy_pred = dummy_model.predict(X_test)

    baseline_mae = float(mean_absolute_error(y_test, dummy_pred))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, dummy_pred)))
    baseline_r2 = float(r2_score(y_test, dummy_pred))

    print(f"[BASELINE] DummyRegressor Test MAE:  {baseline_mae:.4f} INR")
    print(f"[BASELINE] DummyRegressor Test RMSE: {baseline_rmse:.4f} INR")
    print(f"[BASELINE] DummyRegressor Test R²:   {baseline_r2:.4f}")

    # 7. MLflow Experiment Tracking
    print(f"[INFO] Logging experiment run to MLflow store: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"xgboost-seed{seed}-v1") as run:
        # Parameters
        mlflow.log_params({
            **xgb_params,
            "random_seed": seed,
            "test_split_ratio": test_size,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "dataset_filename": os.path.basename(data_path),
            "dataset_sha256": dataset_sha256,
            "numeric_features": str(NUMERIC_FEATURES),
            "categorical_features": str(CATEGORICAL_FEATURES)
        })

        # System & Library Environment Tags
        mlflow.set_tags({
            "dataset_type": "synthetic_demonstration",
            "model_type": "XGBRegressor",
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "xgboost_version": xgb.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "candidate_output_path": model_output
        })

        # Evaluation Metrics
        mlflow.log_metrics({
            "test_mae": mae,
            "test_rmse": rmse,
            "test_r2": r2,
            "train_r2": train_r2,
            "baseline_mae": baseline_mae,
            "baseline_rmse": baseline_rmse,
            "baseline_r2": baseline_r2
        })

        # Log Scikit-learn Pipeline Model to MLflow Artifacts
        # (MLflow 2.x API: `artifact_path`; the `name` keyword only exists in MLflow 3.x)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="pipeline_model",
            serialization_format="cloudpickle"
        )

        run_id = run.info.run_id
        print(f"[MLFLOW] Successfully logged run with ID: {run_id}")

    # 8. Save Candidate Model Artifact Locally
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump(pipeline, model_output)
    candidate_sha256 = compute_file_sha256(model_output)
    print(f"[SUCCESS] Candidate model saved to: {model_output}")
    print(f"[INFO] Candidate Model SHA-256: {candidate_sha256}")

    return {
        "run_id": run_id,
        "dataset_sha256": dataset_sha256,
        "candidate_sha256": candidate_sha256,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "xgb_metrics": {"mae": mae, "rmse": rmse, "r2": r2, "train_r2": train_r2},
        "baseline_metrics": {"mae": baseline_mae, "rmse": baseline_rmse, "r2": baseline_r2}
    }


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost pricing engine with MLflow tracking.")
    parser.add_argument("--data-path", type=str, default="ml/data/parking_synthetic_demo.csv", help="Input dataset path")
    parser.add_argument("--model-output", type=str, default="models/xgboost_parking_v1.pkl", help="Output model path")
    parser.add_argument("--test-size", type=float, default=0.20, help="Held-out test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--force", action="store_true", help="Force overwrite candidate model if already exists")
    parser.add_argument("--experiment-name", type=str, default="ParkEase-Pricing-Engine", help="MLflow experiment name")
    parser.add_argument("--tracking-uri", type=str, default="file:./mlruns", help="MLflow tracking store URI")
    args = parser.parse_args()

    train_and_track(
        data_path=args.data_path,
        model_output=args.model_output,
        test_size=args.test_size,
        seed=args.seed,
        force=args.force,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri
    )


if __name__ == "__main__":
    main()
