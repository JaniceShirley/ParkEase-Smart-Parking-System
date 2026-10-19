"""
Focused Test Suite for Checkpoint 2B:
- Dataset schema and type validation.
- Invalid input rejection (negative durations, bad categories, out-of-bounds congestion).
- Reproducible train/test split.
- Metric calculation accuracy.
- MLflow parameter, tag, and metric logging.
- Model artifact creation and loading.
- Prediction output shape, finite values, and non-negative boundaries.
"""

import os
import pytest
import numpy as np
import pandas as pd
import joblib
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.generate_synthetic_data import (
    validate_dataset,
    ALLOWED_VEHICLE_TYPES,
    ALLOWED_EV_VALUES,
    ALLOWED_WEATHER_VALUES
)
from ml.train import (
    load_and_validate_dataset,
    build_pipeline,
    train_and_track,
    ALL_INPUT_FEATURES,
    TARGET_FEATURE,
    compute_file_sha256
)

DATA_PATH = "ml/data/parking_synthetic_demo.csv"
MODEL_PATH = "models/xgboost_parking_v1.pkl"


def test_dataset_exists_and_checksum():
    """Verify demonstration dataset exists with valid checksum."""
    assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}"
    checksum = compute_file_sha256(DATA_PATH)
    assert len(checksum) == 64
    assert checksum == "4b4110df28576f05e493c264f0554eda21aeaba9ae8ca83d631903a09110feba"


def test_dataset_schema_and_types():
    """Test schema, row count, null values, and valid ranges."""
    df = pd.read_csv(DATA_PATH)
    report = validate_dataset(df, expected_records=2500)
    assert report["is_valid"] is True, f"Validation errors: {report['errors']}"
    assert report["row_count"] == 2500
    assert report["column_count"] == 6
    assert sum(report["missing_values"].values()) == 0
    assert report["duplicates"] == 0


def test_invalid_input_rejection(tmp_path):
    """Test that training pipeline rejects datasets with invalid or negative values."""
    df = pd.read_csv(DATA_PATH)

    # 1. Negative duration rejection
    bad_df1 = df.copy()
    bad_df1.loc[0, "Duration (min)"] = -10.0
    bad_csv1 = tmp_path / "bad_dur.csv"
    bad_df1.to_csv(bad_csv1, index=False)
    with pytest.raises(ValueError, match="Negative duration"):
        load_and_validate_dataset(str(bad_csv1))

    # 2. Out-of-bounds congestion rejection
    bad_df2 = df.copy()
    bad_df2.loc[0, "Congestion Level"] = 1.5
    bad_csv2 = tmp_path / "bad_cong.csv"
    bad_df2.to_csv(bad_csv2, index=False)
    with pytest.raises(ValueError, match="Congestion Level values out of bounds"):
        load_and_validate_dataset(str(bad_csv2))

    # 3. Invalid vehicle category rejection
    bad_df3 = df.copy()
    bad_df3.loc[0, "Vehicle Type"] = "spaceship"
    bad_csv3 = tmp_path / "bad_veh.csv"
    bad_df3.to_csv(bad_csv3, index=False)
    with pytest.raises(ValueError, match="Invalid Vehicle Type"):
        load_and_validate_dataset(str(bad_csv3))

    # 4. Negative target fee rejection
    bad_df4 = df.copy()
    bad_df4.loc[0, "Parking Fee (INR)"] = -5.0
    bad_csv4 = tmp_path / "bad_fee.csv"
    bad_df4.to_csv(bad_csv4, index=False)
    with pytest.raises(ValueError, match="Negative Parking Fee"):
        load_and_validate_dataset(str(bad_csv4))


def test_reproducible_train_test_split():
    """Verify that train_test_split with fixed random_state=42 is 100% deterministic."""
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(DATA_PATH)
    X = df[ALL_INPUT_FEATURES]
    y = df[TARGET_FEATURE]

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.20, random_state=42)

    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_frame_equal(X_test1, X_test2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    pd.testing.assert_series_equal(y_test1, y_test2)


def test_metric_calculation():
    """Test that metric calculations correctly compute MAE, RMSE, and R²."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([11.0, 19.0, 32.0, 38.0])

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    assert np.isclose(mae, 1.5)
    assert np.isclose(rmse, np.sqrt(2.5))
    assert r2 > 0.95


def test_model_artifact_loading_and_inference():
    """Verify that candidate model loads cleanly and predicts valid shapes and finite values."""
    assert os.path.exists(MODEL_PATH), f"Candidate model not found at {MODEL_PATH}"
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict"), "Loaded model has no predict method"

    # Test single-row inference matching Flask input format
    sample_input = pd.DataFrame([{
        "Duration (min)": 120.0,
        "Vehicle Type": "car",
        "Electric": "electric",
        "Weather": "good",
        "Congestion Level": 0.45
    }], columns=ALL_INPUT_FEATURES)

    prediction = model.predict(sample_input)
    assert len(prediction) == 1
    assert np.isfinite(prediction[0])
    assert prediction[0] > 0.0

    # Test multi-row batch inference
    batch_input = pd.DataFrame([
        [15.0, "2-wheeler", "non-electric", "average", 0.10],
        [60.0, "auto", "electric", "bad", 0.85],
        [300.0, "suv/van", "non-electric", "good", 0.50]
    ], columns=ALL_INPUT_FEATURES)

    batch_preds = model.predict(batch_input)
    assert len(batch_preds) == 3
    assert np.all(np.isfinite(batch_preds))
    assert np.all(batch_preds > 0.0)


def test_mlflow_logged_experiment(tmp_path):
    """Verify that MLflow correctly records experiments, parameters, metrics, and tags in an isolated store."""
    temp_tracking_uri = f"file:{tmp_path}/mlruns"
    temp_model_path = str(tmp_path / "temp_candidate.pkl")

    train_and_track(
        data_path=DATA_PATH,
        model_output=temp_model_path,
        test_size=0.20,
        seed=42,
        force=True,
        experiment_name="ParkEase-Pricing-Engine-Test",
        tracking_uri=temp_tracking_uri
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=temp_tracking_uri)
    exp = client.get_experiment_by_name("ParkEase-Pricing-Engine-Test")
    assert exp is not None, "Experiment 'ParkEase-Pricing-Engine-Test' was not created"

    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) > 0, "No runs recorded in temporary MLflow store"

    latest_run = runs[0]
    assert latest_run.data.tags.get("dataset_type") == "synthetic_demonstration"
    assert "test_mae" in latest_run.data.metrics
    assert "test_rmse" in latest_run.data.metrics
    assert "test_r2" in latest_run.data.metrics
    assert "baseline_mae" in latest_run.data.metrics
    assert "n_estimators" in latest_run.data.params
    assert latest_run.data.params.get("random_seed") == "42"
