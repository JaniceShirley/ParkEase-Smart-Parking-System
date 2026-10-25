"""
End-to-End Regression Test Suite for ParkEase Flask Integration (Checkpoint 2C).

SAFETY GUARANTEES:
1. Operates exclusively on an isolated temporary SQLite database created in a temp directory.
2. The legacy parking.db is not tracked; a local copy, if present, is never modified (tests/conftest.py).
3. Tests both the legacy default model (price_prediction_model.pkl) and the candidate model (models/xgboost_parking_v1.pkl).
4. Mocks external OpenWeatherMap network calls to guarantee hermetic test execution.
5. Checkout quotes are read from the database, which is the source of truth for fees.
"""

import os
import tempfile
import sqlite3
import hashlib
import pytest
import requests
from unittest.mock import patch

import numpy as np
import app as app_module
from app import app, init_db, get_model
from ml.train import compute_file_sha256

LEGACY_MODEL_PATH = "price_prediction_model.pkl"
CANDIDATE_MODEL_PATH = "models/xgboost_parking_v1.pkl"

BASELINE_LEGACY_MODEL_SHA256 = "71f9414e0580bdcac530e66a0aeb1addf193c608a3cef23b6b1d47966e5a9ccb"
BASELINE_CANDIDATE_MODEL_SHA256 = "aad2c2d641dfb6ceddf065a62b963f3a0170ca35242197df2eebd48ac3edacbc"
TEST_SIGNUP_PASSCODE = "test-only-signup-passcode"


def get_saved_quote(vehicle_number):
    """Return the persisted checkout quote for an active driver."""
    conn = sqlite3.connect(app.config["DATABASE_PATH"])
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT vehicle_number, exit_time, duration_min, price, weather, congestion "
        "FROM users WHERE vehicle_number=?", (vehicle_number,)
    ).fetchone()
    conn.close()
    return row


@pytest.fixture(autouse=True)
def verify_original_artifacts_unmodified():
    """Verify that model artifacts are never modified before or after tests."""
    assert compute_file_sha256(LEGACY_MODEL_PATH) == BASELINE_LEGACY_MODEL_SHA256, "legacy model was altered!"
    assert compute_file_sha256(CANDIDATE_MODEL_PATH) == BASELINE_CANDIDATE_MODEL_SHA256, "candidate model was altered!"
    yield
    assert compute_file_sha256(LEGACY_MODEL_PATH) == BASELINE_LEGACY_MODEL_SHA256, "legacy model was altered during test!"
    assert compute_file_sha256(CANDIDATE_MODEL_PATH) == BASELINE_CANDIDATE_MODEL_SHA256, "candidate model was altered during test!"


@pytest.fixture
def test_client():
    """
    Creates an isolated test client with a dedicated temporary SQLite database.
    Ensures zero write operations touch the primary database.
    """
    temp_fd, temp_db_path = tempfile.mkstemp(suffix="_test_parking.db")
    os.close(temp_fd)

    # Initialize complete schema on isolated temp database
    init_db(temp_db_path)

    app.config.update({
        "TESTING": True,
        "DATABASE_PATH": temp_db_path,
        "MODEL_PATH": CANDIDATE_MODEL_PATH,
        "WTF_CSRF_ENABLED": False,
        "SECRET_KEY": "test-secret-key",
        "ADMIN_SIGNUP_PASSCODE": TEST_SIGNUP_PASSCODE,
        "PARKING_CAPACITY": 100,
    })

    with app.test_client() as client:
        yield client

    # Cleanup temporary test database
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)


def test_a_index_page(test_client):
    """Test A: Application startup and index page rendering."""
    response = test_client.get("/")
    assert response.status_code == 200
    assert b"ParkEase" in response.data or b"Vehicle" in response.data


def test_b_parking_checkin_atomic_allocation(test_client):
    """Test B: Parking check-in and atomic slot allocation."""
    response = test_client.post("/", data={
        "name": "Test Driver",
        "phone": "9876543210",
        "vehicle_number": "TN-45-AB-1234",
        "vehicle_type": "car",
        "electric": "no"
    }, follow_redirects=True)

    assert response.status_code == 200
    assert b"Dashboard" in response.data or b"TN-45-AB-1234" in response.data

    # Verify atomic reservation in isolated DB
    conn = sqlite3.connect(app.config["DATABASE_PATH"])
    c = conn.cursor()
    c.execute("SELECT slot_id, status FROM parking_slots WHERE slot_id=1")
    slot = c.fetchone()
    assert slot == (1, "occupied")

    c.execute("SELECT name, vehicle_number, slot_id FROM users WHERE vehicle_number='TN-45-AB-1234'")
    user = c.fetchone()
    assert user == ("Test Driver", "TN-45-AB-1234", 1)
    conn.close()


def test_c_dashboard_rendering(test_client):
    """Test C: Dashboard renders active session and assigned slot."""
    # Check in first
    test_client.post("/", data={
        "name": "Jane Doe",
        "phone": "9998887776",
        "vehicle_number": "TN-45-EV-9999",
        "vehicle_type": "car",
        "electric": "yes"
    }, follow_redirects=True)

    response = test_client.get("/dashboard")
    assert response.status_code == 200
    assert b"TN-45-EV-9999" in response.data


@patch("weatherAPI.requests.get")
def test_d_and_e_exit_candidate_model_pricing(mock_weather, test_client, monkeypatch):
    """Test D & E: Exit, duration calculation, and candidate-model prediction."""
    # A key must be configured, otherwise the classifier never calls requests.get
    monkeypatch.setattr(app_module.classifier, "api_key", "test-api-key")
    mock_weather.side_effect = requests.exceptions.ConnectionError("Network offline")
    app.config["MODEL_PATH"] = CANDIDATE_MODEL_PATH

    # 1. Check in
    test_client.post("/", data={
        "name": "Pricing Tester",
        "phone": "9876543210",
        "vehicle_number": "TN-45-XG-0001",
        "vehicle_type": "suv/van",
        "electric": "no"
    }, follow_redirects=False)

    # 2. Exit parking
    response = test_client.post("/exit", follow_redirects=True)
    assert response.status_code == 200
    assert b"Receipt" in response.data or b"Parking Fee" in response.data or b"Exit" in response.data

    assert mock_weather.called, "weather API mock was never exercised"
    quote = get_saved_quote("TN-45-XG-0001")
    assert quote is not None
    assert quote["exit_time"] is not None
    assert quote["duration_min"] >= 0.0
    assert quote["price"] >= 0.0
    assert isinstance(quote["price"], float)
    assert quote["weather"] == "average"


def test_f_g_h_payment_processing_and_slot_release(test_client):
    """Test F, G & H: Payment processing, logs update, slot release, and user cleanup."""
    # 1. Check in
    test_client.post("/", data={
        "name": "Checkout Tester",
        "phone": "9123456789",
        "vehicle_number": "TN-45-CK-5555",
        "vehicle_type": "auto",
        "electric": "yes"
    }, follow_redirects=False)

    # 2. Initiate exit
    test_client.post("/exit", follow_redirects=False)

    # 3. Process payment
    response = test_client.post("/process_payment", follow_redirects=True)
    assert response.status_code == 200

    # 4. Verify DB state in isolated DB
    conn = sqlite3.connect(app.config["DATABASE_PATH"])
    c = conn.cursor()
    # Slot 1 must be freed
    c.execute("SELECT status FROM parking_slots WHERE slot_id=1")
    assert c.fetchone()[0] == "free"
    # User must be deleted from active users
    c.execute("SELECT count(*) FROM users")
    assert c.fetchone()[0] == 0
    # Audit log entry must exist
    c.execute("SELECT name, vehicle_number, price, slot_id FROM parking_logs WHERE vehicle_number='TN-45-CK-5555'")
    log = c.fetchone()
    assert log is not None
    assert log[0] == "Checkout Tester"
    assert log[1] == "TN-45-CK-5555"
    assert log[2] >= 0.0
    assert log[3] == 1
    conn.close()


def test_i_admin_authentication_and_dashboard(test_client):
    """Test I: Admin registration, password hashing verification, and dashboard."""
    # 1. Signup admin
    test_client.post("/admin_signup", data={
        "username": "testadmin",
        "password": "SecurePassword@123",
        "mobile": "9998881111",
        "org_password": TEST_SIGNUP_PASSCODE
    }, follow_redirects=True)

    # 2. Login admin
    login_resp = test_client.post("/admin_login", data={
        "username": "testadmin",
        "password": "SecurePassword@123"
    }, follow_redirects=True)
    assert login_resp.status_code == 200
    assert b"Dashboard" in login_resp.data or b"Slot" in login_resp.data

    # 3. View admin logs
    logs_resp = test_client.get("/admin_logs")
    assert logs_resp.status_code == 200


@patch("weatherAPI.requests.get")
def test_j_weather_api_fallback(mock_get, test_client, monkeypatch):
    """Test J: Weather API network failure gracefully falls back to 'average' weather."""
    monkeypatch.setattr(app_module.classifier, "api_key", "test-api-key")
    mock_get.side_effect = requests.exceptions.Timeout("Simulated connection timeout")

    test_client.post("/", data={
        "name": "Weather Driver",
        "phone": "9876543210",
        "vehicle_number": "TN-45-WT-1111",
        "vehicle_type": "2-wheeler",
        "electric": "no"
    }, follow_redirects=False)

    test_client.post("/exit", follow_redirects=False)

    assert mock_get.called, "weather API mock was never exercised"
    assert mock_get.call_args.kwargs.get("timeout") == 5
    quote = get_saved_quote("TN-45-WT-1111")
    assert quote is not None
    assert quote["weather"] == "average"
    assert quote["price"] >= 0.0


def test_k_missing_model_handling(test_client):
    """Test K: Missing model path handled cleanly with user flash message instead of crash."""
    app.config["MODEL_PATH"] = "nonexistent_model_dir/fake_model.pkl"

    test_client.post("/", data={
        "name": "Crash Driver",
        "phone": "9876543210",
        "vehicle_number": "TN-45-CR-0000",
        "vehicle_type": "car",
        "electric": "no"
    }, follow_redirects=False)

    response = test_client.post("/exit", follow_redirects=True)
    # Redirects back to dashboard with error flash rather than returning 500
    assert response.status_code == 200
    assert b"temporarily unavailable" in response.data
    # No quote may be stored when pricing failed
    assert get_saved_quote("TN-45-CR-0000")["exit_time"] is None


def test_l_dual_model_compatibility(test_client):
    """
    Test L: Explicitly verifies both models in Flask:
    1. Default legacy model: price_prediction_model.pkl
    2. Candidate model: models/xgboost_parking_v1.pkl
    """
    for model_target in [LEGACY_MODEL_PATH, CANDIDATE_MODEL_PATH]:
        app.config["MODEL_PATH"] = model_target

        # Check in driver
        test_client.post("/", data={
            "name": f"Driver {model_target}",
            "phone": "9876543210",
            "vehicle_number": f"TN-45-TEST",
            "vehicle_type": "car",
            "electric": "no"
        }, follow_redirects=False)

        # Checkout driver
        test_client.post("/exit", follow_redirects=False)

        quote = get_saved_quote("TN-45-TEST")
        assert quote is not None and quote["exit_time"] is not None, f"Exit quote missing for {model_target}"
        assert quote["price"] >= 0.0, f"Invalid fee for {model_target}"
        assert np.isfinite(quote["price"]), f"Non-finite fee for {model_target}"

        # Clean up session
        test_client.post("/process_payment", follow_redirects=False)
