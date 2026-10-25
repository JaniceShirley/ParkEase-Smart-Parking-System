"""
Shared pytest configuration.

Environment variables are pinned *before* the application is imported so that a
developer's local .env (loaded by python-dotenv, which never overrides variables
that are already set) cannot change database paths, enable real weather calls or
inject secrets into the test run.
"""

import os
import shutil
import tempfile

import pytest

_IMPORT_TIME_DIR = tempfile.mkdtemp(prefix="parkease-tests-")

os.environ["APP_ENV"] = "testing"
os.environ.setdefault("FLASK_SECRET_KEY", "test-only-secret-key-not-for-production-use")
os.environ["FLASK_DEBUG"] = "false"
os.environ["DATABASE_PATH"] = os.path.join(_IMPORT_TIME_DIR, "import_time.db")
os.environ["OPENWEATHER_API_KEY"] = ""
os.environ["ADMIN_SIGNUP_PASSCODE"] = ""
os.environ["RATELIMIT_STORAGE_URI"] = "memory://"

TEST_SIGNUP_PASSCODE = "test-only-signup-passcode"
CANDIDATE_MODEL_PATH = "models/xgboost_parking_v1.pkl"


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_IMPORT_TIME_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def local_legacy_db_untouched():
    """
    The legacy parking.db is no longer tracked. If a developer checkout still has a
    local copy it must never be modified; if it is absent, no test may create it.
    """
    from tests.helpers import LOCAL_LEGACY_DB, file_state

    before = file_state(LOCAL_LEGACY_DB)
    yield
    assert file_state(LOCAL_LEGACY_DB) == before, "the local legacy parking.db was created or modified"
    for suffix in ("-journal", "-wal", "-shm"):
        assert not os.path.exists(LOCAL_LEGACY_DB + suffix)


@pytest.fixture
def legacy_db(tmp_path, monkeypatch):
    """A fictional stand-in registered as the protected legacy database for this test."""
    from tests.helpers import make_fictional_legacy_db
    import database

    path = make_fictional_legacy_db(tmp_path / "legacy_repo" / "parking.db")
    monkeypatch.setattr(database, "COMMITTED_BASELINE_DB", str(path))
    return path


@pytest.fixture(autouse=True)
def reset_rate_limits():
    from app import limiter
    limiter.reset()
    yield


@pytest.fixture
def signup_passcode():
    return TEST_SIGNUP_PASSCODE


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Test client bound to a fresh temporary database. Config changes are undone after each test."""
    from app import app, init_db

    db_path = str(tmp_path / "parking.db")
    monkeypatch.setitem(app.config, "TESTING", True)
    monkeypatch.setitem(app.config, "DATABASE_PATH", db_path)
    monkeypatch.setitem(app.config, "MODEL_PATH", CANDIDATE_MODEL_PATH)
    monkeypatch.setitem(app.config, "WTF_CSRF_ENABLED", False)
    monkeypatch.setitem(app.config, "ADMIN_SIGNUP_PASSCODE", TEST_SIGNUP_PASSCODE)
    monkeypatch.setitem(app.config, "PARKING_CAPACITY", 100)
    init_db(db_path)

    with app.test_client() as test_client:
        yield test_client
