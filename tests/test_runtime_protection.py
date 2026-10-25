"""
Regression tests for runtime safety guards:

* the legacy parking.db location in the repository folder can never become the
  runtime database, however DATABASE_PATH spells it (relative, absolute, '..',
  case variant, symlink, hard link), whether or not a local legacy file exists;
* APP_ENV cannot be used to sidestep production protections.

The legacy file is no longer tracked. These tests never need it: wherever "the
protected legacy database" is required they build a small *fictional* stand-in in a
temporary folder and register it as the protected file. tests/conftest.py also
verifies that a developer's local copy, if any, is never modified or created.
"""

import os
import shutil
import subprocess
import sys

import pytest
from dotenv import dotenv_values

import config
import database
from tests.helpers import REPO_ROOT, file_state, make_fictional_legacy_db

STRONG_TEST_KEY = "k" * 48
LEGACY_MESSAGE = "legacy parking.db location"

# Shape of the .env.example shipped before this branch (placeholders only).
OLD_ENV_TEMPLATE = """\
APP_ENV=development
FLASK_SECRET_KEY=replace_with_a_secure_random_key_in_production
FLASK_DEBUG=False
DATABASE_PATH=parking.db
MODEL_PATH=price_prediction_model.pkl
OPENWEATHER_API_KEY=your_openweather_api_key_here
ADMIN_SIGNUP_PASSCODE=your_organization_signup_passcode_here
"""


def sqlite_sidecars(path):
    return [p for p in (path + "-journal", path + "-wal", path + "-shm") if os.path.exists(p)]


def is_case_insensitive(path):
    directory, name = os.path.split(path)
    return os.path.exists(os.path.join(directory, name.upper()))


@pytest.fixture
def legacy_repo(tmp_path, monkeypatch):
    """A fake repository folder whose parking.db is a fictional stand-in, registered as protected."""
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    legacy = make_fictional_legacy_db(repo / "parking.db")
    monkeypatch.setattr(database, "COMMITTED_BASELINE_DB", str(legacy))
    return repo


@pytest.fixture
def fresh_checkout_repo(tmp_path, monkeypatch):
    """A fake repository folder with no legacy file at all, as in a fresh clone."""
    repo = tmp_path / "fresh"
    (repo / "tests").mkdir(parents=True)
    monkeypatch.setattr(database, "COMMITTED_BASELINE_DB", str(repo / "parking.db"))
    return repo


# ---------------------------------------------------------------------------
# Identity guard
# ---------------------------------------------------------------------------

def test_guard_targets_the_repository_parking_db_location():
    assert database.COMMITTED_BASELINE_DB == os.path.join(REPO_ROOT, "parking.db")


def test_absolute_and_normalised_paths_are_detected(legacy_repo, monkeypatch):
    legacy = str(legacy_repo / "parking.db")
    assert database.refers_to_committed_baseline(legacy)
    assert database.refers_to_committed_baseline(os.path.join(legacy_repo, "tests", "..", "parking.db"))

    monkeypatch.chdir(legacy_repo)
    assert database.refers_to_committed_baseline("parking.db")
    assert database.refers_to_committed_baseline("./parking.db")

    monkeypatch.chdir(legacy_repo / "tests")
    assert database.refers_to_committed_baseline("../parking.db")
    assert not database.refers_to_committed_baseline("parking.db"), "relative paths resolve against the CWD"


def test_case_variant_is_detected(legacy_repo):
    if not is_case_insensitive(str(legacy_repo / "parking.db")):
        pytest.skip("filesystem is case-sensitive; a case variant is a different file")
    assert database.refers_to_committed_baseline(str(legacy_repo / "PARKING.DB"))


def test_symlink_and_hard_link_are_detected(legacy_repo, tmp_path):
    legacy = legacy_repo / "parking.db"
    symlink, hardlink = tmp_path / "innocent-looking.db", tmp_path / "another-name.db"
    try:
        os.symlink(legacy, symlink)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported here")
    assert database.refers_to_committed_baseline(str(symlink))
    try:
        os.link(legacy, hardlink)
    except (OSError, NotImplementedError):
        pytest.skip("hard links not supported here")
    assert database.refers_to_committed_baseline(str(hardlink))


def test_copies_and_other_files_are_not_the_legacy_db(legacy_repo, tmp_path, monkeypatch):
    copy = tmp_path / "parking.db"
    shutil.copyfile(legacy_repo / "parking.db", copy)
    assert not database.refers_to_committed_baseline(str(copy)), "a private copy is a different file"
    assert not database.refers_to_committed_baseline(str(tmp_path / "missing.db"))
    assert not database.refers_to_committed_baseline("instance/parking.db")

    monkeypatch.chdir(tmp_path)
    os.remove(copy)
    assert not database.refers_to_committed_baseline("parking.db")


def test_missing_legacy_file_location_is_still_refused(fresh_checkout_repo, monkeypatch):
    # Fresh clone: no legacy file exists, but its location must not be reused either.
    location = fresh_checkout_repo / "parking.db"
    assert database.refers_to_committed_baseline(str(location))
    assert database.refers_to_committed_baseline(os.path.join(fresh_checkout_repo, "tests", "..", "parking.db"))
    monkeypatch.chdir(fresh_checkout_repo)
    assert database.refers_to_committed_baseline("parking.db")
    assert database.refers_to_committed_baseline("./parking.db")
    assert not database.refers_to_committed_baseline("instance/parking.db")
    assert not location.exists()


def test_filesystem_errors_propagate(monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("simulated")
    monkeypatch.setattr(database.os.path, "samefile", denied)
    with pytest.raises(PermissionError):
        database.refers_to_committed_baseline("anything.db")


def test_init_db_refuses_the_legacy_db(legacy_repo):
    legacy = str(legacy_repo / "parking.db")
    before = file_state(legacy)
    with pytest.raises(database.CommittedDatabaseError):
        database.init_db(legacy, 100)
    assert file_state(legacy) == before
    assert sqlite_sidecars(legacy) == []


def test_init_db_refuses_to_recreate_missing_legacy_db(fresh_checkout_repo):
    location = fresh_checkout_repo / "parking.db"
    with pytest.raises(database.CommittedDatabaseError):
        database.init_db(str(location), 100)
    assert not location.exists()


# ---------------------------------------------------------------------------
# Startup configuration
# ---------------------------------------------------------------------------

ENVIRONMENTS = [
    {"APP_ENV": "development"},
    {"APP_ENV": "testing", "PYTEST_VERSION": "8"},
    {"APP_ENV": "production", "FLASK_SECRET_KEY": STRONG_TEST_KEY},
]
SPELLINGS = ["parking.db", "./parking.db", "tests/../parking.db", "ABSOLUTE"]


@pytest.mark.parametrize("environ", ENVIRONMENTS)
@pytest.mark.parametrize("db_path", SPELLINGS)
@pytest.mark.parametrize("repo_fixture", ["legacy_repo", "fresh_checkout_repo"])
def test_settings_refuse_legacy_database(request, monkeypatch, environ, db_path, repo_fixture):
    repo = request.getfixturevalue(repo_fixture)
    monkeypatch.chdir(repo)
    if db_path == "ABSOLUTE":
        db_path = str(repo / "parking.db")
    with pytest.raises(config.ConfigurationError, match=LEGACY_MESSAGE) as excinfo:
        config.load_settings({**environ, "DATABASE_PATH": db_path})
    assert str(repo) not in str(excinfo.value), "error message should not reveal filesystem layout"


def test_settings_fail_closed_on_unverifiable_path(monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("simulated")
    monkeypatch.setattr(database.os.path, "samefile", denied)
    with pytest.raises(config.ConfigurationError, match="could not be verified"):
        config.load_settings({"APP_ENV": "development", "DATABASE_PATH": "somewhere.db"})


@pytest.mark.parametrize("repo_fixture", ["legacy_repo", "fresh_checkout_repo"])
def test_old_env_template_cannot_reactivate_legacy_database(request, tmp_path, monkeypatch, repo_fixture):
    old_env = tmp_path / "old.env"
    old_env.write_text(OLD_ENV_TEMPLATE)
    values = dotenv_values(old_env)
    assert values["DATABASE_PATH"] == "parking.db"

    monkeypatch.chdir(request.getfixturevalue(repo_fixture))
    with pytest.raises(config.ConfigurationError, match=LEGACY_MESSAGE):
        config.load_settings(values)
    with pytest.raises(config.ConfigurationError, match=LEGACY_MESSAGE):
        config.load_settings({**values, "APP_ENV": "production", "FLASK_SECRET_KEY": STRONG_TEST_KEY})


def test_default_database_is_not_the_legacy_location(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    settings = config.load_settings({"APP_ENV": "development"})
    assert settings.database_path == "instance/parking.db"
    assert not database.refers_to_committed_baseline(settings.database_path)


# ---------------------------------------------------------------------------
# Startup in a separate process, against a copy of the application code
# ---------------------------------------------------------------------------

def _copy_app(root):
    for name in ("app.py", "config.py", "database.py", "weatherAPI.py"):
        shutil.copyfile(os.path.join(REPO_ROOT, name), root / name)
    for name in ("templates", "static"):
        shutil.copytree(os.path.join(REPO_ROOT, name), root / name)
    return root


@pytest.fixture(scope="module")
def repo_copy(tmp_path_factory):
    """Application code plus a fictional legacy parking.db, which is what the guard protects there."""
    root = _copy_app(tmp_path_factory.mktemp("repo_copy"))
    make_fictional_legacy_db(root / "parking.db")
    return root


@pytest.fixture(scope="module")
def fresh_copy(tmp_path_factory):
    """Application code without any legacy parking.db, like a fresh clone."""
    return _copy_app(tmp_path_factory.mktemp("fresh_copy"))


def run_in_copy(repo_copy, args, timeout=180, **env_overrides):
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        # Present-but-empty so python-dotenv cannot fill them from any .env
        "FLASK_SECRET_KEY": "",
        "FLASK_DEBUG": "",
        "ADMIN_SIGNUP_PASSCODE": "",
        "OPENWEATHER_API_KEY": "",
        "DATABASE_PATH": "instance/runtime.db",
    }
    env.update(env_overrides)
    return subprocess.run(args, cwd=repo_copy, env=env, capture_output=True, text=True, timeout=timeout)


def import_app(repo_copy, **env_overrides):
    return run_in_copy(repo_copy, [sys.executable, "-c", "import app"], **env_overrides)


def test_startup_control_case_uses_separate_database(repo_copy):
    result = import_app(repo_copy, APP_ENV="development")
    assert result.returncode == 0, result.stderr[-2000:]
    assert (repo_copy / "instance" / "runtime.db").exists()


@pytest.mark.parametrize("variant", ["relative", "dot-segments", "absolute", "case"])
def test_startup_refuses_legacy_database(repo_copy, variant):
    legacy = repo_copy / "parking.db"
    db_path = {
        "relative": "parking.db",
        "dot-segments": "templates/../parking.db",
        "absolute": str(legacy),
        "case": "PARKING.DB",
    }[variant]
    if variant == "case" and not is_case_insensitive(str(legacy)):
        pytest.skip("filesystem is case-sensitive")

    before = file_state(str(legacy))
    result = import_app(repo_copy, APP_ENV="development", DATABASE_PATH=db_path)

    assert result.returncode != 0
    assert LEGACY_MESSAGE in result.stderr
    assert file_state(str(legacy)) == before, "startup must not migrate or open the legacy file for writing"
    assert sqlite_sidecars(str(legacy)) == []


@pytest.mark.parametrize("db_path", ["parking.db", "templates/../parking.db"])
def test_fresh_checkout_startup_refuses_and_creates_nothing(fresh_copy, db_path):
    result = import_app(fresh_copy, APP_ENV="development", DATABASE_PATH=db_path)
    assert result.returncode != 0
    assert LEGACY_MESSAGE in result.stderr
    assert not (fresh_copy / "parking.db").exists()


def test_testing_env_refused_outside_pytest(repo_copy):
    result = import_app(repo_copy, APP_ENV="testing")
    assert result.returncode != 0
    assert "APP_ENV=testing is only allowed" in result.stderr


@pytest.mark.parametrize("preload", [True, False], ids=["preload", "no-preload"])
def test_gunicorn_refuses_non_production_env(repo_copy, preload):
    gunicorn = os.path.join(os.path.dirname(sys.executable), "gunicorn")
    if not os.path.exists(gunicorn):
        pytest.skip("gunicorn is not installed next to this interpreter")
    # Explicit development must not be served by Gunicorn, in either loading mode.
    args = [gunicorn] + (["--preload"] if preload else []) + ["-w", "1", "-b", "127.0.0.1:0", "app:app"]
    try:
        result = run_in_copy(repo_copy, args, timeout=60, APP_ENV="development")
    except subprocess.TimeoutExpired:
        pytest.fail("Gunicorn started serving a non-production configuration")
    assert result.returncode != 0
    assert "Gunicorn requires APP_ENV=production" in (result.stdout + result.stderr)


# ---------------------------------------------------------------------------
# APP_ENV selection rules
# ---------------------------------------------------------------------------

GUNICORN = {"SERVER_SOFTWARE": "gunicorn/23.0.0"}


def test_testing_env_requires_pytest_session():
    with pytest.raises(config.ConfigurationError, match="APP_ENV=testing"):
        config.resolve_app_env({"APP_ENV": "testing"})
    assert config.resolve_app_env({"APP_ENV": "testing", "PYTEST_VERSION": "8.3.4"}) == "testing"


@pytest.mark.parametrize("environ", [
    {"APP_ENV": "development"},
    {"APP_ENV": "testing", "PYTEST_VERSION": "8.3.4"},
])
def test_gunicorn_requires_production(environ):
    with pytest.raises(config.ConfigurationError, match="Gunicorn requires APP_ENV=production"):
        config.resolve_app_env({**environ, **GUNICORN})


@pytest.mark.parametrize("environ", [{}, {"FLASK_ENV": "development"}, {"APP_ENV": "production"}])
def test_gunicorn_without_explicit_env_runs_as_production(environ):
    # Missing APP_ENV (and the ignored FLASK_ENV) can no longer select a weaker mode.
    assert config.resolve_app_env({**environ, **GUNICORN}) == "production"


@pytest.mark.parametrize("value", ["true", "1", "false", "maybe"])
def test_non_production_server_override_is_rejected(value):
    for environ in ({"APP_ENV": "development"}, {"APP_ENV": "production"}, {}):
        with pytest.raises(config.ConfigurationError, match="ALLOW_NON_PRODUCTION_SERVER"):
            config.resolve_app_env({**environ, "ALLOW_NON_PRODUCTION_SERVER": value, **GUNICORN})


def test_production_checks_apply_under_gunicorn():
    with pytest.raises(config.ConfigurationError, match="FLASK_SECRET_KEY"):
        config.load_settings({"APP_ENV": "production", **GUNICORN})
    with pytest.raises(config.ConfigurationError, match="FLASK_DEBUG"):
        config.load_settings({"APP_ENV": "production", "FLASK_SECRET_KEY": STRONG_TEST_KEY,
                              "FLASK_DEBUG": "true", **GUNICORN})


def test_flask_env_fallback_selects_production_checks():
    with pytest.raises(config.ConfigurationError, match="FLASK_SECRET_KEY"):
        config.load_settings({"FLASK_ENV": "production"})


@pytest.mark.parametrize("repo_key", [
    "test-secret-key",
    "test-only-secret-key-not-for-production-use",
    "ci-test-secret-key-non-empty",
])
def test_secret_keys_published_in_tests_and_ci_rejected_in_production(repo_key):
    with pytest.raises(config.ConfigurationError, match="FLASK_SECRET_KEY"):
        config.resolve_secret_key("production", repo_key)
