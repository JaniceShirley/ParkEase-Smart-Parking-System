"""
Security regression tests: secret-key and debug configuration, administrator
signup and login, CSRF protection and login rate limiting.

All credentials below are test-only placeholders.
"""

import os
import re
import sqlite3
import subprocess
import sys

import pytest
from werkzeug.security import check_password_hash, generate_password_hash

import config
from app import app

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRONG_TEST_KEY = "t" * 48


# ---------------------------------------------------------------------------
# Secret key
# ---------------------------------------------------------------------------

def test_production_requires_secret_key():
    with pytest.raises(config.ConfigurationError, match="FLASK_SECRET_KEY"):
        config.resolve_secret_key("production", None)
    with pytest.raises(config.ConfigurationError, match="FLASK_SECRET_KEY"):
        config.resolve_secret_key("production", "   ")


@pytest.mark.parametrize("known_key", sorted(config.KNOWN_INSECURE_SECRET_KEYS))
def test_production_rejects_published_secret_keys(known_key):
    with pytest.raises(config.ConfigurationError):
        config.resolve_secret_key("production", known_key)


def test_production_rejects_short_secret_key():
    with pytest.raises(config.ConfigurationError, match="at least"):
        config.resolve_secret_key("production", "short-but-unique-key")


def test_production_accepts_strong_secret_key():
    assert config.resolve_secret_key("production", STRONG_TEST_KEY) == (STRONG_TEST_KEY, False)


@pytest.mark.parametrize("app_env", ["development", "testing"])
@pytest.mark.parametrize("configured", [None, "", "dev-insecure-secret-key-change-in-prod"])
def test_non_production_never_uses_predictable_key(app_env, configured):
    key_one, generated = config.resolve_secret_key(app_env, configured)
    key_two, _ = config.resolve_secret_key(app_env, configured)
    assert generated is True
    assert key_one != key_two, "fallback key must be random per process"
    assert len(key_one) >= 64
    assert key_one not in config.KNOWN_INSECURE_SECRET_KEYS


def test_app_env_must_be_known():
    with pytest.raises(config.ConfigurationError, match="APP_ENV"):
        config.resolve_app_env({"APP_ENV": "prod"})


def _import_app_in_subprocess(tmp_path, **overrides):
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "DATABASE_PATH": str(tmp_path / "subprocess.db"),
        # Present-but-empty values stop python-dotenv from filling them from a local .env
        "FLASK_SECRET_KEY": "",
        "FLASK_DEBUG": "",
        "ADMIN_SIGNUP_PASSCODE": "",
        "OPENWEATHER_API_KEY": "",
    }
    env.update(overrides)
    return subprocess.run(
        [sys.executable, "-c", "import app"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )


def test_app_refuses_to_start_in_production_without_secret_key(tmp_path):
    result = _import_app_in_subprocess(tmp_path, APP_ENV="production")
    assert result.returncode != 0
    assert "FLASK_SECRET_KEY" in result.stderr
    assert not (tmp_path / "subprocess.db").exists(), "no database should be created on refused startup"


def test_app_refuses_to_start_in_production_with_published_key(tmp_path):
    result = _import_app_in_subprocess(
        tmp_path, APP_ENV="production", FLASK_SECRET_KEY="dev-insecure-secret-key-change-in-prod"
    )
    assert result.returncode != 0
    assert "FLASK_SECRET_KEY" in result.stderr


# ---------------------------------------------------------------------------
# Debug mode
# ---------------------------------------------------------------------------

def test_debug_is_off_by_default():
    assert config.resolve_debug("development", None) is False
    assert config.resolve_debug("production", None) is False


def test_debug_is_explicit_opt_in():
    assert config.resolve_debug("development", "true") is True
    assert config.resolve_debug("development", "0") is False


def test_debug_rejected_in_production():
    with pytest.raises(config.ConfigurationError, match="FLASK_DEBUG"):
        config.resolve_debug("production", "1")


def test_invalid_debug_value_rejected():
    with pytest.raises(config.ConfigurationError, match="FLASK_DEBUG"):
        config.resolve_debug("development", "maybe")


def test_app_refuses_debug_in_production(tmp_path):
    result = _import_app_in_subprocess(
        tmp_path, APP_ENV="production", FLASK_SECRET_KEY=STRONG_TEST_KEY, FLASK_DEBUG="1"
    )
    assert result.returncode != 0
    assert "FLASK_DEBUG" in result.stderr


def test_production_cookies_are_secure_by_default():
    settings = config.load_settings({"APP_ENV": "production", "FLASK_SECRET_KEY": STRONG_TEST_KEY})
    assert settings.session_cookie_secure is True
    assert settings.debug is False


# ---------------------------------------------------------------------------
# Administrator signup passcode
# ---------------------------------------------------------------------------

def test_signup_disabled_without_passcode():
    assert config.resolve_signup_passcode("development", None) is None
    assert config.resolve_signup_passcode("production", "") is None


@pytest.mark.parametrize("weak", ["ParkEase@123", "your_organization_signup_passcode_here", "short-code"])
def test_weak_signup_passcode_disabled_outside_production(weak):
    assert config.resolve_signup_passcode("development", weak) is None


@pytest.mark.parametrize("weak", ["ParkEase@123", "short-code"])
def test_weak_signup_passcode_rejected_in_production(weak):
    with pytest.raises(config.ConfigurationError, match="ADMIN_SIGNUP_PASSCODE"):
        config.resolve_signup_passcode("production", weak)


def test_passcode_comparison():
    assert config.passcode_matches("correct-passcode-123", "correct-passcode-123") is True
    assert config.passcode_matches("correct-passcode-123", "wrong") is False
    assert config.passcode_matches(None, "anything") is False
    assert config.passcode_matches("correct-passcode-123", None) is False


def _admin_rows():
    conn = sqlite3.connect(app.config["DATABASE_PATH"])
    rows = conn.execute("SELECT username, password FROM admin").fetchall()
    conn.close()
    return rows


def _signup(client, passcode, username="testadmin", password="TestOnlyPassword1"):
    return client.post("/admin_signup", data={
        "username": username,
        "password": password,
        "mobile": "0000000001",
        "org_password": passcode,
    })


def test_signup_disabled_when_passcode_not_configured(client, monkeypatch):
    monkeypatch.setitem(app.config, "ADMIN_SIGNUP_PASSCODE", None)
    page = client.get("/admin_signup")
    assert b"registration is disabled" in page.data
    assert b'name="org_password"' not in page.data

    response = _signup(client, "ParkEase@123")
    assert response.status_code == 403
    assert _admin_rows() == []


def test_signup_rejects_wrong_and_legacy_default_passcode(client):
    for wrong in ["ParkEase@123", "", "test-only-signup-passcod"]:
        response = _signup(client, wrong)
        assert response.status_code == 403
    assert _admin_rows() == []


def test_signup_validates_fields(client, signup_passcode):
    response = _signup(client, signup_passcode, password="short")
    assert response.status_code == 400
    assert b"at least 10 characters" in response.data
    assert _admin_rows() == []


def test_signup_stores_hashed_password(client, signup_passcode):
    response = _signup(client, signup_passcode)
    assert response.status_code == 302
    [(username, stored)] = _admin_rows()
    assert username == "testadmin"
    assert stored != "TestOnlyPassword1"
    assert check_password_hash(stored, "TestOnlyPassword1")


# ---------------------------------------------------------------------------
# Administrator login
# ---------------------------------------------------------------------------

def _insert_admin(username, stored_password):
    conn = sqlite3.connect(app.config["DATABASE_PATH"])
    conn.execute("INSERT INTO admin (username, password, mobile) VALUES (?, ?, ?)",
                 (username, stored_password, "0000000002"))
    conn.commit()
    conn.close()


def _login(client, username, password):
    return client.post("/admin_login", data={"username": username, "password": password})


def test_plaintext_stored_password_is_never_accepted(client):
    _insert_admin("legacyadmin", "legacy-plaintext-1")
    response = _login(client, "legacyadmin", "legacy-plaintext-1")
    assert response.status_code == 200
    assert b"Invalid credentials" in response.data
    with client.session_transaction() as sess:
        assert "admin_id" not in sess
    # The legacy value must not be silently upgraded to a hash either
    assert ("legacyadmin", "legacy-plaintext-1") in _admin_rows()


def test_hashed_password_login_and_logout(client):
    _insert_admin("hashedadmin", generate_password_hash("TestOnlyPassword2"))
    response = _login(client, "hashedadmin", "TestOnlyPassword2")
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/admin_dashboard")

    assert client.get("/admin_logout").status_code == 405, "logout must not be triggerable by a GET link"
    response = client.post("/admin_logout")
    assert response.status_code == 302
    with client.session_transaction() as sess:
        assert "admin_id" not in sess


def test_wrong_password_and_unknown_user_rejected(client):
    _insert_admin("hashedadmin", generate_password_hash("TestOnlyPassword2"))
    for username, password in [("hashedadmin", "wrong-password"), ("nobody", "TestOnlyPassword2")]:
        response = _login(client, username, password)
        assert b"Invalid credentials" in response.data
        with client.session_transaction() as sess:
            assert "admin_id" not in sess


def test_cli_reset_password_recovers_legacy_account(client):
    _insert_admin("legacyadmin", "legacy-plaintext-1")
    runner = app.test_cli_runner()
    result = runner.invoke(args=["reset-admin-password", "legacyadmin"],
                           input="TestOnlyPassword3\nTestOnlyPassword3\n")
    assert result.exit_code == 0, result.output
    assert "TestOnlyPassword3" not in result.output

    response = _login(client, "legacyadmin", "TestOnlyPassword3")
    assert response.status_code == 302


def test_cli_create_admin(client):
    runner = app.test_cli_runner()
    result = runner.invoke(args=["create-admin", "--username", "cliadmin", "--mobile", "0000000003"],
                           input="TestOnlyPassword4\nTestOnlyPassword4\n")
    assert result.exit_code == 0, result.output
    [(username, stored)] = _admin_rows()
    assert username == "cliadmin"
    assert check_password_hash(stored, "TestOnlyPassword4")

    duplicate = runner.invoke(args=["create-admin", "--username", "cliadmin", "--mobile", "0000000003"],
                              input="TestOnlyPassword4\nTestOnlyPassword4\n")
    assert duplicate.exit_code != 0
    assert "already exists" in duplicate.output


# ---------------------------------------------------------------------------
# CSRF
# ---------------------------------------------------------------------------

def _csrf_token(html):
    match = re.search(rb'name="csrf_token" value="([^"]+)"', html)
    assert match, "page is missing a CSRF token field"
    return match.group(1).decode()


CHECKIN_FORM = {
    "name": "Csrf Tester",
    "phone": "0000000004",
    "vehicle_number": "TEST-0004",
    "vehicle_type": "car",
    "electric": "no",
}


def test_post_without_csrf_token_is_rejected(client, monkeypatch):
    monkeypatch.setitem(app.config, "WTF_CSRF_ENABLED", True)
    response = client.post("/", data=CHECKIN_FORM)
    assert response.status_code == 400
    assert b"Form expired" in response.data

    conn = sqlite3.connect(app.config["DATABASE_PATH"])
    assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
    conn.close()


@pytest.mark.parametrize("path", ["/admin_login", "/admin_signup", "/exit", "/process_payment", "/admin_logout"])
def test_state_changing_endpoints_require_csrf(client, monkeypatch, path):
    monkeypatch.setitem(app.config, "WTF_CSRF_ENABLED", True)
    assert client.post(path, data={}).status_code == 400


def test_full_checkout_with_csrf_enabled(client, monkeypatch):
    monkeypatch.setitem(app.config, "WTF_CSRF_ENABLED", True)

    token = _csrf_token(client.get("/").data)
    response = client.post("/", data={**CHECKIN_FORM, "csrf_token": token})
    assert response.status_code == 302

    dashboard = client.get("/dashboard")
    response = client.post("/exit", data={"csrf_token": _csrf_token(dashboard.data)})
    assert response.status_code == 302

    receipt = client.get("/exit_receipt")
    response = client.post("/process_payment", data={"csrf_token": _csrf_token(receipt.data)})
    assert response.status_code == 302
    with client.session_transaction() as sess:
        assert "user_id" not in sess


@pytest.mark.parametrize("template", [
    "index.html", "dashboard.html", "exit.html", "admin_login.html",
    "admin_signup.html", "admin_dashboard.html", "admin_logs.html",
])
def test_every_post_form_has_csrf_token(template):
    with open(os.path.join(REPO_ROOT, "templates", template), encoding="utf-8") as f:
        html = f.read()
    forms = re.findall(r'<form[^>]*method="POST"[^>]*>(.*?)</form>', html, flags=re.S | re.I)
    assert forms, f"{template} has no POST forms"
    for form in forms:
        assert "csrf_token()" in form, f"{template} has a POST form without a CSRF token"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def test_login_is_rate_limited(client):
    statuses = [_login(client, "nobody", f"wrong-{i}").status_code for i in range(6)]
    assert statuses[:5] == [200] * 5
    assert statuses[5] == 429
    # GET requests are not counted against the login limit
    assert client.get("/admin_login").status_code == 200


def test_signup_is_rate_limited(client):
    statuses = [_signup(client, "wrong-passcode").status_code for _ in range(6)]
    assert statuses[:5] == [403] * 5
    assert statuses[5] == 429
