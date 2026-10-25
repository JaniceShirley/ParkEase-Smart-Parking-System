"""
Production-by-default configuration, debugger restrictions, rate-limit validation
and proxy-trust validation.

Process-level tests run against a copy of the application code (with a fictional
parking.db as its protected legacy file), never against the real repository files.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

import config
from tests.helpers import make_fictional_legacy_db

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRONG_TEST_KEY = "p" * 48
BIN_DIR = os.path.dirname(sys.executable)


# ---------------------------------------------------------------------------
# APP_ENV defaults
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("environ", [{}, {"APP_ENV": ""}, {"APP_ENV": "   "}, {"FLASK_ENV": "development"}])
def test_missing_app_env_means_production(environ):
    assert config.resolve_app_env(environ) == "production"
    with pytest.raises(config.ConfigurationError, match="FLASK_SECRET_KEY"):
        config.load_settings(environ)


def test_default_production_settings_are_strict():
    settings = config.load_settings({"FLASK_SECRET_KEY": STRONG_TEST_KEY})
    assert settings.app_env == "production"
    assert settings.debug is False
    assert settings.session_cookie_secure is True
    assert settings.secret_key_generated is False
    assert settings.admin_signup_passcode is None


def test_default_production_still_rejects_weak_signup_passcode():
    with pytest.raises(config.ConfigurationError, match="ADMIN_SIGNUP_PASSCODE"):
        config.load_settings({"FLASK_SECRET_KEY": STRONG_TEST_KEY, "ADMIN_SIGNUP_PASSCODE": "ParkEase@123"})


def test_explicit_development():
    settings = config.load_settings({"APP_ENV": "development"})
    assert settings.app_env == "development"
    assert settings.secret_key_generated is True
    assert settings.session_cookie_secure is False
    assert settings.debug is False


def test_explicit_production():
    settings = config.load_settings({"APP_ENV": "PRODUCTION", "FLASK_SECRET_KEY": STRONG_TEST_KEY})
    assert settings.app_env == "production"


# ---------------------------------------------------------------------------
# Debug flags
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("environ", [
    {"FLASK_SECRET_KEY": STRONG_TEST_KEY},
    {"APP_ENV": "production", "FLASK_SECRET_KEY": STRONG_TEST_KEY},
    {"APP_ENV": "testing", "PYTEST_VERSION": "8.3.4"},
])
@pytest.mark.parametrize("flask_debug,argv", [
    ("1", ()),
    ("true", ()),
    (None, ("flask", "run", "--debugger")),
])
def test_debugger_refused_outside_development(environ, flask_debug, argv):
    env = dict(environ)
    if flask_debug is not None:
        env["FLASK_DEBUG"] = flask_debug
    with pytest.raises(config.ConfigurationError, match="only allowed with APP_ENV=development"):
        config.load_settings(env, argv)


def test_debugger_allowed_in_explicit_development():
    assert config.load_settings({"APP_ENV": "development", "FLASK_DEBUG": "1"}).debug is True
    assert config.load_settings({"APP_ENV": "development"}, ("flask", "run", "--debugger")).app_env == "development"


def test_debug_off_values_are_accepted_everywhere():
    for value in ("0", "false", ""):
        assert config.load_settings({"FLASK_SECRET_KEY": STRONG_TEST_KEY, "FLASK_DEBUG": value}).debug is False


# ---------------------------------------------------------------------------
# Rate-limit validation
# ---------------------------------------------------------------------------

DEV = {"APP_ENV": "development"}


@pytest.mark.parametrize("value", [None, "", "   "])
def test_rate_limits_default_when_unset(value):
    env = dict(DEV)
    if value is not None:
        env["LOGIN_RATE_LIMIT"] = value
        env["SIGNUP_RATE_LIMIT"] = value
    settings = config.load_settings(env)
    assert settings.login_rate_limit == config.DEFAULT_LOGIN_RATE_LIMIT
    assert settings.signup_rate_limit == config.DEFAULT_SIGNUP_RATE_LIMIT


@pytest.mark.parametrize("value", ["3/minute", "10 per hour; 2 per minute", "1 per day"])
def test_valid_rate_limits_accepted(value):
    settings = config.load_settings({**DEV, "LOGIN_RATE_LIMIT": value, "SIGNUP_RATE_LIMIT": value})
    assert settings.login_rate_limit == value.strip()
    assert settings.signup_rate_limit == value.strip()


@pytest.mark.parametrize("variable", ["LOGIN_RATE_LIMIT", "SIGNUP_RATE_LIMIT"])
@pytest.mark.parametrize("value", [
    "not-a-limit", "5 per fortnight", "5 per minute;garbage", "per minute",
    "0 per minute", "-1 per minute", "5", "unlimited",
])
def test_invalid_rate_limits_rejected(variable, value):
    with pytest.raises(config.ConfigurationError, match=variable):
        config.load_settings({**DEV, variable: value})


# ---------------------------------------------------------------------------
# TRUSTED_PROXY_COUNT validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [(None, 0), ("", 0), ("0", 0), ("1", 1), (" 2 ", 2), ("5", 5)])
def test_trusted_proxy_count_valid(value, expected):
    env = dict(DEV)
    if value is not None:
        env["TRUSTED_PROXY_COUNT"] = value
    assert config.load_settings(env).trusted_proxy_count == expected


@pytest.mark.parametrize("value", ["-1", "abc", "1.5", "+1", "true", "6", "0x1", "1e1"])
def test_trusted_proxy_count_invalid(value):
    with pytest.raises(config.ConfigurationError, match="TRUSTED_PROXY_COUNT"):
        config.load_settings({**DEV, "TRUSTED_PROXY_COUNT": value})


# ---------------------------------------------------------------------------
# Process-level behaviour (repository copy)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def repo_copy(tmp_path_factory):
    root = tmp_path_factory.mktemp("prod_defaults_copy")
    for name in ("app.py", "config.py", "database.py", "weatherAPI.py"):
        shutil.copyfile(os.path.join(REPO_ROOT, name), root / name)
    for name in ("templates", "static"):
        shutil.copytree(os.path.join(REPO_ROOT, name), root / name)
    make_fictional_legacy_db(root / "parking.db")
    return root


def base_env(**overrides):
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        # Present-but-empty so python-dotenv cannot fill them from any .env
        "APP_ENV": "",
        "FLASK_SECRET_KEY": "",
        "FLASK_DEBUG": "",
        "ADMIN_SIGNUP_PASSCODE": "",
        "OPENWEATHER_API_KEY": "",
        "LOGIN_RATE_LIMIT": "",
        "SIGNUP_RATE_LIMIT": "",
        "TRUSTED_PROXY_COUNT": "",
        "DATABASE_PATH": "instance/runtime.db",
    }
    env.update(overrides)
    return env


def run(repo_copy, args, timeout=60, **overrides):
    try:
        return subprocess.run(args, cwd=repo_copy, env=base_env(**overrides),
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{args[0]} kept running instead of refusing to start")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def nothing_listening(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def test_import_without_app_env_requires_production_secret(repo_copy):
    result = run(repo_copy, [sys.executable, "-c", "import app"])
    assert result.returncode != 0
    assert "FLASK_SECRET_KEY must be set" in result.stderr


def test_flask_debug_cli_refused_without_explicit_development(repo_copy):
    flask = os.path.join(BIN_DIR, "flask")
    result = run(repo_copy, [flask, "--app", "app", "--debug", "routes"], FLASK_SECRET_KEY=STRONG_TEST_KEY)
    assert result.returncode != 0
    assert "only allowed with APP_ENV=development" in result.stderr


@pytest.mark.parametrize("extra", [["--debugger", "--no-reload"], ["--debugger", "--reload"]],
                         ids=["debugger", "debugger+reloader"])
def test_flask_run_debugger_refused_without_explicit_development(repo_copy, extra):
    flask = os.path.join(BIN_DIR, "flask")
    port = free_port()
    result = run(repo_copy, [flask, "--app", "app", "run", "--port", str(port)] + extra,
                 FLASK_SECRET_KEY=STRONG_TEST_KEY)
    assert result.returncode != 0
    assert "only allowed with APP_ENV=development" in result.stderr
    assert nothing_listening(port)


def test_flask_debug_run_with_reloader_refused(repo_copy):
    flask = os.path.join(BIN_DIR, "flask")
    port = free_port()
    result = run(repo_copy, [flask, "--app", "app", "--debug", "run", "--port", str(port)],
                 FLASK_SECRET_KEY=STRONG_TEST_KEY)
    assert result.returncode != 0
    assert nothing_listening(port)


def test_flask_debug_allowed_in_explicit_development(repo_copy):
    flask = os.path.join(BIN_DIR, "flask")
    result = run(repo_copy, [flask, "--app", "app", "--debug", "routes"], APP_ENV="development")
    assert result.returncode == 0, result.stderr[-1500:]
    assert "admin_login" in result.stdout


@pytest.mark.parametrize("preload", [True, False], ids=["preload", "no-preload"])
def test_gunicorn_without_app_env_requires_secret(repo_copy, preload):
    gunicorn = os.path.join(BIN_DIR, "gunicorn")
    if not os.path.exists(gunicorn):
        pytest.skip("gunicorn is not installed next to this interpreter")
    args = [gunicorn] + (["--preload"] if preload else []) + ["-w", "1", "-b", "127.0.0.1:0", "app:app"]
    result = run(repo_copy, args)
    assert result.returncode != 0
    assert "FLASK_SECRET_KEY must be set" in (result.stdout + result.stderr)


@pytest.mark.parametrize("preload", [True, False], ids=["preload", "no-preload"])
def test_gunicorn_without_app_env_serves_strict_production(repo_copy, preload):
    gunicorn = os.path.join(BIN_DIR, "gunicorn")
    if not os.path.exists(gunicorn):
        pytest.skip("gunicorn is not installed next to this interpreter")
    port = free_port()
    args = [gunicorn] + (["--preload"] if preload else []) + ["-w", "1", "-b", f"127.0.0.1:{port}", "app:app"]
    proc = subprocess.Popen(args, cwd=repo_copy, env=base_env(FLASK_SECRET_KEY=STRONG_TEST_KEY),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        response = None
        for _ in range(120):
            try:
                response = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2)
                break
            except OSError:
                if proc.poll() is not None:
                    pytest.fail("gunicorn exited instead of serving a valid production configuration")
                time.sleep(0.25)
        assert response is not None and response.status == 200
        cookie = response.headers.get("Set-Cookie", "")
        assert "Secure" in cookie and "HttpOnly" in cookie
    finally:
        proc.terminate()
        proc.wait(timeout=30)


RATE_LIMIT_SCRIPT = """
import json, sys
import app as m
m.app.config["WTF_CSRF_ENABLED"] = False
client = m.app.test_client()
statuses = []
for spoofed in json.loads(sys.argv[1]):
    headers = {"X-Forwarded-For": spoofed} if spoofed else {}
    statuses.append(client.post("/admin_login", data={"username": "nobody", "password": "x"},
                                headers=headers).status_code)
print(json.dumps(statuses))
"""


def rate_limit_statuses(repo_copy, forwarded_for, **overrides):
    result = run(repo_copy, [sys.executable, "-c", RATE_LIMIT_SCRIPT, json.dumps(forwarded_for)],
                 APP_ENV="development", **overrides)
    assert result.returncode == 0, result.stderr[-1500:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_custom_login_rate_limit_is_enforced(repo_copy):
    assert rate_limit_statuses(repo_copy, [None] * 3, LOGIN_RATE_LIMIT="2 per minute") == [200, 200, 429]


def test_default_login_rate_limit_is_enforced(repo_copy):
    assert rate_limit_statuses(repo_copy, [None] * 6) == [200] * 5 + [429]


@pytest.mark.parametrize("variable", ["LOGIN_RATE_LIMIT", "SIGNUP_RATE_LIMIT"])
def test_invalid_rate_limit_stops_startup(repo_copy, variable):
    result = run(repo_copy, [sys.executable, "-c", "import app"], APP_ENV="development",
                 **{variable: "not-a-limit"})
    assert result.returncode != 0
    assert f"{variable} is not a valid rate limit" in result.stderr


def test_invalid_proxy_count_stops_startup(repo_copy):
    result = run(repo_copy, [sys.executable, "-c", "import app"], APP_ENV="development",
                 TRUSTED_PROXY_COUNT="-1")
    assert result.returncode != 0
    assert "TRUSTED_PROXY_COUNT" in result.stderr


def test_without_proxy_trust_spoofed_headers_do_not_evade_limit(repo_copy):
    spoofed = [f"203.0.113.{i}" for i in range(6)]
    assert rate_limit_statuses(repo_copy, spoofed, LOGIN_RATE_LIMIT="2 per minute") == [200, 200, 429, 429, 429, 429]


def test_with_one_trusted_proxy_limit_applies_per_forwarded_client(repo_copy):
    # As seen behind one proxy that appends the real client address.
    clients = ["198.51.100.1", "198.51.100.1", "198.51.100.1", "198.51.100.2"]
    statuses = rate_limit_statuses(repo_copy, clients, LOGIN_RATE_LIMIT="2 per minute", TRUSTED_PROXY_COUNT="1")
    assert statuses == [200, 200, 429, 200]
