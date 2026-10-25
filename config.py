"""
Environment configuration and security validation for ParkEase.

All settings are resolved from a mapping (normally ``os.environ``) by
``load_settings`` so that the security rules can be unit-tested without
importing the Flask application.
"""

import hmac
import logging
import re
import secrets
from dataclasses import dataclass

from limits import parse_many

import database

logger = logging.getLogger("ParkEase.config")

VALID_APP_ENVS = ("development", "testing", "production")

# Values that have appeared in documentation, templates, tests, CI or source code.
# They are public knowledge and must never sign sessions.
KNOWN_INSECURE_SECRET_KEYS = frozenset({
    "your_secret_key",
    "your_secret_key_change_in_production",
    "replace_with_a_secure_random_key_in_production",
    "dev-insecure-secret-key-change-in-prod",
    "changeme",
    "secret",
    "test-secret-key",
    "test-only-secret-key-not-for-production-use",
    "ci-test-secret-key-non-empty",
})
MIN_PRODUCTION_SECRET_KEY_LENGTH = 32

# Passcodes that have been published in this repository.
KNOWN_INSECURE_SIGNUP_PASSCODES = frozenset({
    "ParkEase@123",
    "your_organization_signup_passcode_here",
})
MIN_SIGNUP_PASSCODE_LENGTH = 12

DEFAULT_DATABASE_PATH = "instance/parking.db"
DEFAULT_MODEL_PATH = "price_prediction_model.pkl"
DEFAULT_PARKING_CAPACITY = 100
MAX_PARKING_CAPACITY = 10000
DEFAULT_LOGIN_RATE_LIMIT = "5 per minute;20 per hour"
DEFAULT_SIGNUP_RATE_LIMIT = "5 per hour"
MAX_TRUSTED_PROXY_COUNT = 5

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}


class ConfigurationError(RuntimeError):
    """Raised when the environment configuration is unsafe or invalid."""


def parse_bool(value, name, default=False):
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ConfigurationError(f"{name} must be a boolean (true/false), got an unrecognized value.")


def running_under_pytest(environ):
    # pytest (>= 8.2) sets PYTEST_VERSION for the duration of a test session.
    return bool(environ.get("PYTEST_VERSION"))


def running_under_production_server(environ):
    # Gunicorn's arbiter exports SERVER_SOFTWARE=gunicorn/<version> before loading the app.
    return environ.get("SERVER_SOFTWARE", "").lower().startswith("gunicorn")


def resolve_app_env(environ):
    """
    Return the application environment. Production is the default.

    * Unset or empty APP_ENV means ``production``; development must be selected
      explicitly with APP_ENV=development. FLASK_ENV is ignored (Flask removed it,
      and it must not be able to select a weaker mode).
    * ``testing`` is accepted only inside a pytest session.
    * Under Gunicorn anything other than ``production`` is refused. There is no
      override: the former ALLOW_NON_PRODUCTION_SERVER flag is rejected if set.
    """
    if (environ.get("ALLOW_NON_PRODUCTION_SERVER") or "").strip():
        raise ConfigurationError(
            "ALLOW_NON_PRODUCTION_SERVER is no longer supported; remove it. Gunicorn always "
            "requires APP_ENV=production."
        )
    app_env = (environ.get("APP_ENV") or "production").strip().lower() or "production"
    if app_env not in VALID_APP_ENVS:
        raise ConfigurationError(
            f"APP_ENV must be one of {', '.join(VALID_APP_ENVS)}; got an unrecognized value."
        )
    if app_env == "testing" and not running_under_pytest(environ):
        raise ConfigurationError("APP_ENV=testing is only allowed while running the automated test suite.")
    if app_env != "production" and running_under_production_server(environ):
        raise ConfigurationError("Gunicorn requires APP_ENV=production.")
    if environ.get("FLASK_ENV") and not environ.get("APP_ENV"):
        logger.warning("FLASK_ENV is ignored; use APP_ENV. Running in production mode.")
    return app_env


def resolve_database_path(environ):
    db_path = environ.get("DATABASE_PATH") or DEFAULT_DATABASE_PATH
    try:
        database.assert_not_committed_baseline(db_path)
    except database.CommittedDatabaseError as e:
        raise ConfigurationError(str(e)) from None
    except OSError as e:
        raise ConfigurationError(
            f"DATABASE_PATH could not be verified ({type(e).__name__}); refusing to start."
        ) from None
    return db_path


def resolve_secret_key(app_env, configured_key):
    """
    Return ``(secret_key, was_generated)``.

    Production requires an explicit key that is not a published placeholder and is
    at least 32 characters long. Development and testing fall back to a random
    per-process key, so a missing key never results in a predictable one.
    """
    key = (configured_key or "").strip()
    insecure = not key or key in KNOWN_INSECURE_SECRET_KEYS

    if app_env == "production":
        if insecure:
            raise ConfigurationError(
                "FLASK_SECRET_KEY must be set to a unique, non-default value in production."
            )
        if len(key) < MIN_PRODUCTION_SECRET_KEY_LENGTH:
            raise ConfigurationError(
                f"FLASK_SECRET_KEY must be at least {MIN_PRODUCTION_SECRET_KEY_LENGTH} characters in production."
            )
        return key, False

    if insecure:
        logger.warning(
            "FLASK_SECRET_KEY is missing or uses a published placeholder; using a random per-process key. "
            "Sessions will not survive a restart."
        )
        return secrets.token_hex(32), True
    return key, False


def resolve_debug(app_env, flask_debug, argv=()):
    """
    Debug mode and the interactive debugger are allowed only with APP_ENV=development.

    ``flask --debug`` sets FLASK_DEBUG=1 before importing the app; ``flask run
    --debugger`` enables the Werkzeug debugger without touching FLASK_DEBUG, so the
    command line is checked too. Flask imports the app before starting its server
    (also when the reloader is used), so refusing here stops the command.
    """
    debug = parse_bool(flask_debug, "FLASK_DEBUG", default=False)
    debugger_flag = "--debugger" in argv
    if (debug or debugger_flag) and app_env != "development":
        raise ConfigurationError(
            "FLASK_DEBUG, --debug and --debugger are only allowed with APP_ENV=development."
        )
    return debug


def resolve_signup_passcode(app_env, configured_passcode):
    """
    Return the administrator signup passcode, or ``None`` when signup is disabled.

    Signup is disabled when no passcode is configured. A configured passcode that is
    published or too short is rejected in production and disabled elsewhere.
    """
    passcode = (configured_passcode or "").strip()
    if not passcode:
        return None
    if passcode in KNOWN_INSECURE_SIGNUP_PASSCODES or len(passcode) < MIN_SIGNUP_PASSCODE_LENGTH:
        message = (
            "ADMIN_SIGNUP_PASSCODE is a published default or shorter than "
            f"{MIN_SIGNUP_PASSCODE_LENGTH} characters."
        )
        if app_env == "production":
            raise ConfigurationError(message)
        logger.warning("%s Administrator signup is disabled.", message)
        return None
    return passcode


def passcode_matches(expected, supplied):
    """Constant-time passcode comparison. Always False when signup is disabled."""
    if not expected or supplied is None:
        return False
    return hmac.compare_digest(expected.encode("utf-8"), supplied.encode("utf-8"))


def parse_int(value, name, default, minimum, maximum):
    if value is None or value.strip() == "":
        return default
    if not re.fullmatch(r"[0-9]+", value.strip()):
        raise ConfigurationError(f"{name} must be a non-negative whole number.")
    try:
        parsed = int(value)
    except ValueError:
        raise ConfigurationError(f"{name} must be an integer.") from None
    if not minimum <= parsed <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}.")
    return parsed


def parse_rate_limit(value, name, default):
    """
    Validate a Flask-Limiter limit string such as "5 per minute;20 per hour".

    Unset or blank uses the secure default. Anything else must parse completely and
    every limit must allow at least one request; otherwise startup is refused, because
    Flask-Limiter would silently skip an unparseable limit and leave the route unprotected.
    """
    if value is None or value.strip() == "":
        return default
    try:
        limits = parse_many(value)
    except ValueError:
        raise ConfigurationError(
            f"{name} is not a valid rate limit (expected e.g. '5 per minute;20 per hour')."
        ) from None
    if not limits or any(limit.amount < 1 for limit in limits):
        raise ConfigurationError(f"{name} must allow at least 1 request per period.")
    return value.strip()


@dataclass(frozen=True)
class Settings:
    app_env: str
    secret_key: str
    secret_key_generated: bool
    debug: bool
    database_path: str
    model_path: str
    admin_signup_passcode: str | None
    parking_capacity: int
    login_rate_limit: str
    signup_rate_limit: str
    ratelimit_storage_uri: str
    session_cookie_secure: bool
    trusted_proxy_count: int


def load_settings(environ, argv=()):
    """Resolve and validate all settings. ``argv`` is the process command line (for --debugger)."""
    app_env = resolve_app_env(environ)
    secret_key, generated = resolve_secret_key(app_env, environ.get("FLASK_SECRET_KEY"))
    return Settings(
        app_env=app_env,
        secret_key=secret_key,
        secret_key_generated=generated,
        debug=resolve_debug(app_env, environ.get("FLASK_DEBUG"), argv),
        database_path=resolve_database_path(environ),
        model_path=environ.get("MODEL_PATH") or DEFAULT_MODEL_PATH,
        admin_signup_passcode=resolve_signup_passcode(app_env, environ.get("ADMIN_SIGNUP_PASSCODE")),
        parking_capacity=parse_int(
            environ.get("PARKING_CAPACITY"), "PARKING_CAPACITY",
            DEFAULT_PARKING_CAPACITY, 1, MAX_PARKING_CAPACITY,
        ),
        login_rate_limit=parse_rate_limit(
            environ.get("LOGIN_RATE_LIMIT"), "LOGIN_RATE_LIMIT", DEFAULT_LOGIN_RATE_LIMIT,
        ),
        signup_rate_limit=parse_rate_limit(
            environ.get("SIGNUP_RATE_LIMIT"), "SIGNUP_RATE_LIMIT", DEFAULT_SIGNUP_RATE_LIMIT,
        ),
        ratelimit_storage_uri=environ.get("RATELIMIT_STORAGE_URI") or "memory://",
        session_cookie_secure=parse_bool(
            environ.get("SESSION_COOKIE_SECURE"), "SESSION_COOKIE_SECURE",
            default=(app_env == "production"),
        ),
        # Must equal the number of reverse proxies that overwrite or append X-Forwarded-For;
        # a higher value lets clients spoof their IP and evade rate limits.
        trusted_proxy_count=parse_int(
            environ.get("TRUSTED_PROXY_COUNT"), "TRUSTED_PROXY_COUNT", 0, 0, MAX_TRUSTED_PROXY_COUNT,
        ),
    )
