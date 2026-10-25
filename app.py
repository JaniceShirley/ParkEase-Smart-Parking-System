import os
import re
import struct
import sqlite3
import sys
import logging
from contextlib import closing
from datetime import datetime

import click
from dotenv import load_dotenv
from flask import Flask, request, render_template, session, redirect, url_for, flash
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect, CSRFError
import joblib
import pandas as pd
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash

import config
import database
from database import SLOT_FREE, SLOT_OCCUPIED
from weatherAPI import TrichyWeatherClassifier

# Load environment configuration
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ParkEase")

# Validates APP_ENV (production unless explicitly development), FLASK_SECRET_KEY,
# FLASK_DEBUG / --debugger, ADMIN_SIGNUP_PASSCODE, rate limits, etc.
# Raises config.ConfigurationError (a RuntimeError) on unsafe settings.
settings = config.load_settings(os.environ, sys.argv)

app = Flask(__name__)
app.secret_key = settings.secret_key

app.config.update(
    # Flask derives DEBUG from FLASK_DEBUG on its own; pin it to the validated value.
    DEBUG=settings.debug,
    APP_ENV=settings.app_env,
    DATABASE_PATH=settings.database_path,
    MODEL_PATH=settings.model_path,
    ADMIN_SIGNUP_PASSCODE=settings.admin_signup_passcode,
    PARKING_CAPACITY=settings.parking_capacity,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=settings.session_cookie_secure,
    # Tokens are bound to the signed session; a time limit would break checkout
    # for drivers whose dashboard has been open for longer than the limit.
    WTF_CSRF_TIME_LIMIT=None,
)

if settings.trusted_proxy_count:
    logger.info(
        "Trusting X-Forwarded-For/-Proto from %d reverse proxy hop(s). This must match the deployment.",
        settings.trusted_proxy_count,
    )
    app.wsgi_app = ProxyFix(
        app.wsgi_app,
        x_for=settings.trusted_proxy_count,
        x_proto=settings.trusted_proxy_count,
    )

csrf = CSRFProtect(app)
limiter = Limiter(get_remote_address, app=app, storage_uri=settings.ratelimit_storage_uri)

VALID_VEHICLE_TYPES = ('auto', 'car', '2-wheeler', 'suv/van')
VALID_ELECTRIC_VALUES = ('yes', 'no')
MAX_NAME_LENGTH = 100
PHONE_PATTERN = re.compile(r"^\+?[0-9]{7,15}$")
VEHICLE_NUMBER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9 -]{0,19}$")
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,50}$")
MIN_ADMIN_PASSWORD_LENGTH = 10
PASSWORD_HASH_PREFIXES = ('pbkdf2:', 'scrypt:')

# Used so a login for an unknown or unhashed account costs the same as a real check.
_DUMMY_PASSWORD_HASH = generate_password_hash("parkease-timing-equalizer")

ACTIVE_USER_COLUMNS = (
    "id, name, phone, vehicle_number, vehicle_type, electric, time, slot_id, "
    "exit_time, duration_min, price, weather, congestion"
)
LOG_COLUMNS = (
    "id, name, phone, vehicle_number, vehicle_type, electric, entry_time, exit_time, "
    "slot_id, duration_min, price, weather, congestion"
)


def get_db_path():
    """Resolve database path dynamically, prioritizing app.config for testing isolation."""
    return app.config.get("DATABASE_PATH") or settings.database_path


def db_connect():
    conn = database.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    return conn


# Initialize weather classifier
classifier = TrichyWeatherClassifier()

# Model Caching: Load model once at startup, reload dynamically if target path changes
LOADED_MODEL = None
LOADED_MODEL_PATH = None

def get_model(model_path=None):
    """
    Load and cache ML pricing model.
    Prioritizes explicit argument, then app.config (populated from MODEL_PATH).
    """
    global LOADED_MODEL, LOADED_MODEL_PATH
    target_path = model_path or app.config.get("MODEL_PATH") or settings.model_path
    if LOADED_MODEL is None or LOADED_MODEL_PATH != target_path:
        if os.path.exists(target_path):
            try:
                LOADED_MODEL = joblib.load(target_path)
                LOADED_MODEL_PATH = target_path
                logger.info("ML price prediction model loaded successfully from %s", target_path)
            except Exception as e:
                logger.error("Failed to load model from %s: %s", target_path, e)
                raise
        else:
            logger.error("Model artifact not found at %s", target_path)
            raise FileNotFoundError(f"Model file not found at {target_path}")
    return LOADED_MODEL

# Attempt startup model load with default path
try:
    get_model()
except Exception as e:
    logger.warning("Model could not be loaded at process startup (%s). Price estimation will fail until resolved.", e)

def init_db(db_path=None):
    database.init_db(db_path or get_db_path(), app.config["PARKING_CAPACITY"])

# Ensure schema exists on module load (supports WSGI / Gunicorn)
try:
    init_db()
except Exception as e:
    logger.warning("Database schema verification notice: %s", e)


def validate_checkin_form(form):
    """Return (cleaned_data, errors) for the driver check-in form."""
    errors = []
    name = form.get('name', '').strip()
    phone = re.sub(r"[\s-]", "", form.get('phone', ''))
    vehicle_number = re.sub(r"\s+", " ", form.get('vehicle_number', '').strip().upper())
    vehicle_type = form.get('vehicle_type', '')
    electric = form.get('electric', '')

    if not name or len(name) > MAX_NAME_LENGTH:
        errors.append(f'Name is required and must be at most {MAX_NAME_LENGTH} characters.')
    if not PHONE_PATTERN.match(phone):
        errors.append('Phone number must contain 7 to 15 digits.')
    if not VEHICLE_NUMBER_PATTERN.match(vehicle_number):
        errors.append('Vehicle number must be 1 to 20 letters, digits, spaces or hyphens.')
    if vehicle_type not in VALID_VEHICLE_TYPES:
        errors.append('Invalid vehicle type selected.')
    if electric not in VALID_ELECTRIC_VALUES:
        errors.append('Please choose whether the vehicle is electric.')

    data = {
        'name': name,
        'phone': phone,
        'vehicle_number': vehicle_number,
        'vehicle_type': vehicle_type,
        'electric': electric,
    }
    return data, errors


def validate_admin_fields(username, password, mobile):
    errors = []
    if not USERNAME_PATTERN.match(username):
        errors.append('Username must be 3 to 50 letters, digits, dots, hyphens or underscores.')
    if len(password) < MIN_ADMIN_PASSWORD_LENGTH:
        errors.append(f'Password must be at least {MIN_ADMIN_PASSWORD_LENGTH} characters.')
    if not PHONE_PATTERN.match(re.sub(r"[\s-]", "", mobile)):
        errors.append('Mobile number must contain 7 to 15 digits.')
    return errors


def is_password_hash(stored_password):
    return isinstance(stored_password, str) and stored_password.startswith(PASSWORD_HASH_PREFIXES)


def fetch_active_user(conn, user_id):
    return conn.execute(f'SELECT {ACTIVE_USER_COLUMNS} FROM users WHERE id=?', (user_id,)).fetchone()


def end_stale_driver_session():
    """The session points at a user row that no longer exists; reset it instead of looping."""
    session.pop('user_id', None)
    session.pop('exit_data', None)
    flash('Your parking session could not be found. Please check in again or contact the attendant.', 'error')
    return redirect(url_for('index'))


def slot_counts(conn):
    total, occupied = conn.execute(
        'SELECT COUNT(*), COALESCE(SUM(CASE WHEN status=? THEN 1 ELSE 0 END), 0) FROM parking_slots',
        (SLOT_OCCUPIED,),
    ).fetchone()
    return total, occupied


def decode_legacy_number(value):
    """
    Older versions stored NumPy scalars, which SQLite saved as raw bytes
    (float32 → 4 bytes, float64 → 8 bytes). Decode them back to floats.
    """
    if isinstance(value, (bytes, bytearray)):
        if len(value) == 4:
            return float(struct.unpack('<f', value)[0])
        if len(value) == 8:
            return float(struct.unpack('<d', value)[0])
        return None
    return value


@app.errorhandler(CSRFError)
def handle_csrf_error(error):
    logger.warning("Rejected request with missing or invalid CSRF token: %s", error.description)
    return render_template(
        'error.html',
        title='Form expired',
        message='Your form session has expired or is invalid. Please go back, reload the page and try again.',
    ), 400


@app.errorhandler(429)
def handle_rate_limit(error):
    return render_template(
        'error.html',
        title='Too many attempts',
        message='Too many attempts. Please wait a few minutes and try again.',
    ), 429


@app.route('/', methods=['GET', 'POST'])
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        data, errors = validate_checkin_form(request.form)
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('index.html', form=request.form), 400

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Atomic slot reservation and user creation within a single transaction
        with closing(db_connect()) as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute("""
                    UPDATE parking_slots
                    SET status = ?
                    WHERE slot_id = (
                        SELECT slot_id FROM parking_slots
                        WHERE status = ?
                        ORDER BY slot_id ASC LIMIT 1
                    )
                    RETURNING slot_id
                """, (SLOT_OCCUPIED, SLOT_FREE)).fetchone()
                if not row:
                    conn.rollback()
                    flash('No parking slots are available right now.', 'error')
                    return render_template('index.html', form=request.form), 400

                slot_id = row[0]
                cursor = conn.execute('''INSERT INTO users (name, phone, vehicle_number, vehicle_type, electric, time, slot_id)
                                         VALUES (?, ?, ?, ?, ?, ?, ?)''',
                                      (data['name'], data['phone'], data['vehicle_number'], data['vehicle_type'],
                                       data['electric'], timestamp, slot_id))
                user_id = cursor.lastrowid
                conn.commit()
            except sqlite3.Error:
                conn.rollback()
                logger.exception("Error during atomic check-in transaction")
                flash('We could not reserve a slot. Please try again.', 'error')
                return render_template('index.html', form=request.form), 503

        session['user_id'] = user_id
        return redirect(url_for('dashboard'))

    return render_template('index.html', form={})

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('index'))

    with closing(db_connect()) as conn:
        user = fetch_active_user(conn, session['user_id'])

    if user is None:
        return end_stale_driver_session()
    if user['exit_time'] is not None:
        # Checkout already quoted; the fee is fixed until payment.
        return redirect(url_for('exit_receipt'))

    display_user = (
        user['name'], user['phone'], user['vehicle_number'],
        user['vehicle_type'].title(), user['electric'].title(),
        user['time'], user['slot_id'],
    )
    return render_template('dashboard.html', user=display_user)

@app.route('/exit', methods=['POST'])
def exit_parking():
    if 'user_id' not in session:
        return redirect(url_for('index'))

    with closing(db_connect()) as conn:
        user = fetch_active_user(conn, session['user_id'])
        if user is not None:
            total_slots, occupied = slot_counts(conn)

    if user is None:
        return end_stale_driver_session()
    if user['exit_time'] is not None:
        # The fee was already calculated; never recalculate it.
        return redirect(url_for('exit_receipt'))

    start_time = datetime.strptime(user['time'], '%Y-%m-%d %H:%M:%S')
    exit_time = datetime.now()
    duration_min = max(0.0, round((exit_time - start_time).total_seconds() / 60, 1))
    weather = classifier.classify_weather()
    congestion = round(occupied / total_slots, 4) if total_slots else 0.0

    # Format input for cached ML model
    input_df = pd.DataFrame([[duration_min, user['vehicle_type'],
                            "electric" if user['electric'] == "yes" else "non-electric",
                            weather, congestion]],
                          columns=["Duration (min)", "Vehicle Type", "Electric", "Weather", "Congestion Level"])
    try:
        model = get_model()
        raw_price = float(model.predict(input_df)[0])
        # Enforce business rule: non-negative parking fee boundary
        price = max(0.0, round(raw_price, 2))
    except Exception as e:
        logger.error("Price estimation error during checkout: %s", e)
        flash("Dynamic pricing service temporarily unavailable.", "error")
        return redirect(url_for('dashboard'))

    # Persist the quote once. The exit_time IS NULL guard means a concurrent or
    # repeated request cannot replace an existing quote with a cheaper one.
    with closing(db_connect()) as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute('''UPDATE users
                            SET exit_time=?, duration_min=?, price=?, weather=?, congestion=?
                            WHERE id=? AND exit_time IS NULL''',
                         (exit_time.strftime('%Y-%m-%d %H:%M:%S'), duration_min, price,
                          weather, congestion, user['id']))
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            logger.exception("Error while saving checkout quote")
            flash("We could not complete checkout. Please try again.", "error")
            return redirect(url_for('dashboard'))

    return redirect(url_for('exit_receipt'))

@app.route('/exit_receipt')
def exit_receipt():
    if 'user_id' not in session:
        return redirect(url_for('index'))

    with closing(db_connect()) as conn:
        user = fetch_active_user(conn, session['user_id'])

    if user is None:
        return end_stale_driver_session()
    if user['exit_time'] is None:
        return redirect(url_for('dashboard'))

    return render_template(
        'exit.html',
        name=user['name'],
        vehicle_number=user['vehicle_number'],
        entry_time=user['time'],
        exit_time=user['exit_time'],
        duration_min=user['duration_min'],
        price=user['price'],
        slot_id=user['slot_id'],
        weather=user['weather'],
        congestion=user['congestion'],
    )

@app.route('/process_payment', methods=['POST'])
def process_payment():
    if 'user_id' not in session:
        return redirect(url_for('index'))

    user_id = session['user_id']
    with closing(db_connect()) as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            user = fetch_active_user(conn, user_id)
            if user is None:
                conn.rollback()
                return end_stale_driver_session()
            if user['exit_time'] is None:
                conn.rollback()
                flash('Please request checkout before confirming payment.', 'error')
                return redirect(url_for('dashboard'))

            # Log the parking session using the stored quote, never client-supplied values
            conn.execute('''INSERT INTO parking_logs
                            (name, phone, vehicle_number, vehicle_type, electric,
                             entry_time, exit_time, slot_id, duration_min, price, weather, congestion)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                         (user['name'], user['phone'], user['vehicle_number'], user['vehicle_type'],
                          user['electric'], user['time'], user['exit_time'], user['slot_id'],
                          user['duration_min'], user['price'], user['weather'], user['congestion']))
            conn.execute('UPDATE parking_slots SET status=? WHERE slot_id=?', (SLOT_FREE, user['slot_id']))
            conn.execute('DELETE FROM users WHERE id=?', (user_id,))
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            logger.exception("Error during payment processing transaction")
            flash('Payment could not be recorded. Please try again.', 'error')
            return redirect(url_for('exit_receipt'))

    # Clear session data
    session.pop('user_id', None)
    session.pop('exit_data', None)
    flash('Payment confirmed. Thank you for using ParkEase!', 'success')
    return redirect(url_for('index'))

@app.route('/admin_signup', methods=['GET', 'POST'])
@limiter.limit(settings.signup_rate_limit, methods=['POST'])
def admin_signup():
    expected_passcode = app.config.get("ADMIN_SIGNUP_PASSCODE")
    signup_enabled = bool(expected_passcode)

    if request.method == 'POST':
        if not signup_enabled:
            flash('Administrator registration is disabled.', 'error')
            return render_template('admin_signup.html', signup_enabled=False), 403

        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        mobile = request.form.get('mobile', '').strip()
        org_password = request.form.get('org_password', '')

        if not config.passcode_matches(expected_passcode, org_password):
            flash('Invalid organization passcode.', 'error')
            return render_template('admin_signup.html', signup_enabled=True), 403

        errors = validate_admin_fields(username, password, mobile)
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('admin_signup.html', signup_enabled=True), 400

        hashed_password = generate_password_hash(password)
        with closing(db_connect()) as conn:
            try:
                conn.execute('INSERT INTO admin (username, password, mobile) VALUES (?, ?, ?)',
                             (username, hashed_password, mobile))
                conn.commit()
            except sqlite3.IntegrityError:
                flash('Username already exists.', 'error')
                return render_template('admin_signup.html', signup_enabled=True), 409
        flash('Administrator account created. Please log in.', 'success')
        return redirect(url_for('admin_login'))
    return render_template('admin_signup.html', signup_enabled=signup_enabled)

@app.route('/admin_login', methods=['GET', 'POST'])
@limiter.limit(settings.login_rate_limit, methods=['POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        with closing(db_connect()) as conn:
            admin = conn.execute('SELECT id, password FROM admin WHERE username=?', (username,)).fetchone()

        authenticated = False
        if admin is not None and is_password_hash(admin['password']):
            authenticated = check_password_hash(admin['password'], password)
        else:
            # Unknown user or unhashed legacy password: spend the same time, then reject.
            check_password_hash(_DUMMY_PASSWORD_HASH, password)
            if admin is not None:
                logger.warning(
                    "Admin account id=%s has an unhashed legacy password and cannot log in. "
                    "Reset it with: flask --app app reset-admin-password <username>",
                    admin['id'],
                )

        if authenticated:
            session['admin_id'] = admin['id']
            return redirect(url_for('admin_dashboard'))

        flash('Invalid credentials', 'error')
    return render_template('admin_login.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))

    with closing(db_connect()) as conn:
        # Get all parking slots
        slot_rows = conn.execute('SELECT slot_id, status FROM parking_slots ORDER BY slot_id').fetchall()
        slots = [{'slot_id': row['slot_id'], 'status': row['status'], 'user_details': None} for row in slot_rows]

        # Active users, in the original column order used by the template
        users = conn.execute(
            'SELECT id, name, phone, vehicle_number, vehicle_type, electric, time, slot_id FROM users'
        ).fetchall()
        total_slots, occupied_count = slot_counts(conn)

    # Create a mapping of slot_id to user details
    user_map = {user['slot_id']: user for user in users}

    # Update slots with user details if occupied
    for slot in slots:
        if slot['status'] == SLOT_OCCUPIED and slot['slot_id'] in user_map:
            slot['user_details'] = user_map[slot['slot_id']]

    free_count = total_slots - occupied_count
    occupancy_rate = round((occupied_count / total_slots) * 100, 1) if total_slots else 0.0

    return render_template('admin_dashboard.html',
                         slots=slots,
                         total_slots=total_slots,
                         occupied_count=occupied_count,
                         free_count=free_count,
                         occupancy_rate=occupancy_rate)

@app.route('/admin_logs')
def admin_logs():
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))

    with closing(db_connect()) as conn:
        rows = conn.execute(f'SELECT {LOG_COLUMNS} FROM parking_logs ORDER BY exit_time DESC').fetchall()

    logs = []
    for row in rows:
        log = list(row)
        for index in (9, 10, 12):  # duration_min, price, congestion
            log[index] = decode_legacy_number(log[index])
        logs.append(log)

    return render_template('admin_logs.html', logs=logs)

@app.route('/admin_logout', methods=['POST'])
def admin_logout():
    session.pop('admin_id', None)
    return redirect(url_for('admin_login'))


@app.cli.command('init-db')
def init_db_command():
    """Create or migrate the database at DATABASE_PATH."""
    init_db()
    click.echo(f"Database ready at {get_db_path()}")


@app.cli.command('create-admin')
@click.option('--username', prompt=True)
@click.option('--mobile', prompt=True)
@click.password_option()
def create_admin_command(username, mobile, password):
    """Create an administrator account with a hashed password."""
    username = username.strip()
    errors = validate_admin_fields(username, password, mobile)
    if errors:
        raise click.ClickException(" ".join(errors))
    init_db()
    with closing(db_connect()) as conn:
        try:
            conn.execute('INSERT INTO admin (username, password, mobile) VALUES (?, ?, ?)',
                         (username, generate_password_hash(password), mobile.strip()))
            conn.commit()
        except sqlite3.IntegrityError:
            raise click.ClickException("Username already exists.") from None
    click.echo(f"Administrator '{username}' created.")


@app.cli.command('reset-admin-password')
@click.argument('username')
@click.password_option()
def reset_admin_password_command(username, password):
    """Set a new hashed password for an existing administrator."""
    if len(password) < MIN_ADMIN_PASSWORD_LENGTH:
        raise click.ClickException(f"Password must be at least {MIN_ADMIN_PASSWORD_LENGTH} characters.")
    with closing(db_connect()) as conn:
        cursor = conn.execute('UPDATE admin SET password=? WHERE username=?',
                              (generate_password_hash(password), username))
        conn.commit()
    if cursor.rowcount == 0:
        raise click.ClickException(f"No administrator named '{username}'.")
    click.echo(f"Password updated for '{username}'.")


if __name__ == '__main__':
    # Debug mode requires APP_ENV=development and FLASK_DEBUG=true (enforced by config.py).
    app.run(debug=settings.debug)
