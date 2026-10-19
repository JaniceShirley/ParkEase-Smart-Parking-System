import os
import sqlite3
import logging
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, render_template, session, redirect, url_for, flash
import joblib
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from weatherAPI import TrichyWeatherClassifier

# Load environment configuration
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ParkEase")

app = Flask(__name__)

# Application Environment Configuration
APP_ENV = os.environ.get("APP_ENV", os.environ.get("FLASK_ENV", "development")).lower()
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY")

if APP_ENV == "production":
    if not FLASK_SECRET_KEY or FLASK_SECRET_KEY in [
        "your_secret_key",
        "replace_with_a_secure_random_key_in_production",
        "your_secret_key_change_in_production",
        ""
    ]:
        raise RuntimeError("CRITICAL: FLASK_SECRET_KEY must be set to a secure, non-default value in production.")
else:
    if not FLASK_SECRET_KEY or FLASK_SECRET_KEY in [
        "your_secret_key",
        "replace_with_a_secure_random_key_in_production",
        "your_secret_key_change_in_production"
    ]:
        logger.warning("Using default/insecure FLASK_SECRET_KEY for development. Set FLASK_SECRET_KEY in .env for production.")
        FLASK_SECRET_KEY = FLASK_SECRET_KEY or "dev-insecure-secret-key-change-in-prod"

app.secret_key = FLASK_SECRET_KEY

# Configure session cookies
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

DB = os.environ.get("DATABASE_PATH", "parking.db")
MODEL_PATH = os.environ.get("MODEL_PATH", "price_prediction_model.pkl")
ADMIN_SIGNUP_PASSCODE = os.environ.get("ADMIN_SIGNUP_PASSCODE", "ParkEase@123")

def get_db_path():
    """Resolve database path dynamically, prioritizing app.config for testing isolation."""
    return app.config.get("DATABASE_PATH") or os.environ.get("DATABASE_PATH", DB)

# Initialize weather classifier
classifier = TrichyWeatherClassifier()

# Model Caching: Load model once at startup, reload dynamically if target path changes
LOADED_MODEL = None
LOADED_MODEL_PATH = None

def get_model(model_path=None):
    """
    Load and cache ML pricing model.
    Prioritizes explicit argument, app.config, and environment variable, defaulting to price_prediction_model.pkl.
    """
    global LOADED_MODEL, LOADED_MODEL_PATH
    target_path = model_path or app.config.get("MODEL_PATH") or os.environ.get("MODEL_PATH", "price_prediction_model.pkl")
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
    target_db = db_path or get_db_path()
    conn = sqlite3.connect(target_db)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS parking_slots (slot_id INTEGER PRIMARY KEY, status TEXT NOT NULL)''')
    c.execute('SELECT COUNT(*) FROM parking_slots')
    if c.fetchone()[0] == 0:
        for i in range(1, 101):
            c.execute("INSERT INTO parking_slots (slot_id, status) VALUES (?, ?)", (i, 'free'))
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        phone TEXT NOT NULL,
        vehicle_number TEXT NOT NULL,
        vehicle_type TEXT NOT NULL,
        electric TEXT NOT NULL,
        time TEXT NOT NULL,
        slot_id INTEGER NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS admin (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        mobile TEXT NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS parking_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        phone TEXT NOT NULL,
        vehicle_number TEXT NOT NULL,
        vehicle_type TEXT NOT NULL,
        electric TEXT NOT NULL,
        entry_time TEXT NOT NULL,
        exit_time TEXT,
        slot_id INTEGER NOT NULL,
        duration_min REAL,
        price REAL,
        weather TEXT,
        congestion REAL)''')
    conn.commit()
    conn.close()

# Ensure schema exists on module load (supports WSGI / Gunicorn)
try:
    init_db()
except Exception as e:
    logger.warning("Database schema verification notice: %s", e)

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))   

    if request.method == 'POST':
        name = request.form['name']
        phone = request.form['phone']
        vehicle_number = request.form['vehicle_number']
        vehicle_type = request.form['vehicle_type']
        electric = request.form['electric']
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Validate vehicle type matches model expectations
        valid_vehicle_types = ['auto', 'car', '2-wheeler', 'suv/van']
        if vehicle_type not in valid_vehicle_types:
            flash('Invalid vehicle type selected', 'error')
            return redirect(url_for('index'))

        # Atomic slot reservation and user creation within a single transaction
        conn = sqlite3.connect(get_db_path(), timeout=20.0)
        try:
            conn.execute("BEGIN IMMEDIATE")
            c = conn.cursor()
            c.execute("""
                UPDATE parking_slots 
                SET status = 'occupied' 
                WHERE slot_id = (
                    SELECT slot_id FROM parking_slots 
                    WHERE status = 'free' 
                    ORDER BY slot_id ASC LIMIT 1
                )
                RETURNING slot_id
            """)
            row = c.fetchone()
            if not row:
                conn.rollback()
                conn.close()
                return "<h2>No slots available</h2>", 400

            slot_id = row[0]
            c.execute('''INSERT INTO users (name, phone, vehicle_number, vehicle_type, electric, time, slot_id)
                         VALUES (?, ?, ?, ?, ?, ?, ?)''', 
                         (name, phone, vehicle_number, vehicle_type, electric, timestamp, slot_id))
            user_id = c.lastrowid
            conn.commit()
            session['user_id'] = user_id
        except Exception as e:
            conn.rollback()
            logger.error("Error during atomic check-in transaction: %s", e)
            raise
        finally:
            conn.close()

        return redirect(url_for('dashboard'))

    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('index'))

    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute('SELECT name, phone, vehicle_number, vehicle_type, electric, time, slot_id FROM users WHERE id=?',
              (session['user_id'],))
    user = c.fetchone()
    conn.close()

    if user:
        user = list(user)
        user[4] = user[4].title()
        user[3] = user[3].title()
        user = tuple(user)
        return render_template('dashboard.html', user=user)
    return "<h2>User not found</h2>", 404

@app.route('/exit', methods=['POST'])
def exit_parking():
    if 'user_id' not in session:
        return redirect(url_for('index'))

    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute('SELECT name, vehicle_number, vehicle_type, electric, time, slot_id FROM users WHERE id=?', (session['user_id'],))
    user_data = c.fetchone()

    if user_data:
        name, vehicle_number, vehicle_type, electric, start_time_str, slot_id = user_data
        start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
        exit_time = datetime.now()
        duration_min = max(0.0, round((exit_time - start_time).total_seconds() / 60, 1))
        weather = classifier.classify_weather()
        c.execute('SELECT COUNT(*) FROM parking_slots WHERE status="occupied"')
        filled = c.fetchone()[0]
        congestion = round(filled / 100.0, 4)
        conn.close()

        # Format input for cached ML model
        input_df = pd.DataFrame([[duration_min, vehicle_type,
                                "electric" if electric == "yes" else "non-electric",
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

        # Store exit data in session for receipt and payment logging
        session['exit_data'] = {
            'name': name,
            'vehicle_number': vehicle_number,
            'entry_time': start_time_str,
            'exit_time': exit_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_min': duration_min,
            'price': price,
            'slot_id': slot_id,
            'weather': weather,
            'congestion': congestion
        }
        
        return redirect(url_for('exit_receipt'))

    conn.close()
    return redirect(url_for('index'))

@app.route('/exit_receipt')
def exit_receipt():
    if 'exit_data' not in session or 'user_id' not in session:
        return redirect(url_for('index'))
    
    return render_template('exit.html', **session['exit_data'])

@app.route('/process_payment', methods=['POST'])
def process_payment():
    if 'exit_data' not in session or 'user_id' not in session:
        return redirect(url_for('index'))

    conn = sqlite3.connect(get_db_path(), timeout=20.0)
    try:
        conn.execute("BEGIN IMMEDIATE")
        c = conn.cursor()
        exit_data = session['exit_data']
        
        # Log the parking session with exact exit_data metrics
        c.execute('''INSERT INTO parking_logs 
                    (name, phone, vehicle_number, vehicle_type, electric, 
                     entry_time, exit_time, slot_id, duration_min, price, weather, congestion)
                    SELECT name, phone, vehicle_number, vehicle_type, electric, 
                           time, ?, slot_id, ?, ?, ?, ?
                    FROM users WHERE id=?''',
                 (exit_data['exit_time'], exit_data['duration_min'], exit_data['price'], 
                  exit_data.get('weather', 'average'), exit_data.get('congestion', 0.0), session['user_id']))
        
        c.execute('UPDATE parking_slots SET status="free" WHERE slot_id=?', (exit_data['slot_id'],))
        c.execute('DELETE FROM users WHERE id=?', (session['user_id'],))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error("Error during payment processing transaction: %s", e)
        raise
    finally:
        conn.close()
    
    # Clear session data
    session.pop('user_id', None)
    session.pop('exit_data', None)
    
    return redirect(url_for('index'))

@app.route('/admin_signup', methods=['GET', 'POST'])
def admin_signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        mobile = request.form['mobile']
        org_password = request.form['org_password']
        if org_password != ADMIN_SIGNUP_PASSCODE:
            return "<h2>Invalid organization password!</h2>"

        hashed_password = generate_password_hash(password)
        conn = sqlite3.connect(get_db_path())
        c = conn.cursor()
        try:
            c.execute('INSERT INTO admin (username, password, mobile) VALUES (?, ?, ?)', 
                      (username, hashed_password, mobile))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return "<h2>Username already exists.</h2>"
        conn.close()
        return redirect(url_for('admin_login'))
    return render_template('admin_signup.html')

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(get_db_path())
        c = conn.cursor()
        c.execute('SELECT id, password FROM admin WHERE username=?', (username,))
        admin = c.fetchone()

        authenticated = False
        if admin:
            admin_id, stored_password = admin[0], admin[1]
            
            # Check Werkzeug hash
            if stored_password.startswith(('pbkdf2:', 'scrypt:')):
                if check_password_hash(stored_password, password):
                    authenticated = True
            # Graceful upgrade for legacy plaintext password
            elif stored_password == password:
                authenticated = True
                new_hash = generate_password_hash(password)
                c.execute('UPDATE admin SET password=? WHERE id=?', (new_hash, admin_id))
                conn.commit()

            if authenticated:
                conn.close()
                session['admin_id'] = admin_id
                return redirect(url_for('admin_dashboard'))

        conn.close()
        flash('Invalid credentials', 'error')
    return render_template('admin_login.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
    
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    
    # Get all parking slots
    c.execute('SELECT slot_id, status FROM parking_slots ORDER BY slot_id')
    slots = [{'slot_id': row[0], 'status': row[1], 'user_details': None} for row in c.fetchall()]
    
    # Get all users with their slot information
    c.execute('SELECT * FROM users')
    users = c.fetchall()
    
    # Create a mapping of slot_id to user details
    user_map = {user[7]: user for user in users}
    
    # Update slots with user details if occupied
    for slot in slots:
        if slot['status'] == 'occupied' and slot['slot_id'] in user_map:
            slot['user_details'] = user_map[slot['slot_id']]
    
    # Calculate statistics
    c.execute('SELECT COUNT(*) FROM parking_slots WHERE status="occupied"')
    occupied_count = c.fetchone()[0]
    free_count = 100 - occupied_count
    occupancy_rate = round((occupied_count / 100) * 100, 1)
    
    conn.close()
    
    return render_template('admin_dashboard.html', 
                         slots=slots,
                         occupied_count=occupied_count,
                         free_count=free_count,
                         occupancy_rate=occupancy_rate)

@app.route('/admin_logs')
def admin_logs():
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
    
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute('SELECT * FROM parking_logs ORDER BY exit_time DESC')
    logs = c.fetchall()
    logs = [list(log) for log in logs]

    for log in logs:
        if isinstance(log[10], bytes):
            try:
                log[10] = float(int.from_bytes(log[10], byteorder='little', signed=False))
            except Exception:
                log[10] = None
        if isinstance(log[12], bytes):
            try:
                log[12] = float(int.from_bytes(log[12], byteorder='little', signed=False))
            except Exception:
                log[12] = None

    conn.close()
    
    return render_template('admin_logs.html', logs=logs)

@app.route('/admin_logout')
def admin_logout():
    session.pop('admin_id', None)
    return redirect(url_for('admin_login'))

if __name__ == '__main__':
    debug_mode = (APP_ENV != "production")
    app.run(debug=debug_mode)