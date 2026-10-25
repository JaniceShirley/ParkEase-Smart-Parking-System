# ParkEase - Smart Parking System

ParkEase is an intelligent, automated parking management and dynamic pricing web application built with Python and Flask. The system simulates a parking facility (100 slots by default), provides real-time slot allocation, manages driver check-in/checkout workflows, predicts parking fees using Machine Learning, and provides an administrative portal for facility monitoring.

---

## Features

- **Automated Slot Allocation:** Atomic slot assignment supporting four vehicle categories (`car`, `2-wheeler`, `auto`, `suv/van`) and electric vehicles (EV).
- **Dynamic Fee Prediction:** Integrated Machine Learning pipeline estimating parking fees dynamically based on duration, vehicle class, EV status, local weather conditions, and real-time facility congestion.
- **Tamper-Resistant Checkout:** The fee is calculated once when the driver requests checkout, stored in the database, and used unchanged at payment.
- **Weather-Aware Pricing:** Live Tiruchirappalli (Trichy) weather metrics fetched via OpenWeatherMap API with automatic fault-tolerant fallback.
- **Administrative Portal:** Protected dashboard featuring live slot occupancy visual grids, occupancy statistics, and comprehensive audit logs.
- **Secure Authentication:** Salted password hashing, CSRF protection on every form, and rate-limited login and signup.

---

## Technology Stack

- **Backend:** Python (3.10 – 3.11), Flask 2.3.2, Werkzeug 3.0.6, Flask-WTF, Flask-Limiter, Gunicorn
- **Database:** SQLite 3 (native transactions with `BEGIN IMMEDIATE` and `RETURNING`; SQLite 3.35+)
- **Machine Learning:** Scikit-Learn 1.6.1, XGBoost 2.1.4, Pandas 2.0.3, NumPy 1.26.4, Joblib
- **External APIs:** OpenWeatherMap REST API
- **Frontend:** Semantic HTML5, Vanilla CSS3 (Custom Design System), JavaScript (ES6)

> Python 3.12+ is not supported by the pinned `pandas==2.0.3`. CI runs on Python 3.11.

---

## Directory Structure

```
ParkEase-Smart-Parking-System/
├── .env.example                 # Environment configuration template
├── .github/workflows/ci.yml     # CI: tests, training smoke test, artifact checksums
├── app.py                       # Flask application, routes, CLI commands
├── config.py                    # Environment parsing and security validation
├── database.py                  # SQLite schema creation and migrations
├── weatherAPI.py                # OpenWeatherMap wrapper and weather classifier
├── price_prediction_model.pkl   # Production pricing model (Scikit-Learn/XGBoost pipeline)
├── models/xgboost_parking_v1.pkl  # Candidate model produced by ml/train.py
├── ml/                          # Synthetic data generator, training pipeline, demo dataset
├── tools/create_demo_db.py      # Creates a synthetic demonstration database
├── table view.py                # Maintenance: clear parking_logs (dry run unless --confirm)
├── requirements.txt             # Pinned runtime dependencies
├── requirements-test.txt        # Runtime + test dependencies
├── static/                      # CSS, JavaScript and images
├── templates/                   # Jinja2 templates
└── tests/                       # pytest suite
```

---

## Local Setup & Installation

### 1. Prerequisites
- Python 3.10 or 3.11
- Git
- macOS only: XGBoost needs the OpenMP runtime (`brew install libomp`, or `conda install llvm-openmp` in a conda environment)

### 2. Clone and Create Virtual Environment

```bash
git clone https://github.com/JaniceShirley/ParkEase-Smart-Parking-System.git
cd ParkEase-Smart-Parking-System

python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt          # to run the app
pip install -r requirements-test.txt     # to also run tests and ml/train.py
pip check
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

The template selects **production**, which is also the default when `APP_ENV` is not set at all. For local development, edit `.env` and set `APP_ENV=development` (see [Running the Application](#running-the-application)).

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `APP_ENV` | `production` | `production` or `development`. Development must be selected explicitly. `testing` is accepted only inside a pytest run. Gunicorn always requires `production`. `FLASK_ENV` is ignored. Any other value is rejected. |
| `FLASK_SECRET_KEY` | *(random per process in development)* | Signs session cookies. **Required in production**: at least 32 characters and not a published placeholder. |
| `FLASK_DEBUG` | `false` | Enables debug mode. `FLASK_DEBUG=true`, `flask --debug` and `flask run --debugger` are **refused unless `APP_ENV=development`**. |
| `DATABASE_PATH` | `instance/parking.db` | SQLite database. Created automatically. `instance/` is git-ignored. The legacy `parking.db` location in the repository root is refused. |
| `MODEL_PATH` | `price_prediction_model.pkl` | Pricing model loaded at startup. Only load model files you trust: they are pickles. |
| `PARKING_CAPACITY` | `100` | Number of parking slots created in a new database (1–10000). |
| `OPENWEATHER_API_KEY` | *(empty)* | Optional. Without it weather is treated as `average`. |
| `ADMIN_SIGNUP_PASSCODE` | *(empty → signup disabled)* | Enables `/admin_signup` when set to a non-default value of at least 12 characters. |
| `LOGIN_RATE_LIMIT` | `5 per minute;20 per hour` | Failed and successful login POSTs per client IP. Validated at startup: a malformed value or a zero limit stops the app (it never silently disables rate limiting). Empty uses the default. |
| `SIGNUP_RATE_LIMIT` | `5 per hour` | Signup POSTs per client IP. Validated the same way. |
| `RATELIMIT_STORAGE_URI` | `memory://` | Rate-limit store. Use a shared store (e.g. `redis://…`) with multiple workers. |
| `SESSION_COOKIE_SECURE` | `true` in production | Send the session cookie over HTTPS only. |
| `TRUSTED_PROXY_COUNT` | `0` | Whole number 0–5: how many reverse proxies' `X-Forwarded-For` / `X-Forwarded-Proto` headers are trusted. **It must equal the actual number of proxies in front of the app**; a higher value lets clients forge their IP address and evade rate limits. Keep `0` when clients connect directly. |

Generate a secret key with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Never commit `.env`; it is listed in `.gitignore`.

---

## Database

- The app uses `DATABASE_PATH` (default `instance/parking.db`). On first start it creates the schema and `PARKING_CAPACITY` free slots. Existing databases are migrated additively (new nullable columns only; no rows are deleted).
- You can also initialise it explicitly:

  ```bash
  flask --app app init-db
  ```

- **Demo data.** To explore the admin pages with realistic-looking but entirely fictional data:

  ```bash
  python tools/create_demo_db.py                   # writes instance/demo_parking.db, never overwrites
  APP_ENV=development DATABASE_PATH=instance/demo_parking.db python app.py
  ```

- **Administrators.** Create accounts from the command line (passwords are prompted, never echoed, and stored hashed). The commands load the same configuration as the app, so run them with the same environment (production needs `FLASK_SECRET_KEY`; locally use `APP_ENV=development`):

  ```bash
  flask --app app create-admin
  flask --app app reset-admin-password <username>
  ```

- **Legacy `parking.db`.** Earlier versions kept a `parking.db` in the repository root. It contained personal data, has been removed from version control and is listed in `.gitignore`; CI fails if it is ever committed again. Older checkouts may still have a local copy. The app refuses to start if `DATABASE_PATH` refers to that location in any form (relative or absolute path, `..` segments, different letter case, symlink or hard link), whether or not the file exists, so an old `.env` containing `DATABASE_PATH=parking.db` fails with a clear error instead of reusing or recreating it. If you need the old data locally, copy it to a private location first (for example `instance/legacy.db`) and point `DATABASE_PATH` at the copy. Accounts with unhashed (plaintext) passwords cannot log in; reset them with `reset-admin-password`.

- **Clearing logs.** `python "table view.py"` reports how many rows `parking_logs` holds; add `--confirm` to delete them. It refuses to modify the legacy `parking.db` in the repository folder.

---

## Running the Application

### Local Development
Development mode must be selected explicitly; without it the app runs in production mode and refuses to start without a strong `FLASK_SECRET_KEY`. Either set `APP_ENV=development` in your local `.env`, or pass it on the command line:

```bash
APP_ENV=development python app.py                         # debugger off
APP_ENV=development FLASK_DEBUG=true python app.py        # debugger on
APP_ENV=development flask --app app --debug run           # same, via the Flask CLI
```
Access the application at `http://127.0.0.1:5000`. In development the secret key may be left empty (a random one is generated per process), cookies are not marked `Secure`, and the debugger is allowed. Never use development mode on a server reachable by others.

### Production WSGI Server (Gunicorn)
```bash
export FLASK_SECRET_KEY=...    # from your secrets manager
gunicorn -w 2 -b 127.0.0.1:5000 app:app
```
Production is the default, and Gunicorn refuses any other `APP_ENV`.

---

## Running Automated Tests

```bash
pip install -r requirements-test.txt
pytest tests/ -v
```

The suite is hermetic: `tests/conftest.py` pins test-only settings before the app is imported, uses temporary databases, mocks the weather API, and verifies that both model files and the demo dataset are never modified, and that a local legacy `parking.db` (if present) is never modified or created. The tests do not need any real database: where a protected legacy file is required, they create a small fictional one. A local `.env` does not affect the results.

To run the training pipeline without touching committed artifacts:

```bash
python ml/train.py --force --model-output /tmp/candidate.pkl --tracking-uri file:/tmp/mlruns
```

---

## Production Deployment Checklist

1. Production mode (the default; `APP_ENV=production` or unset) — the app refuses to start without a strong `FLASK_SECRET_KEY`, or with `FLASK_DEBUG`, `--debug` or `--debugger`. Gunicorn refuses to serve any other `APP_ENV`, and `APP_ENV=testing` only works inside pytest. Never set `APP_ENV=development` on a server.
2. Set `FLASK_SECRET_KEY` from a secrets manager or environment, never from a committed file.
3. Serve over HTTPS (session cookies are `Secure` in production). Behind a reverse proxy, set `TRUSTED_PROXY_COUNT` to exactly the number of proxies.
4. Run under Gunicorn (or another WSGI server), never `python app.py`.
5. Use a shared `RATELIMIT_STORAGE_URI` (e.g. Redis) when running more than one worker.
6. Leave `ADMIN_SIGNUP_PASSCODE` empty and create administrators with `flask --app app create-admin`.
7. Keep `DATABASE_PATH` outside the repository and back it up; it contains customer names and phone numbers.

---

## Machine Learning Model & Pricing Policy

The parking fee prediction pipeline expects 5 features in the following schema:
1. `Duration (min)`: Number of minutes parked (Numeric $\ge 0.0$).
2. `Vehicle Type`: Categorical (`'auto'`, `'car'`, `'2-wheeler'`, `'suv/van'`).
3. `Electric`: Categorical (`'electric'`, `'non-electric'`).
4. `Weather`: Categorical (`'good'`, `'average'`, `'bad'`).
5. `Congestion Level`: Facility occupancy ratio ($0.0 \le x \le 1.0$), computed from the actual number of slots.

### Non-Negative Fee Boundary
The model is loaded once into memory upon process initialization. When inferring parking fees, outputs are clamped to zero at the application boundary:
$$\text{Fee} = \max(0.00, \text{predicted\_price})$$
*Note: Clamping enforces a business rule prohibiting negative charges on short stays; it does not alter the underlying model weights.*

### Checkout Quote
The fee, exit time, weather and congestion are stored on the driver's record the first time checkout is requested. Repeating the checkout request returns the same quote, and payment logs exactly the stored values.

---

## Weather API Integration & Fallback Policy

- Live weather for Tiruchirappalli (Trichy) is queried via OpenWeatherMap.
- To prevent network disruptions from breaking the checkout workflow, requests use a 5-second timeout.
- Any network, timeout, HTTP, parsing or unexpected failure triggers a safe fallback to `'average'` weather, allowing price prediction to proceed without disruption.

---

## Security & Database Architecture

- **Atomic Transactions:** Slot allocation uses `BEGIN IMMEDIATE` and `UPDATE ... RETURNING slot_id` to guarantee that concurrent requests cannot be assigned the same slot. Checkout quotes are written with an `exit_time IS NULL` guard, and payment logs, slot release and record removal happen in one transaction.
- **Secret Key:** No predictable fallback exists. Production refuses missing, short or published keys; other environments use a random per-process key.
- **Debug Mode:** Production is the default. Debug mode and the interactive debugger (`FLASK_DEBUG`, `flask --debug`, `flask run --debugger`) are refused unless `APP_ENV=development` is set explicitly.
- **Password Security:** Administrator passwords are hashed with `werkzeug.security`. Stored plaintext passwords are never accepted. A failed login runs a password-hash check whether or not the username exists, so response timing does not reveal valid usernames.
- **Administrator Registration:** Disabled unless `ADMIN_SIGNUP_PASSCODE` is set to a strong value; passcodes are compared in constant time.
- **CSRF Protection:** Every state-changing form (check-in, checkout, payment, login, signup, logout) carries a Flask-WTF CSRF token. Logout is POST-only.
- **Rate Limiting:** Login and signup POSTs are limited per client IP with Flask-Limiter.
- **Session Hardening:** Cookies are `HttpOnly`, `SameSite=Lax`, and `Secure` in production. Fees are never read from the session cookie.
- **Input Validation:** Names (≤100 characters), phone numbers (7–15 digits), vehicle numbers (≤20 letters, digits, spaces or hyphens), vehicle type and EV status are validated server-side.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
