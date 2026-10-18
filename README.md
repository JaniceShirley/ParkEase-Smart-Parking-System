# ParkEase - Smart Parking System

ParkEase is an intelligent, automated parking management and dynamic pricing web application built with Python and Flask. The system simulates a 100-slot parking facility, provides real-time slot allocation, manages driver check-in/checkout workflows, predicts parking fees using Machine Learning, and provides an administrative portal for facility monitoring.

---

## Features

- **Automated Slot Allocation:** Atomic slot assignment supporting four vehicle categories (`car`, `2-wheeler`, `auto`, `suv/van`) and electric vehicles (EV).
- **Dynamic Fee Prediction:** Integrated Machine Learning pipeline estimating parking fees dynamically based on duration, vehicle class, EV status, local weather conditions, and real-time facility congestion.
- **Weather-Aware Pricing:** Live Tiruchirappalli (Trichy) weather metrics fetched via OpenWeatherMap API with automatic fault-tolerant fallback.
- **Administrative Portal:** Protected dashboard featuring live slot occupancy visual grids, occupancy statistics, and comprehensive audit logs.
- **Secure Authentication:** Salted cryptographic password hashing with automatic transparent migration for legacy administrator accounts.

---

## Technology Stack

- **Backend:** Python (3.10 – 3.13), Flask 2.3.2, Gunicorn
- **Database:** SQLite 3 (native transactions with `BEGIN IMMEDIATE` and `RETURNING`)
- **Machine Learning:** Scikit-Learn 1.6.1, XGBoost 2.1.4, Pandas, Joblib
- **External APIs:** OpenWeatherMap REST API
- **Frontend:** Semantic HTML5, Vanilla CSS3 (Custom Design System), JavaScript (ES6)

---

## Directory Structure

```
ParkEase-Smart-Parking-System/
├── .env.example                 # Environment configuration template
├── .gitignore                   # Version control ignore specifications
├── LICENSE                      # MIT License
├── README.md                    # System documentation
├── app.py                       # Core Flask application, routes, and controllers
├── parking.db                   # SQLite application database
├── price_prediction_model.pkl   # Serialized ML pipeline artifact (Scikit-Learn/XGBoost)
├── requirements.txt             # Pinned Python package dependencies
├── weatherAPI.py                # OpenWeatherMap wrapper and weather classifier
├── static/                      # Static assets
│   ├── css/style.css            # Stylesheet design system
│   ├── js/scripts.js            # Client-side form handling and live table filtering
│   ├── favicon.png              # Favicon asset
│   └── ParkEase.png             # Application logo
└── templates/                   # Jinja2 presentation templates
    ├── index.html               # Vehicle entry registration form
    ├── dashboard.html           # Active parking session display
    ├── exit.html                # Parking receipt & payment confirmation
    ├── admin_login.html         # Administrator login portal
    ├── admin_signup.html        # Administrator registration portal
    ├── admin_dashboard.html     # Real-time slot occupancy grid
    └── admin_logs.html          # Historical checkout records table
```

---

## Local Setup & Installation

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 2. Clone and Create Virtual Environment

```bash
git clone https://github.com/JaniceShirley/ParkEase-Smart-Parking-System.git
cd ParkEase-Smart-Parking-System

# Create isolated virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the provided `.env.example` file to `.env`:

```bash
cp .env.example .env
```

Edit `.env` to configure your settings:

```env
# Application Environment ('development' or 'production')
APP_ENV=development

# Flask Secret Key (In production, must be a secure random key)
FLASK_SECRET_KEY=dev-insecure-secret-key-change-in-prod
FLASK_DEBUG=False

# Database and Model Locations
DATABASE_PATH=parking.db
MODEL_PATH=price_prediction_model.pkl

# OpenWeatherMap API Key (Optional for basic testing; defaults to 'average' if missing)
OPENWEATHER_API_KEY=your_openweather_api_key_here

# Administrator Signup Passcode
ADMIN_SIGNUP_PASSCODE=ParkEase@123
```

---

## Running the Application

### Development Server
```bash
python app.py
```
Access the application at `http://127.0.0.1:5000`.

### Production WSGI Server (Gunicorn)
```bash
gunicorn -w 2 -b 127.0.0.1:5000 app:app
```

---

## Machine Learning Model & Pricing Policy

The parking fee prediction pipeline expects 5 features in the following schema:
1. `Duration (min)`: Number of minutes parked (Numeric $\ge 0.0$).
2. `Vehicle Type`: Categorical (`'auto'`, `'car'`, `'2-wheeler'`, `'suv/van'`).
3. `Electric`: Categorical (`'electric'`, `'non-electric'`).
4. `Weather`: Categorical (`'good'`, `'average'`, `'bad'`).
5. `Congestion Level`: Facility occupancy ratio ($0.0 \le x \le 1.0$).

### Non-Negative Fee Boundary
The model is loaded once into memory upon process initialization. When inferring parking fees, outputs are clamped to zero at the application boundary:
$$\text{Fee} = \max(0.00, \text{predicted\_price})$$
*Note: Clamping enforces a business rule prohibiting negative charges on short stays; it does not alter the underlying model weights.*

---

## Weather API Integration & Fallback Policy

- Live weather for Tiruchirappalli (Trichy) is queried via OpenWeatherMap.
- To prevent network disruptions from breaking the checkout workflow, requests use a 5-second timeout.
- Any network, timeout, HTTP, or parsing failure triggers a safe fallback to `'average'` weather, allowing price prediction to proceed without disruption.

---

## Security & Database Architecture

- **Atomic Transactions:** Slot allocation uses `BEGIN IMMEDIATE` and `UPDATE ... RETURNING slot_id` to guarantee that concurrent requests cannot be assigned the same slot.
- **Password Security:** Administrator passwords registered via `/admin_signup` are hashed with `werkzeug.security`. Existing plaintext accounts in legacy databases are verified and upgraded automatically to salted hashes upon successful login.
- **Production Secret Enforcement:** When `APP_ENV=production`, the application refuses to start unless a custom, secure `FLASK_SECRET_KEY` is provided.
- **Session Hardening:** Cookies are flagged `HttpOnly` with `SameSite=Lax`.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.