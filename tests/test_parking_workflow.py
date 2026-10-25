"""
Regression tests for the driver check-in → checkout → payment workflow and the
admin views. All personal details are obviously fictional test values.
"""

import sqlite3
import struct

import pytest

import app as app_module
from app import app, init_db

VALID_CHECKIN = {
    "name": "Test Driver",
    "phone": "0000000010",
    "vehicle_number": "TEST-0010",
    "vehicle_type": "car",
    "electric": "no",
}


def db():
    conn = sqlite3.connect(app.config["DATABASE_PATH"])
    conn.row_factory = sqlite3.Row
    return conn


def check_in(client, **overrides):
    return client.post("/", data={**VALID_CHECKIN, **overrides})


def active_user(vehicle_number=VALID_CHECKIN["vehicle_number"]):
    with db() as conn:
        return conn.execute("SELECT * FROM users WHERE vehicle_number=?", (vehicle_number,)).fetchone()


def occupied_slots():
    with db() as conn:
        return conn.execute("SELECT COUNT(*) FROM parking_slots WHERE status='occupied'").fetchone()[0]


def login_as_admin(client):
    with client.session_transaction() as sess:
        sess["admin_id"] = 1


# ---------------------------------------------------------------------------
# Fee persistence
# ---------------------------------------------------------------------------

def test_fee_is_calculated_once_and_cannot_be_recalculated(client, monkeypatch):
    check_in(client)
    assert client.post("/exit").status_code == 302
    first = dict(active_user())
    assert first["exit_time"] is not None and first["price"] is not None

    # Make any recalculation observable: a second pricing attempt would fail loudly.
    def fail_if_called(*args, **kwargs):
        raise AssertionError("price must not be recalculated after a quote exists")
    monkeypatch.setattr(app_module, "get_model", fail_if_called)
    monkeypatch.setattr(app_module.classifier, "classify_weather", fail_if_called)

    response = client.post("/exit")
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/exit_receipt")
    assert dict(active_user()) == first


def test_dashboard_redirects_to_receipt_once_quoted(client):
    check_in(client)
    client.post("/exit")
    response = client.get("/dashboard")
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/exit_receipt")
    assert b"Total Amount" in client.get("/exit_receipt").data


def test_payment_uses_stored_quote_not_session_values(client):
    check_in(client)
    client.post("/exit")
    quoted_price = active_user()["price"]

    with client.session_transaction() as sess:
        # Values an older version trusted from the session cookie
        sess["exit_data"] = {"price": -999.0, "duration_min": 0.0, "slot_id": 42}

    assert client.post("/process_payment").status_code == 302
    with db() as conn:
        log = conn.execute("SELECT price, slot_id FROM parking_logs").fetchone()
        slot_42 = conn.execute("SELECT status FROM parking_slots WHERE slot_id=42").fetchone()[0]
    assert log["price"] == quoted_price
    assert log["slot_id"] == 1
    assert slot_42 == "free"
    assert occupied_slots() == 0


def test_payment_without_quote_is_refused(client):
    check_in(client)
    response = client.post("/process_payment")
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    assert active_user() is not None
    with db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM parking_logs").fetchone()[0] == 0


def test_receipt_requires_quote(client):
    check_in(client)
    response = client.get("/exit_receipt")
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")


def test_double_payment_is_harmless(client):
    check_in(client)
    client.post("/exit")
    client.post("/process_payment")
    response = client.post("/process_payment")
    assert response.status_code == 302
    with db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM parking_logs").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# Session consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,path", [
    ("get", "/dashboard"), ("post", "/exit"), ("get", "/exit_receipt"), ("post", "/process_payment"),
])
def test_missing_user_record_resets_session(client, method, path):
    with client.session_transaction() as sess:
        sess["user_id"] = 9999

    response = getattr(client, method)(path, follow_redirects=True)
    assert response.status_code == 200
    assert b"could not be found" in response.data
    assert b'name="vehicle_number"' in response.data, "driver should be back on the check-in form"
    with client.session_transaction() as sess:
        assert "user_id" not in sess


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("overrides,message", [
    ({"name": ""}, b"Name is required"),
    ({"name": "x" * 101}, b"Name is required"),
    ({"phone": "not-a-phone"}, b"Phone number"),
    ({"phone": "123"}, b"Phone number"),
    ({"vehicle_number": ""}, b"Vehicle number"),
    ({"vehicle_number": "A" * 21}, b"Vehicle number"),
    ({"vehicle_number": "<script>"}, b"Vehicle number"),
    ({"vehicle_type": "spaceship"}, b"Invalid vehicle type"),
    ({"electric": "maybe"}, b"electric"),
])
def test_invalid_checkin_is_rejected_without_side_effects(client, overrides, message):
    response = check_in(client, **overrides)
    assert response.status_code == 400
    assert message in response.data
    assert occupied_slots() == 0
    with client.session_transaction() as sess:
        assert "user_id" not in sess


def test_missing_form_field_does_not_crash(client):
    response = client.post("/", data={"name": "Test Driver"})
    assert response.status_code == 400
    assert occupied_slots() == 0


def test_checkin_values_are_normalized(client):
    check_in(client, phone="00000 00011", vehicle_number="  test-0011 ", name="  Test Driver  ")
    user = active_user("TEST-0011")
    assert user is not None
    assert user["phone"] == "0000000011"
    assert user["name"] == "Test Driver"


def test_rejected_form_keeps_entered_values(client):
    response = check_in(client, phone="bad")
    assert b'value="Test Driver"' in response.data
    assert b'value="car" selected' in response.data


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------

def test_capacity_is_configurable(client, tmp_path, monkeypatch):
    small_db = str(tmp_path / "small.db")
    monkeypatch.setitem(app.config, "PARKING_CAPACITY", 2)
    monkeypatch.setitem(app.config, "DATABASE_PATH", small_db)
    init_db(small_db)

    for i in range(2):
        with client.session_transaction() as sess:
            sess.pop("user_id", None)
        assert check_in(client, vehicle_number=f"TEST-01{i}").status_code == 302

    with client.session_transaction() as sess:
        sess.pop("user_id", None)
    response = check_in(client, vehicle_number="TEST-0199")
    assert response.status_code == 400
    assert b"No parking slots are available" in response.data

    login_as_admin(client)
    page = client.get("/admin_dashboard").data
    assert b"2 / 2" in page and b"0 / 2" in page
    assert b"100.0%" in page


def test_congestion_uses_actual_capacity(client, tmp_path, monkeypatch):
    small_db = str(tmp_path / "four.db")
    monkeypatch.setitem(app.config, "PARKING_CAPACITY", 4)
    monkeypatch.setitem(app.config, "DATABASE_PATH", small_db)
    init_db(small_db)

    check_in(client)
    client.post("/exit")
    assert active_user()["congestion"] == 0.25


def test_init_db_is_idempotent_and_migrates_legacy_schema(tmp_path):
    legacy = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(legacy)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, "
                 "phone TEXT NOT NULL, vehicle_number TEXT NOT NULL, vehicle_type TEXT NOT NULL, "
                 "electric TEXT NOT NULL, time TEXT NOT NULL, slot_id INTEGER NOT NULL)")
    conn.execute("INSERT INTO users (name, phone, vehicle_number, vehicle_type, electric, time, slot_id) "
                 "VALUES ('Test Driver', '0000000012', 'TEST-0012', 'car', 'no', '2026-01-01 10:00:00', 1)")
    conn.commit()
    conn.close()

    init_db(legacy)
    init_db(legacy)

    conn = sqlite3.connect(legacy)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
    assert {"exit_time", "price", "duration_min", "weather", "congestion"} <= columns
    assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM parking_slots").fetchone()[0] == 100
    conn.close()


# ---------------------------------------------------------------------------
# Admin log formatting
# ---------------------------------------------------------------------------

def test_decode_legacy_number():
    decode = app_module.decode_legacy_number
    assert decode(struct.pack("<f", 42.5)) == 42.5
    assert decode(struct.pack("<d", 12.25)) == 12.25
    assert decode(b"\x01\x02\x03") is None
    assert decode(7.0) == 7.0
    assert decode(None) is None


def test_admin_logs_show_zero_and_legacy_values(client):
    with db() as conn:
        conn.execute(
            "INSERT INTO parking_logs (name, phone, vehicle_number, vehicle_type, electric, entry_time, "
            "exit_time, slot_id, duration_min, price, weather, congestion) VALUES "
            "('Zero Fee', '0000000013', 'TEST-0013', 'car', 'no', '2026-01-01 10:00:00', "
            "'2026-01-01 10:00:00', 1, 0.0, 0.0, 'good', 0.0)"
        )
        conn.execute(
            "INSERT INTO parking_logs (name, phone, vehicle_number, vehicle_type, electric, entry_time, "
            "exit_time, slot_id, duration_min, price, weather, congestion) VALUES "
            "('Legacy Blob', '0000000014', 'TEST-0014', 'car', 'no', '2026-01-01 09:00:00', "
            "'2026-01-01 09:30:00', 2, 30.0, ?, 'average', 0.5)",
            (struct.pack("<f", 37.5),),
        )

    login_as_admin(client)
    page = client.get("/admin_logs").data.decode()
    zero_row = page.split("Zero Fee")[1].split("</tr>")[0]
    assert "<td>0.0</td>" in zero_row
    assert "<td>0.00</td>" in zero_row
    assert "<td>-</td>" not in zero_row
    legacy_row = page.split("Legacy Blob")[1].split("</tr>")[0]
    assert "<td>37.50</td>" in legacy_row


# ---------------------------------------------------------------------------
# Database error handling
# ---------------------------------------------------------------------------

def test_checkin_database_error_rolls_back_slot(client):
    with db() as conn:
        conn.execute("DROP TABLE users")
    response = check_in(client)
    assert response.status_code == 503
    assert b"could not reserve a slot" in response.data
    assert occupied_slots() == 0, "slot reservation must be rolled back"


def test_payment_database_error_keeps_session_and_slot(client):
    check_in(client)
    client.post("/exit")
    with db() as conn:
        conn.execute("DROP TABLE parking_logs")

    response = client.post("/process_payment", follow_redirects=True)
    assert response.status_code == 200
    assert b"Payment could not be recorded" in response.data
    assert active_user() is not None
    assert occupied_slots() == 1
    with client.session_transaction() as sess:
        assert "user_id" in sess
