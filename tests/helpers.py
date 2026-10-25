"""Shared test helpers."""

import hashlib
import os
import sqlite3

import database

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# A developer checkout may still contain the legacy, untracked parking.db; CI checkouts do not.
LOCAL_LEGACY_DB = os.path.join(REPO_ROOT, "parking.db")


def file_state(path):
    """(sha256, size, mtime_ns) of a file, or None if it does not exist."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    st = os.stat(path)
    return digest, st.st_size, st.st_mtime_ns


def make_fictional_legacy_db(path):
    """
    Create a small stand-in for the legacy parking.db containing only fictional data.
    Tests use it wherever they need "the protected legacy database" so that no
    real records are ever required, copied or touched.
    """
    database.init_db(str(path), 5)
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO parking_logs (name, phone, vehicle_number, vehicle_type, electric, entry_time, slot_id) "
        "VALUES ('Fictional Visitor', '0000000000', 'FAKE-0001', 'car', 'no', '2026-01-01 10:00:00', 1)"
    )
    conn.commit()
    conn.close()
    return path
