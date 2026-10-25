"""
SQLite schema management for ParkEase.

Kept free of Flask imports so that maintenance scripts can create or migrate a
database without starting the web application.
"""

import os
import sqlite3

SLOT_FREE = "free"
SLOT_OCCUPIED = "occupied"

# Legacy database location in the repository folder. The file used to be committed,
# contains historical personal data, and is no longer tracked, but older checkouts may
# still have a local copy. It must never be used (or recreated) as a runtime database.
COMMITTED_BASELINE_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parking.db")


class CommittedDatabaseError(RuntimeError):
    """Raised when a path refers to the committed baseline database."""


def refers_to_committed_baseline(db_path):
    """
    True if ``db_path`` is the same file as the legacy parking.db in the repository folder.

    Relative paths resolve against the current working directory, exactly as
    sqlite3.connect resolves them. When both files exist, comparison uses file
    identity (device and inode), so case variants, ``..`` segments, symlinks and
    hard links are all detected. When either is missing (e.g. a fresh checkout,
    where the legacy file is absent) the resolved paths are compared, so an old
    ``DATABASE_PATH=parking.db`` cannot recreate a database at the legacy location.
    Other filesystem errors (e.g. permission denied) propagate so callers can fail closed.
    """
    candidate = os.path.abspath(os.fspath(db_path))
    try:
        return os.path.samefile(candidate, COMMITTED_BASELINE_DB)
    except FileNotFoundError:
        def normalized(path):
            return os.path.normcase(os.path.realpath(path))
        return normalized(candidate) == normalized(COMMITTED_BASELINE_DB)


def assert_not_committed_baseline(db_path):
    if refers_to_committed_baseline(db_path):
        raise CommittedDatabaseError(
            "DATABASE_PATH points to the legacy parking.db location in the repository folder, "
            "which must not be used as the runtime database (the legacy file may contain "
            "historical personal data). Set DATABASE_PATH to a different file "
            "(default: instance/parking.db)."
        )

# Columns added to ``users`` after the original schema. They hold the checkout
# quote so the fee is calculated once and cannot be re-rolled.
_USER_QUOTE_COLUMNS = {
    "exit_time": "TEXT",
    "duration_min": "REAL",
    "price": "REAL",
    "weather": "TEXT",
    "congestion": "REAL",
}


def connect(db_path, timeout=20.0):
    return sqlite3.connect(db_path, timeout=timeout)


def _ensure_parent_dir(db_path):
    parent = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(parent, exist_ok=True)


def init_db(db_path, capacity):
    """
    Create the schema if needed, apply additive migrations and make sure slots
    1..capacity exist. Existing rows are never deleted or modified. Refuses to
    touch the committed baseline database.
    """
    assert_not_committed_baseline(db_path)
    _ensure_parent_dir(db_path)
    conn = connect(db_path)
    try:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS parking_slots (slot_id INTEGER PRIMARY KEY, status TEXT NOT NULL)''')
        c.executemany(
            "INSERT OR IGNORE INTO parking_slots (slot_id, status) VALUES (?, ?)",
            [(i, SLOT_FREE) for i in range(1, capacity + 1)],
        )
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT NOT NULL,
            vehicle_number TEXT NOT NULL,
            vehicle_type TEXT NOT NULL,
            electric TEXT NOT NULL,
            time TEXT NOT NULL,
            slot_id INTEGER NOT NULL)''')
        existing = {row[1] for row in c.execute("PRAGMA table_info(users)")}
        for column, column_type in _USER_QUOTE_COLUMNS.items():
            if column not in existing:
                c.execute(f"ALTER TABLE users ADD COLUMN {column} {column_type}")
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
    finally:
        conn.close()
