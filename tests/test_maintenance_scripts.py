"""Tests for the maintenance scripts: the log-clearing utility and the demo database generator."""

import importlib.util
import os
import sqlite3

import pytest

import database
from ml.train import compute_file_sha256
from tests.helpers import LOCAL_LEGACY_DB, REPO_ROOT, file_state, make_fictional_legacy_db
from tools.create_demo_db import create_demo_db


def load_clear_logs_module():
    spec = importlib.util.spec_from_file_location("clear_logs", os.path.join(REPO_ROOT, "table view.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_db_with_logs(path, rows=3):
    database.init_db(str(path), 10)
    conn = sqlite3.connect(path)
    for n in range(rows):
        conn.execute(
            "INSERT INTO parking_logs (name, phone, vehicle_number, vehicle_type, electric, entry_time, slot_id) "
            "VALUES (?, '0000000000', ?, 'car', 'no', '2026-01-01 10:00:00', 1)",
            (f"Test Visitor {n}", f"TEST-{n}"),
        )
    conn.commit()
    conn.close()


def log_count(path):
    conn = sqlite3.connect(path)
    count = conn.execute("SELECT COUNT(*) FROM parking_logs").fetchone()[0]
    conn.close()
    return count


def test_clear_logs_is_dry_run_by_default(tmp_path, capsys):
    db_path = tmp_path / "work.db"
    make_db_with_logs(db_path)
    assert load_clear_logs_module().main(["--db", str(db_path)]) == 0
    assert log_count(db_path) == 3
    assert "Dry run" in capsys.readouterr().out


def test_clear_logs_deletes_only_with_confirm(tmp_path):
    db_path = tmp_path / "work.db"
    make_db_with_logs(db_path)
    assert load_clear_logs_module().main(["--db", str(db_path), "--confirm"]) == 0
    assert log_count(db_path) == 0


def test_clear_logs_refuses_local_legacy_db():
    # Only meaningful in a checkout that still has the untracked legacy file. Dry-run
    # mode only: even if the guard regressed, nothing would be deleted.
    if not os.path.exists(LOCAL_LEGACY_DB):
        pytest.skip("no local legacy parking.db in this checkout")
    before = file_state(LOCAL_LEGACY_DB)
    assert load_clear_logs_module().main(["--db", LOCAL_LEGACY_DB]) == 1
    assert file_state(LOCAL_LEGACY_DB) == before


def test_clear_logs_points_at_missing_legacy_location_safely(capsys):
    if os.path.exists(LOCAL_LEGACY_DB):
        pytest.skip("this checkout still has a local legacy parking.db")
    assert load_clear_logs_module().main(["--db", LOCAL_LEGACY_DB, "--confirm"]) == 1
    assert not os.path.exists(LOCAL_LEGACY_DB)


@pytest.fixture
def protected_copy(tmp_path, monkeypatch):
    """A fictional stand-in registered as the protected legacy database, so --confirm can be exercised safely."""
    copy = make_fictional_legacy_db(tmp_path / "parking.db")
    monkeypatch.setattr(database, "COMMITTED_BASELINE_DB", str(copy))
    return copy


def _is_case_insensitive(path):
    return os.path.exists(os.path.join(os.path.dirname(path), os.path.basename(path).upper()))


@pytest.mark.parametrize("variant", ["absolute", "relative", "dot-segments", "case", "symlink", "hardlink"])
def test_clear_logs_refuses_every_spelling_of_the_baseline(protected_copy, tmp_path, monkeypatch, variant):
    before = compute_file_sha256(protected_copy)
    if variant == "absolute":
        target = str(protected_copy)
    elif variant == "relative":
        monkeypatch.chdir(tmp_path)
        target = "parking.db"
    elif variant == "dot-segments":
        (tmp_path / "sub").mkdir()
        target = str(tmp_path / "sub" / ".." / "parking.db")
    elif variant == "case":
        if not _is_case_insensitive(str(protected_copy)):
            pytest.skip("filesystem is case-sensitive")
        target = str(tmp_path / "PARKING.DB")
    elif variant == "symlink":
        target = str(tmp_path / "alias.db")
        try:
            os.symlink(protected_copy, target)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported here")
    else:
        target = str(tmp_path / "hardlink.db")
        try:
            os.link(protected_copy, target)
        except (OSError, NotImplementedError):
            pytest.skip("hard links not supported here")

    assert load_clear_logs_module().main(["--db", target, "--confirm"]) == 1
    assert compute_file_sha256(protected_copy) == before


def test_clear_logs_fails_closed_when_identity_check_errors(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "work.db"
    make_db_with_logs(db_path)

    def denied(path):
        raise PermissionError("simulated")
    monkeypatch.setattr(database, "refers_to_committed_baseline", denied)

    assert load_clear_logs_module().main(["--db", str(db_path), "--confirm"]) == 1
    assert log_count(db_path) == 3
    assert "nothing was deleted" in capsys.readouterr().err


def test_clear_logs_handles_non_database_files(tmp_path, capsys):
    not_a_db = tmp_path / "notes.db"
    not_a_db.write_bytes(b"this is not sqlite")
    assert load_clear_logs_module().main(["--db", str(not_a_db), "--confirm"]) == 1
    assert not_a_db.read_bytes() == b"this is not sqlite"

    other_schema = tmp_path / "other.db"
    sqlite3.connect(other_schema).close()
    assert load_clear_logs_module().main(["--db", str(other_schema), "--confirm"]) == 1
    assert "nothing was deleted" in capsys.readouterr().err


def test_clear_logs_rejects_directories(tmp_path):
    assert load_clear_logs_module().main(["--db", str(tmp_path), "--confirm"]) == 1


def test_clear_logs_never_creates_a_database(tmp_path):
    missing = tmp_path / "missing.db"
    assert load_clear_logs_module().main(["--db", str(missing), "--confirm"]) == 1
    assert not missing.exists()


def test_demo_db_contains_only_synthetic_data(tmp_path):
    output = tmp_path / "demo.db"
    create_demo_db(str(output), capacity=20, active=5, history=8, seed=1)

    conn = sqlite3.connect(output)
    users = conn.execute("SELECT name, phone, vehicle_number, slot_id FROM users").fetchall()
    logs = conn.execute("SELECT name, vehicle_number, price FROM parking_logs").fetchall()
    occupied = conn.execute("SELECT COUNT(*) FROM parking_slots WHERE status='occupied'").fetchone()[0]
    admins = conn.execute("SELECT COUNT(*) FROM admin").fetchone()[0]
    conn.close()

    assert len(users) == 5 and occupied == 5
    assert len(logs) == 8
    assert admins == 0
    assert all(name.startswith("Demo Driver") and plate.startswith("DEMO-") for name, _, plate, _ in users)
    assert all(name.startswith("Demo Visitor") and plate.startswith("DEMO-") and price > 0 for name, plate, price in logs)


def test_demo_db_never_overwrites(tmp_path):
    output = tmp_path / "demo.db"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        create_demo_db(str(output), capacity=20, active=1, history=1, seed=1)
    assert output.read_bytes() == b"existing"
