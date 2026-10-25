#!/usr/bin/env python3
"""
Create a SYNTHETIC demonstration database for ParkEase.

Every name, phone number and vehicle number is an obviously fictional
placeholder (e.g. "Demo Driver 001", "0000000001", "DEMO-001"). No
administrator accounts are created; use `flask --app app create-admin`.

    python tools/create_demo_db.py                         # writes instance/demo_parking.db
    python tools/create_demo_db.py --output path/demo.db --active 15 --history 40

An existing file is never overwritten.
"""

import argparse
import os
import random
import sqlite3
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import database  # noqa: E402

VEHICLE_TYPES = ["car", "2-wheeler", "auto", "suv/van"]
WEATHER = ["good", "average", "bad"]
BASE_RATE = {"2-wheeler": 10.0, "auto": 15.0, "car": 20.0, "suv/van": 30.0}
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def demo_fee(vehicle_type, electric, duration_min, weather, congestion):
    """Simple transparent formula for demo rows; it is not the ML model."""
    hours = max(duration_min / 60.0, 0.25)
    fee = BASE_RATE[vehicle_type] * hours * (1.0 + congestion)
    if electric == "yes":
        fee *= 0.9
    if weather == "bad":
        fee *= 1.1
    return round(fee, 2)


def create_demo_db(output, capacity, active, history, seed):
    if os.path.exists(output):
        raise FileExistsError(f"{output} already exists; refusing to overwrite it.")
    if active > capacity:
        raise ValueError("--active cannot exceed the parking capacity.")

    rng = random.Random(seed)
    database.init_db(output, capacity)
    now = datetime.now().replace(microsecond=0)

    conn = sqlite3.connect(output)
    try:
        for n in range(1, history + 1):
            vehicle_type = rng.choice(VEHICLE_TYPES)
            electric = rng.choice(["yes", "no"])
            weather = rng.choice(WEATHER)
            congestion = round(rng.uniform(0.05, 0.95), 4)
            duration = round(rng.uniform(10, 480), 1)
            exit_time = now - timedelta(hours=rng.uniform(1, 24 * 14))
            entry_time = exit_time - timedelta(minutes=duration)
            conn.execute(
                "INSERT INTO parking_logs (name, phone, vehicle_number, vehicle_type, electric, entry_time, "
                "exit_time, slot_id, duration_min, price, weather, congestion) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f"Demo Visitor {n:03d}", f"{n:010d}", f"DEMO-H{n:03d}", vehicle_type, electric,
                 entry_time.strftime(TIME_FORMAT), exit_time.strftime(TIME_FORMAT),
                 rng.randint(1, capacity), duration,
                 demo_fee(vehicle_type, electric, duration, weather, congestion), weather, congestion),
            )

        for slot_id in rng.sample(range(1, capacity + 1), active):
            entry_time = now - timedelta(minutes=rng.randint(5, 300))
            conn.execute(
                "INSERT INTO users (name, phone, vehicle_number, vehicle_type, electric, time, slot_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"Demo Driver {slot_id:03d}", f"{5000 + slot_id:010d}", f"DEMO-{slot_id:03d}",
                 rng.choice(VEHICLE_TYPES), rng.choice(["yes", "no"]),
                 entry_time.strftime(TIME_FORMAT), slot_id),
            )
            conn.execute("UPDATE parking_slots SET status=? WHERE slot_id=?", (database.SLOT_OCCUPIED, slot_id))
        conn.commit()
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Create a synthetic ParkEase demo database.")
    parser.add_argument("--output", default="instance/demo_parking.db")
    parser.add_argument("--capacity", type=int, default=config.DEFAULT_PARKING_CAPACITY)
    parser.add_argument("--active", type=int, default=12, help="Vehicles currently parked")
    parser.add_argument("--history", type=int, default=30, help="Completed sessions in parking_logs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        create_demo_db(args.output, args.capacity, args.active, args.history, args.seed)
    except (FileExistsError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Synthetic demo database written to {args.output}")
    print(f"Run the app against it with: APP_ENV=development DATABASE_PATH={args.output} python app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
