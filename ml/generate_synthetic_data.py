#!/usr/bin/env python3
"""
Synthetic Dataset Generator for ParkEase Smart Parking System.

IMPORTANT SCIENTIFIC NOTICE:
This script generates a clearly labelled SYNTHETIC DEMONSTRATION DATASET.
It is constructed solely for verifying the reproducibility of machine learning
pipelines, hyperparameter tuning, experiment tracking, and Flask integration.
It does NOT represent real-world observational parking data, and metrics
derived from this dataset cannot establish real-world pricing accuracy.

Simulation Model Overview:
1. Hourly Base Rate by Vehicle Type (R_vehicle):
   - 2-wheeler: 15.00 INR/hr
   - auto:      25.00 INR/hr
   - car:       40.00 INR/hr
   - suv/van:   60.00 INR/hr
2. Duration Treatment:
   - Converted to fractional hours: h = max(0.25, duration_min / 60.0) [15 min minimum base]
   - Long-stay discount: beyond 4 hours, additional hours billed at 85% rate:
     effective_hours = min(h, 4.0) + 0.85 * max(0.0, h - 4.0)
     base_fee = R_vehicle * effective_hours
3. EV Adjustment:
   - electric: 15% incentive discount (multiplier = 0.85)
   - non-electric: standard rate (multiplier = 1.00)
4. Weather Adjustment:
   - bad: +15% premium (multiplier = 1.15)
   - average: standard rate (multiplier = 1.00)
   - good: -5% discount (multiplier = 0.95)
5. Congestion Surcharge:
   - Surge factor = 1.0 + (0.30 * congestion_level)  [0% to 30% surge]
6. Noise & Boundary:
   - Gaussian noise ~ N(0, 1.5)
   - Minimum fee enforced: max(5.00, fee)
"""

import os
import sys
import hashlib
import argparse
import numpy as np
import pandas as pd

ALLOWED_VEHICLE_TYPES = ["car", "2-wheeler", "auto", "suv/van"]
ALLOWED_EV_VALUES = ["electric", "non-electric"]
ALLOWED_WEATHER_VALUES = ["good", "average", "bad"]

EXPECTED_COLUMNS = [
    "Duration (min)",
    "Vehicle Type",
    "Electric",
    "Weather",
    "Congestion Level",
    "Parking Fee (INR)"
]

VEHICLE_BASE_RATES = {
    "2-wheeler": 15.0,
    "auto": 25.0,
    "car": 40.0,
    "suv/van": 60.0
}


def generate_synthetic_dataset(n_records=2500, seed=42):
    """
    Generate synthetic parking dataset deterministically.

    Args:
        n_records (int): Number of records to generate (default: 2500).
        seed (int): Random seed for reproducibility (default: 42).

    Returns:
        pd.DataFrame: Deterministic synthetic parking dataset.
    """
    np.random.seed(seed)

    # Realistic stay distributions: short (35%), medium (50%), long stays (15%)
    n_short = int(n_records * 0.35)
    n_med = int(n_records * 0.50)
    n_long = n_records - n_short - n_med

    d_short = np.random.uniform(5.0, 60.0, n_short)
    d_med = np.random.uniform(60.0, 240.0, n_med)
    d_long = np.random.uniform(240.0, 720.0, n_long)

    durations = np.round(np.concatenate([d_short, d_med, d_long]), 1)
    np.random.shuffle(durations)

    # Categorical distributions
    vehicles = np.random.choice(
        ALLOWED_VEHICLE_TYPES,
        size=n_records,
        p=[0.45, 0.35, 0.10, 0.10]
    )
    electric = np.random.choice(
        ALLOWED_EV_VALUES,
        size=n_records,
        p=[0.20, 0.80]
    )
    weather = np.random.choice(
        ALLOWED_WEATHER_VALUES,
        size=n_records,
        p=[0.35, 0.50, 0.15]
    )

    # Congestion drawn from Beta distribution bounded [0, 1]
    congestion = np.round(np.random.beta(a=2.0, b=3.0, size=n_records), 4)
    congestion = np.clip(congestion, 0.0, 1.0)

    # Price Calculation
    fees = []
    for d, v, e, w, c in zip(durations, vehicles, electric, weather, congestion):
        h = max(0.25, d / 60.0)
        effective_h = min(h, 4.0) + 0.85 * max(0.0, h - 4.0)
        base_fee = VEHICLE_BASE_RATES[v] * effective_h

        ev_mult = 0.85 if e == "electric" else 1.0
        weather_mult = 1.15 if w == "bad" else (0.95 if w == "good" else 1.0)
        congestion_mult = 1.0 + (0.30 * c)

        noise = np.random.normal(0.0, 1.5)
        raw_calculated = (base_fee * ev_mult * weather_mult * congestion_mult) + noise
        final_fee = round(max(5.0, raw_calculated), 2)
        fees.append(final_fee)

    df = pd.DataFrame({
        "Duration (min)": durations,
        "Vehicle Type": vehicles,
        "Electric": electric,
        "Weather": weather,
        "Congestion Level": congestion,
        "Parking Fee (INR)": fees
    })

    return df


def validate_dataset(df, expected_records=2500):
    """
    Validate dataset schema, data types, ranges, duplicates, and constraints.

    Args:
        df (pd.DataFrame): Dataset to validate.
        expected_records (int): Expected row count.

    Returns:
        dict: Summary of validation findings.
    """
    report = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "row_count": len(df),
        "column_count": len(df.columns),
        "duplicates": int(df.duplicated().sum()),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_ranges": {},
        "categorical_counts": {}
    }

    # 1. Exact Column Names and Ordering
    if list(df.columns) != EXPECTED_COLUMNS:
        report["is_valid"] = False
        report["errors"].append(f"Column mismatch. Expected: {EXPECTED_COLUMNS}, Got: {list(df.columns)}")

    # 2. Row count
    if len(df) != expected_records:
        report["is_valid"] = False
        report["errors"].append(f"Row count mismatch. Expected: {expected_records}, Got: {len(df)}")

    # 3. Missing values
    total_nulls = df.isnull().sum().sum()
    if total_nulls > 0:
        report["is_valid"] = False
        report["errors"].append(f"Dataset contains {total_nulls} missing values.")

    # 4. Duplicate checks (Report explicitly)
    if report["duplicates"] > 0:
        report["warnings"].append(f"Identified {report['duplicates']} exact duplicate records.")

    # 5. Allowed categorical values
    for col, allowed in [
        ("Vehicle Type", ALLOWED_VEHICLE_TYPES),
        ("Electric", ALLOWED_EV_VALUES),
        ("Weather", ALLOWED_WEATHER_VALUES)
    ]:
        if col in df.columns:
            invalid = set(df[col].unique()) - set(allowed)
            if invalid:
                report["is_valid"] = False
                report["errors"].append(f"Invalid categories in '{col}': {invalid}")
            report["categorical_counts"][col] = df[col].value_counts().to_dict()

    # 6. Numeric ranges
    if "Duration (min)" in df.columns:
        dur_min, dur_max = float(df["Duration (min)"].min()), float(df["Duration (min)"].max())
        report["numeric_ranges"]["Duration (min)"] = (dur_min, dur_max)
        if dur_min < 0:
            report["is_valid"] = False
            report["errors"].append(f"Duration has negative values: {dur_min}")

    if "Congestion Level" in df.columns:
        cong_min, cong_max = float(df["Congestion Level"].min()), float(df["Congestion Level"].max())
        report["numeric_ranges"]["Congestion Level"] = (cong_min, cong_max)
        if cong_min < 0.0 or cong_max > 1.0:
            report["is_valid"] = False
            report["errors"].append(f"Congestion out of bounds [0, 1]: ({cong_min}, {cong_max})")

    if "Parking Fee (INR)" in df.columns:
        fee_min, fee_max = float(df["Parking Fee (INR)"].min()), float(df["Parking Fee (INR)"].max())
        report["numeric_ranges"]["Parking Fee (INR)"] = (fee_min, fee_max)
        if fee_min < 0.0:
            report["is_valid"] = False
            report["errors"].append(f"Target fee has negative values: {fee_min}")

    return report


def compute_file_sha256(filepath):
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic parking demonstration data.")
    parser.add_argument("--output", type=str, default="ml/data/parking_synthetic_demo.csv", help="Output CSV path")
    parser.add_argument("--n-records", type=int, default=2500, help="Number of records to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic output")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"[INFO] Generating {args.n_records} synthetic demonstration records with seed={args.seed}...")
    df = generate_synthetic_dataset(n_records=args.n_records, seed=args.seed)

    print("[INFO] Validating generated dataset...")
    validation = validate_dataset(df, expected_records=args.n_records)

    if not validation["is_valid"]:
        print(f"[ERROR] Validation failed with errors: {validation['errors']}")
        sys.exit(1)

    # Save deterministic CSV (without index)
    df.to_csv(args.output, index=False)
    file_hash = compute_file_sha256(args.output)

    print(f"[SUCCESS] Dataset saved to: {args.output}")
    print(f"[INFO] File SHA-256: {file_hash}")
    print(f"[INFO] Shape: {df.shape}")
    print(f"[INFO] Duplicates: {validation['duplicates']}")
    print(f"[INFO] Missing values: {sum(validation['missing_values'].values())}")
    print(f"[INFO] Numeric Ranges: {validation['numeric_ranges']}")
    print(f"[INFO] Categorical Counts: {validation['categorical_counts']}")


if __name__ == "__main__":
    main()
