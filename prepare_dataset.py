"""
Prepare PM2.5 datasets: convert station pressure to sea level pressure (SLP),
optionally build time/lag/rolling features, and split into train/val/test.

Defaults assume hourly data and use a standard barometric formula under the
standard atmosphere (lapse rate 0.0065 K/m) for SLP.

Usage:
    python prepare_dataset.py \
        --dataset-root dataset \
        --output-root processed \
        [--overwrite] \
        [--skip-features]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

# Approximate city-level elevations (meters). Override with metadata file if you
# have per-station altitudes, which is preferred.
DEFAULT_ALTITUDES_M: Dict[str, float] = {
    "Bangkok": 1.5,
    "Bhaktapur": 1330.0,
    "London": 11.0,
    "New_York": 10.0,
    "Sydney": 3.0,
}


@dataclass
class Args:
    dataset_root: str
    output_root: str
    overwrite: bool
    skip_features: bool
    train_ratio: float
    val_ratio: float
    lag_steps: List[int]
    rolling_windows: List[int]
    altitude_metadata: str


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Convert station pressure to sea level pressure and engineer features.")
    parser.add_argument("--dataset-root", default="dataset", help="Root folder containing city subfolders with CSV files.")
    parser.add_argument("--output-root", default="processed", help="Folder to place feature/split outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the original CSV with the SLP column.")
    parser.add_argument("--skip-features", action="store_true", help="Only compute SLP; skip feature engineering and splits.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio (time-ordered).")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio (time-ordered).")
    parser.add_argument("--lag-steps", type=int, nargs="+", default=[1, 3, 6, 12, 24], help="Lag steps (in records) to include.")
    parser.add_argument("--rolling-windows", type=int, nargs="+", default=[3, 12, 24], help="Rolling window sizes (in records) for mean.")
    parser.add_argument("--altitude-metadata", default="metadata/city_altitudes.json", help="JSON file with {city: altitude_m}.")
    ns = parser.parse_args()
    return Args(
        dataset_root=ns.dataset_root,
        output_root=ns.output_root,
        overwrite=ns.overwrite,
        skip_features=ns.skip_features,
        train_ratio=ns.train_ratio,
        val_ratio=ns.val_ratio,
        lag_steps=ns.lag_steps,
        rolling_windows=ns.rolling_windows,
        altitude_metadata=ns.altitude_metadata,
    )


def parse_timestamp(ts: str) -> datetime:
    # Handle timestamps with trailing Z or +00:00
    ts = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def station_to_sea_level(p_station_hpa: float, temp_c: float, altitude_m: float) -> Optional[float]:
    exp_factor = 1 - (0.0065 * altitude_m) / (temp_c + 0.0065 * altitude_m + 273.15)
    if exp_factor <= 0:
        return None
    return p_station_hpa * (exp_factor ** -5.257)


def load_altitudes(metadata_path: str) -> Dict[str, float]:
    altitudes = dict(DEFAULT_ALTITUDES_M)
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            for city, val in loaded.items():
                try:
                    altitudes[city] = float(val)
                except (TypeError, ValueError):
                    print(f"[WARN] Altitude for {city} is not numeric; using default if available.")
    return altitudes


def list_city_csvs(root: str) -> Iterable[Tuple[str, str]]:
    for city in os.listdir(root):
        city_path = os.path.join(root, city)
        if not os.path.isdir(city_path):
            continue
        for fname in os.listdir(city_path):
            if not fname.endswith(".csv"):
                continue
            yield city, os.path.join(city_path, fname)


def read_rows(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pm25 = float(row["pm25"])
                temp_c = float(row["temperature_c"])
                humidity = float(row["humidity"])
                pressure = float(row["pressure"])
            except (KeyError, TypeError, ValueError):
                continue
            ts_str = row.get("timestamp_utc")
            if not ts_str:
                continue
            ts_dt = parse_timestamp(ts_str)
            rows.append(
                {
                    "timestamp": ts_dt,
                    "timestamp_utc": ts_str,
                    "pm25": pm25,
                    "temperature_c": temp_c,
                    "humidity": humidity,
                    "pressure": pressure,
                }
            )
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def add_sea_level_pressure(rows: List[dict], altitude_m: float) -> List[dict]:
    out: List[dict] = []
    for r in rows:
        slp = station_to_sea_level(r["pressure"], r["temperature_c"], altitude_m)
        if slp is None:
            continue
        r_with_slp = dict(r)
        r_with_slp["sea_level_pressure"] = slp
        out.append(r_with_slp)
    return out


def time_features(dt: datetime) -> dict:
    hour = dt.hour
    dow = dt.weekday()  # 0 Monday
    month = dt.month
    return {
        "sin_hour": math.sin(2 * math.pi * hour / 24),
        "cos_hour": math.cos(2 * math.pi * hour / 24),
        "sin_dow": math.sin(2 * math.pi * dow / 7),
        "cos_dow": math.cos(2 * math.pi * dow / 7),
        "sin_month": math.sin(2 * math.pi * month / 12),
        "cos_month": math.cos(2 * math.pi * month / 12),
    }


def generate_feature_rows(rows: List[dict], lag_steps: List[int], rolling_windows: List[int]) -> List[dict]:
    base_cols = ["pm25", "temperature_c", "humidity", "sea_level_pressure"]
    history: Dict[str, List[float]] = {c: [] for c in base_cols}
    max_window = max(max(lag_steps, default=0), max(rolling_windows, default=0))

    output: List[dict] = []
    for r in rows:
        # Compute lags/rolling from history BEFORE adding current value to avoid leakage.
        lag_feats: Dict[str, Optional[float]] = {}
        roll_feats: Dict[str, Optional[float]] = {}
        for col in base_cols:
            hist = history[col]
            for lag in lag_steps:
                lag_feats[f"{col}_lag_{lag}"] = hist[-lag] if len(hist) >= lag else None
            for window in rolling_windows:
                roll_feats[f"{col}_rollmean_{window}"] = (
                    sum(hist[-window:]) / window if len(hist) >= window else None
                )

        # Add time encodings.
        tf = time_features(r["timestamp"])

        full_row = dict(r)
        full_row.update(tf)
        full_row.update(lag_feats)
        full_row.update(roll_feats)

        # Decide if row has enough history to keep.
        needed = list(lag_feats.values()) + list(roll_feats.values())
        if any(v is None for v in needed):
            # Still append current value to history then continue.
            for col in base_cols:
                history[col].append(full_row[col])
                if len(history[col]) > max_window:
                    history[col].pop(0)
            continue

        output.append(full_row)

        for col in base_cols:
            history[col].append(full_row[col])
            if len(history[col]) > max_window:
                history[col].pop(0)

    return output


def split_rows(
    rows: List[dict], train_ratio: float, val_ratio: float
) -> Tuple[List[dict], List[dict], List[dict]]:
    n = len(rows)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = rows[:train_end]
    val = rows[train_end:val_end]
    test = rows[val_end:]
    return train, val, test


def write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def process_file(path: str, city: str, altitude_m: float, args: Args) -> None:
    print(f"[INFO] Processing {path} (altitude {altitude_m} m)")
    rows = read_rows(path)
    rows = add_sea_level_pressure(rows, altitude_m)
    if not rows:
        print(f"[WARN] No usable rows after SLP conversion for {path}")
        return

    slp_fieldnames = ["timestamp_utc", "pm25", "temperature_c", "humidity", "pressure", "sea_level_pressure"]
    slp_target = path if args.overwrite else path.replace(".csv", "_slp.csv")
    write_csv(slp_target, rows, slp_fieldnames)
    print(f"[INFO] Wrote SLP data -> {slp_target}")

    if args.skip_features:
        return

    feat_rows = generate_feature_rows(rows, args.lag_steps, args.rolling_windows)
    if not feat_rows:
        print(f"[WARN] Not enough history to build features for {path}")
        return

    # Drop internal datetime field before writing.
    for r in feat_rows:
        r.pop("timestamp", None)

    # Stable column order: base, time feats, lags, rolling.
    base_cols = ["timestamp_utc", "pm25", "temperature_c", "humidity", "pressure", "sea_level_pressure"]
    time_cols = ["sin_hour", "cos_hour", "sin_dow", "cos_dow", "sin_month", "cos_month"]
    lag_cols = [c for c in feat_rows[0].keys() if "_lag_" in c]
    roll_cols = [c for c in feat_rows[0].keys() if "_rollmean_" in c]
    fieldnames = base_cols + time_cols + sorted(lag_cols) + sorted(roll_cols)

    rel_name = os.path.basename(path).replace(".csv", "_features.csv")
    feat_path = os.path.join(args.output_root, city, rel_name)
    write_csv(feat_path, feat_rows, fieldnames)
    print(f"[INFO] Wrote feature dataset -> {feat_path}")

    train, val, test = split_rows(feat_rows, args.train_ratio, args.val_ratio)
    base_name = os.path.basename(path).replace(".csv", "")
    split_dir = os.path.join(args.output_root, city)
    write_csv(os.path.join(split_dir, f"{base_name}_train.csv"), train, fieldnames)
    write_csv(os.path.join(split_dir, f"{base_name}_val.csv"), val, fieldnames)
    write_csv(os.path.join(split_dir, f"{base_name}_test.csv"), test, fieldnames)
    print(
        f"[INFO] Split sizes train/val/test: {len(train)}/{len(val)}/{len(test)} "
        f"-> {os.path.join(split_dir, base_name)}_[train|val|test].csv"
    )


def main() -> None:
    args = parse_args()
    altitudes = load_altitudes(args.altitude_metadata)

    for city, path in list_city_csvs(args.dataset_root):
        altitude_m = altitudes.get(city)
        if altitude_m is None:
            print(f"[WARN] Missing altitude for {city}; using 0 m (no adjustment).")
            altitude_m = 0.0
        process_file(path, city, altitude_m, args)


if __name__ == "__main__":
    main()
