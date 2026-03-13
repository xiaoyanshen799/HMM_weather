#!/usr/bin/env python3
"""Merge per-city client split CSVs into one file per split, sorted by timestamp."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

SPLITS = ("features", "train", "val", "test")
TIMESTAMP_COL = "timestamp_utc"


def parse_client_file(path: Path) -> Tuple[str, str]:
    # Expected: client_<city>_<client_id>_<split>.csv
    stem = path.stem
    if not stem.startswith("client_"):
        raise ValueError(f"Unexpected filename (missing 'client_'): {path.name}")

    rest = stem[len("client_"):]
    for split in SPLITS:
        suffix = f"_{split}"
        if rest.endswith(suffix):
            core = rest[: -len(suffix)]
            idx = core.rfind("_")
            if idx == -1:
                raise ValueError(f"Unexpected filename (missing client id): {path.name}")
            city = core[:idx]
            client_id = core[idx + 1 :]
            if not city or not client_id:
                raise ValueError(f"Unexpected filename fields: {path.name}")
            if not client_id.isdigit():
                raise ValueError(f"Skip non-client shard file: {path.name}")
            return city, split

    raise ValueError(f"Unexpected split in filename: {path.name}")


def discover_files(processed_root: Path) -> Dict[str, Dict[str, List[Path]]]:
    grouped: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in processed_root.rglob("client_*_*.csv"):
        if not path.is_file():
            continue
        try:
            city, split = parse_client_file(path)
        except ValueError:
            continue
        grouped[city][split].append(path)

    for city in grouped:
        for split in grouped[city]:
            grouped[city][split].sort()

    return grouped


def read_rows(path: Path) -> Tuple[List[str], List[dict]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV missing header: {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def merge_split(files: List[Path], output_path: Path) -> int:
    if not files:
        return 0

    merged_rows: List[dict] = []
    header: List[str] = []

    for i, file_path in enumerate(files):
        cols, rows = read_rows(file_path)
        if i == 0:
            header = cols
            if TIMESTAMP_COL not in header:
                raise ValueError(f"Missing '{TIMESTAMP_COL}' in {file_path}")
        elif cols != header:
            raise ValueError(
                "Header mismatch between files:\n"
                f"  base: {files[0].name}\n"
                f"  curr: {file_path.name}"
            )
        merged_rows.extend(rows)

    merged_rows.sort(key=lambda r: r[TIMESTAMP_COL])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(merged_rows)

    return len(merged_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("processed"),
        help="Root folder containing per-city processed client CSVs.",
    )
    parser.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help="Optional list of city names to merge (example: London Bangkok).",
    )
    parser.add_argument(
        "--output-name-template",
        default="client_{city}_merged_{split}.csv",
        help="Output file name template with placeholders: {city}, {split}.",
    )
    args = parser.parse_args()

    grouped = discover_files(args.processed_root)
    if not grouped:
        raise SystemExit(f"No client CSV files found under: {args.processed_root}")

    target_cities = set(args.cities) if args.cities else set(grouped.keys())

    for city in sorted(target_cities):
        if city not in grouped:
            print(f"[WARN] City not found: {city}")
            continue

        print(f"\\n=== {city} ===")
        for split in SPLITS:
            files = grouped[city].get(split, [])
            if not files:
                print(f"- {split}: skipped (no files)")
                continue

            out_name = args.output_name_template.format(city=city, split=split)
            out_path = args.processed_root / city / out_name
            count = merge_split(files, out_path)
            print(f"- {split}: {len(files)} files -> {count} rows -> {out_path}")


if __name__ == "__main__":
    main()
