#!/usr/bin/env python3
"""
Validate structural integrity of a Certum artifacts run directory.

This script ensures:
- Required files exist
- JSONL structure is valid
- Calibration fields are present
- Scored outputs contain required fields

It does NOT perform analysis.
It only validates integrity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

REQUIRED_ENERGY_FIELDS = ["energy"]
REQUIRED_CALIBRATION_FIELDS = [
    "tau_energy",
    "tau_sensitivity",
    "tau_pr",
    "tau_by_percentile",
    "tau_energy_by_percentile",
    "tau_pr_by_percentile",
    "pos_energies"
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name}: Invalid JSON at line {i}: {e}")
    return rows


def validate_jsonl_structure(rows: List[Dict[str, Any]], filename: str):
    if not rows:
        raise ValueError(f"{filename} is empty.")

    for field in REQUIRED_ENERGY_FIELDS:
        if field not in rows[0]:
            raise ValueError(f"{filename} missing required field: {field}")


def validate_calibration(calibration_json: Dict[str, Any]):
    if "calibration" not in calibration_json:
        raise ValueError("Calibration JSON missing 'calibration' section.")

    cal = calibration_json["calibration"]

    for field in REQUIRED_CALIBRATION_FIELDS:
        if field not in cal:
            raise ValueError(f"Calibration JSON missing required field: {field}")


def validate_run_directory(run_dir: Path):
    if not run_dir.exists():
        raise ValueError(f"Run directory does not exist: {run_dir}")

    pos_files = list(run_dir.glob("pos_*.jsonl"))
    neg_files = list(run_dir.glob("neg_*.jsonl"))
    cal_files = list(run_dir.glob("*report*.json"))

    if not pos_files:
        raise ValueError("No pos_*.jsonl files found.")

    if not neg_files:
        raise ValueError("No neg_*.jsonl files found.")

    if not cal_files:
        raise ValueError("No calibration JSON (*negcal*.json) found.")

    print(f"Found {len(pos_files)} positive files.")
    print(f"Found {len(neg_files)} negative files.")
    print(f"Found {len(cal_files)} calibration files.")

    # Validate each file
    for pos in pos_files:
        rows = load_jsonl(pos)
        validate_jsonl_structure(rows, pos.name)
        print(f"✓ {pos.name} valid ({len(rows)} rows)")

    for neg in neg_files:
        rows = load_jsonl(neg)
        validate_jsonl_structure(rows, neg.name)
        print(f"✓ {neg.name} valid ({len(rows)} rows)")

    for cal in cal_files:
        cal_json = load_json(cal)
        validate_calibration(cal_json)
        print(f"✓ {cal.name} valid calibration")

    print("\n✅ Artifacts validation PASSED.")


def main():
    run_dirs = sorted(Path("artifacts/runs").glob("*"))
    if not run_dirs:
        raise RuntimeError("No runs found.")

    run_dir = run_dirs[-1]

    parser = argparse.ArgumentParser(
        description="Validate Certum gate artifacts directory."
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=run_dir,
        help="Path to artifacts/<run_id> directory",
    )

    parser.add_argument("--artifacts_dir", default="artifacts")
    parser.add_argument("--cache_db", default=None)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if any ERRORs are found.")
    args = parser.parse_args()

    args = parser.parse_args()
    print(f"\n📊 Using run: {args.run}\n")
    try:
        validate_run_directory(args.run)
    except Exception as e:
        print(f"\n❌ Validation FAILED: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
