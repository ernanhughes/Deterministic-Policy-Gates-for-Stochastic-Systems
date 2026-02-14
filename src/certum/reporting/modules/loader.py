import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover_modes(run_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Discover all adversarial modes inside a run directory.
    Returns:
        {
            "deranged": {
                "pos": Path,
                "neg": Path,
                "calibration": Path
            },
            ...
        }
    """
    modes = {}

    for pos_file in run_dir.glob("pos_*.jsonl"):
        mode = pos_file.stem.replace("pos_", "")
        neg_file = run_dir / f"neg_{mode}.jsonl"

        cal_files = list(run_dir.glob(f"*{mode}*.json"))
        cal_file = None
        for cf in cal_files:
            if "negcal" in cf.name:
                cal_file = cf
                break

        if neg_file.exists() and cal_file:
            modes[mode] = {
                "pos": pos_file,
                "neg": neg_file,
                "calibration": cal_file
            }

    return modes
