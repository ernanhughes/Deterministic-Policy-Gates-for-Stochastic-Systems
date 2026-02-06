#!/usr/bin/env bash
set -e

python scripts/download_feverous.py
python -m verity_gate.run_eval
