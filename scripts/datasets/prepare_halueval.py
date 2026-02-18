#!/usr/bin/env python3

"""
Extract HaluEval summarization subset and convert
into Certum-compatible JSONL format.

Output format:
{
    "claim": str,
    "evidence": [str],
    "label": 0 or 1  # 1 = hallucinated
}
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="halueval_summarization.jsonl")
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    print("Loading HaluEval summarization dataset...")
    dataset = load_dataset("pminervini/HaluEval", "summarization")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with out_path.open("w", encoding="utf-8") as f:
        for row in dataset["data"]:

            if args.max_rows and count >= args.max_rows:
                break

            document = row.get("document")
            right_summary = row.get("right_summary")
            hallucinated_summary = row.get("hallucinated_summary")

            if not document:
                continue

            # Create NON-HALLUCINATED sample
            if right_summary:
                sample = {
                    "claim": right_summary.strip(),
                    "evidence": [document.strip()],
                    "label": 0
                }
                f.write(json.dumps(sample) + "\n")
                count += 1

            # Create HALLUCINATED sample
            if hallucinated_summary:
                sample = {
                    "claim": hallucinated_summary.strip(),
                    "evidence": [document.strip()],
                    "label": 1
                }
                f.write(json.dumps(sample) + "\n")
                count += 1

    print(f"Wrote {count} samples to {out_path}")


if __name__ == "__main__":
    main()
