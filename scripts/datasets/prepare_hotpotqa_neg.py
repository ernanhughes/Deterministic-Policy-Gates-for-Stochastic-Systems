import json

from datasets import load_dataset

dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki")

out_file = []

for split in ["train", "validation", "test"]:
    for example in dataset[split]:
        claim = example["question"]
        evidence = example["context"]
        label = "pos"

        record = {
            "claim": claim,
            "evidence": [evidence],
            "label": label,
        }

        out_file.append(record)

def write_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")

write_jsonl(out_file, "datasets/hotpot/hotpot.jsonl")
