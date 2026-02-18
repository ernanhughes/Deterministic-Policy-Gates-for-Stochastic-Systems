import json

from datasets import load_dataset

dataset = load_dataset("kiddothe2b/contract-nli", "contractnli_a")

out_pos = []
out_neg = []

for split in ["train", "validation", "test"]:
    for example in dataset[split]:
        claim = example["hypothesis"]
        evidence = example["premise"]
        label = example["label"]

        record = {
            "claim": claim,
            "evidence": [evidence],
            "label": label,
        }

        # treat entailment as positive
        if label == 1:
            out_pos.append(record)
        else:
            out_neg.append(record)

def write_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")

write_jsonl(out_pos, "contract_nli_pos.jsonl")
write_jsonl(out_neg, "contract_nli_neg.jsonl")
