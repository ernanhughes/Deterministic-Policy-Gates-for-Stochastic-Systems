import pandas as pd
import json
import random
from pathlib import Path
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
INPUT_CSV = "E:/data/MAUD_train.csv"
OUTPUT_DIR = Path("artifacts/maud")
N_SAMPLES = 300000
SEED = 1337

random.seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------

def build_claim(question, answer):
    """
    Turn structured label into natural claim.
    """
    return f"The {question.lower()} in this contract is {answer}."


def split_evidence(text):
    """
    Split contract text into evidence spans.
    """
    # crude but effective
    sents = text.replace("\n", " ").split(". ")
    return [s.strip() for s in sents if len(s.strip()) > 30]


# -----------------------------
# Main Conversion
# -----------------------------

df = pd.read_csv(INPUT_CSV)

# Filter to main rows only
df = df[df["data_type"] == "main"]

# Drop missing
df = df.dropna(subset=["text", "answer", "question"])

# Sample
df = df.sample(min(N_SAMPLES, len(df)), random_state=SEED)

all_answers = df["answer"].unique().tolist()

pos_records = []
neg_records = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    text = row["text"]
    answer = row["answer"]
    question = row["question"]

    evidence = split_evidence(text)
    if len(evidence) < 2:
        continue

    # -------- Positive --------
    claim_pos = build_claim(question, answer)

    pos_records.append({
        "claim": claim_pos,
        "evidence": evidence,
        "label": 1
    })

    # -------- Hard Negative --------
    wrong_answer = random.choice([a for a in all_answers if a != answer])
    claim_neg = build_claim(question, wrong_answer)

    neg_records.append({
        "claim": claim_neg,
        "evidence": evidence,
        "label": 0
    })


# -----------------------------
# Save JSONL
# -----------------------------

with open(OUTPUT_DIR / "pos_maud.jsonl", "w") as f:
    for r in pos_records:
        f.write(json.dumps(r) + "\n")

with open(OUTPUT_DIR / "neg_maud.jsonl", "w") as f:
    for r in neg_records:
        f.write(json.dumps(r) + "\n")

print("Done.")
print("POS:", len(pos_records))
print("NEG:", len(neg_records))
