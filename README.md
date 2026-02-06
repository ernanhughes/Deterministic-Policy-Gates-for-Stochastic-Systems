# Deterministic-Policy-Gates-for-Stochastic-Systems# Deterministic Policy Gates for Stochastic Systems

This repository demonstrates how stochastic AI systems can be constrained,
audited, and evaluated using **deterministic policy gates**.

We apply the approach to AI-generated claims evaluated against the
**FEVEROUS development dataset**, focusing on:

- semantic drift
- hallucination energy
- policy-bounded verification
- fail-open safety guarantees

This is **not** a leaderboard system.
This is a **verifiability system**.

---

## Quickstart

```bash
python scripts/download_feverous.py
python -m verity_gate.run_eval
```

Results are written to ```artifacts/```.

## Repo Structure

src/verity_gate/ – core implementation
configs/ – experiment + policy configuration
datasets/ – FEVEROUS dev set
artifacts/ – outputs
paper/ – markdown paper + figures