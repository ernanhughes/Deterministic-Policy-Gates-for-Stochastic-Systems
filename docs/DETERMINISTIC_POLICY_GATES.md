## 1️⃣ Enhanced Worked Example (Drop-In Ready)

This example is designed to be:

* faithful to your actual pipeline
* auditable step-by-step
* impossible to misinterpret as “truth scoring”

You can paste this verbatim under a section like  
`## Worked Example: From Claim to Policy Decision`.

---

### 🧪 Worked Example: Deterministic Gating of a Single Claim

#### Input

**Claim**

> _“The 1951 Colgate Red Raiders football team had a total of four wins out of nine games.”_

**Evidence Spans (FEVEROUS)**

```
E₁: "The 1951 Colgate Red Raiders football team finished the season with a 4–5 record."
E₂: "Colgate competed as an independent during the 1951 college football season."
E₃: "The Red Raiders played nine games in the 1951 season."
```

---

### Step 1: Embedding

Each text span is embedded using a sentence transformer (e.g. `all-MiniLM-L6-v2`):

```python
claim_vec    = embed(claim)        # shape (d,)
evidence_vecs = embed(evidence)    # shape (n, d)
```

All vectors are unit-normalized to ensure cosine geometry.

---

### Step 2: Evidence Subspace Construction

1. Compute cosine similarity between the claim and each evidence vector.
2. Select top-K most similar evidence vectors (`K=12` or fewer).
3. Perform thin SVD on the selected evidence matrix:
    
    ```
    E_k = U · Σ · Vᵀ
    ```
    
4. Retain the top-R right singular vectors (`R=8`) to form an orthonormal basis:
    
    ```
    U_r ∈ ℝᵈˣʳ
    ```
    

This basis represents the **semantic span supported by the evidence**.

---

### Step 3: Hallucination Energy

Project the claim vector into the evidence subspace:

```
explained = ||U_rᵀ · c||²
energy    = 1 − explained
```

**Observed values**

```
explained = 0.7176
energy    = 0.2824
```

Interpretation:

* ~72% of the claim’s semantic mass lies inside the evidence span
* ~28% lies outside (unsupported)

This is **not** a truth judgment — only a measure of semantic support.

---

### Step 4: Oracle Calibration

To normalize scale, construct a **control (oracle) claim** known to be supported:

```python
oracle_claim = evidence_texts[0]
oracle_energy = hallucination_energy(oracle_claim, evidence)
```

**Observed**

```
oracle_energy ≈ 0.00000006
```

This confirms the embedding + evidence geometry is well-conditioned.

---

### Step 5: Energy Gap

Compute oracle-relative deviation:

```
energy_gap = claim_energy − oracle_energy
           ≈ 0.2824
```

This value answers the question:

> _“How much more unsupported semantic mass does this claim contain compared to a guaranteed-supported statement?”_

---

### Step 6: Policy Application

#### Fixed Policy (Standard)

```python
POLICIES = {
    "standard": 0.45
}
```

Decision:

```
energy = 0.2824 ≤ 0.45  →  ACCEPT
```

#### Adaptive Policy (P10)

From a 100-sample sweep:

```
P10 threshold τ ≈ 0.4559
```

Decision:

```
energy_gap = 0.2824 ≤ 0.4559 → ACCEPT
```

---

### Final Verdict

```json
{
  "verdict_fixed": "accept",
  "verdict_adaptive": "accept",
  "energy": 0.2824,
  "oracle_energy": 0.00000006,
  "energy_gap": 0.2824
}
```

---

### Why This Example Matters

* The claim is **specific and factual**
* The gate accepts it **without ever asserting truth**
* Acceptance occurs because **policy allows this degree of supported variation**
* Changing policy (e.g., `strict`) would deterministically change the outcome

This is **policy governance**, not model confidence.
