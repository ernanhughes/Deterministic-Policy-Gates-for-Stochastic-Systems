# certum/validation/halueval_validator.py
import re

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from certum.utils.text_utils import split_into_paragraphs, split_into_sentences


class HaluEvalEnergyValidator:
    """
    Pure validation:
    Can hallucination energy separate hallucinated vs non-hallucinated responses?
    """

    def __init__(self, energy_computer, embedder):
        self.energy_computer = energy_computer
        self.embedder = embedder

    def _ensure_vectors(self, samples):
        for s in tqdm(samples, desc="Embedding samples"):
            if "claim_vec" not in s:
                s["claim_vec"] = self.embedder.embed([s["claim"]])[0]

            document_text = s["evidence"][0]

            paragraphs = split_into_paragraphs(document_text)
            s["evidence_vecs"] = self.embedder.embed(paragraphs)

    def evaluate(self, samples):
        self._ensure_vectors(samples)

        labels = []
        axis_values = {}

        axis_names = [
            "max_energy",
            "mean_energy",
            "p90_energy",
            "frac_above_05",
            "mean_coverage",
            "min_coverage",
            "combined_score",
        ]

        for name in axis_names:
            axis_values[name] = []

        for s in tqdm(samples, desc="Computing sentence geometry"):

            sentences = split_into_sentences(s["claim"])
            if not sentences:
                continue

            sentence_energies = []
            sentence_coverages = []

            for sent in sentences:
                sent_vec = self.embedder.embed([sent])[0]

                result = self.energy_computer.compute(
                    claim_vec=sent_vec,
                    evidence_vecs=s["evidence_vecs"],
                )

                sentence_energies.append(result.energy)

                # Coverage metric
                sims = s["evidence_vecs"] @ sent_vec
                coverage = float(np.mean(sims > 0.3))
                sentence_coverages.append(coverage)

            sentence_energies = np.array(sentence_energies)
            sentence_coverages = np.array(sentence_coverages)

            # --- Aggregation ---
            max_energy = float(np.max(sentence_energies))
            mean_energy = float(np.mean(sentence_energies))
            p90_energy = float(np.percentile(sentence_energies, 90))
            frac_above_05 = float(np.mean(sentence_energies > 0.5))

            mean_coverage = float(np.mean(sentence_coverages))
            min_coverage = float(np.min(sentence_coverages))

            labels.append(s["label"])

            axis_values["max_energy"].append(max_energy)
            axis_values["mean_energy"].append(mean_energy)
            axis_values["p90_energy"].append(p90_energy)
            axis_values["frac_above_05"].append(frac_above_05)
            axis_values["mean_coverage"].append(mean_coverage)
            axis_values["min_coverage"].append(min_coverage)

            # Combined deterministic score
            combined_score = max_energy * (1 - mean_coverage)
            axis_values["combined_score"].append(combined_score)

        labels = np.array(labels)

        results = {}

        for axis, values in axis_values.items():
            values = np.array(values, dtype=float)

            if len(np.unique(labels)) < 2:
                continue

            try:
                auc = roc_auc_score(labels, values)
            except Exception:
                auc = 0.5

            mean_pos = values[labels == 1].mean()
            mean_neg = values[labels == 0].mean()

            std_pos = values[labels == 1].std()
            std_neg = values[labels == 0].std()
            pooled = np.sqrt((std_pos**2 + std_neg**2) / 2)
            effect = (mean_pos - mean_neg) / pooled if pooled > 0 else 0.0

            results[axis] = {
                "auc": float(auc),
                "mean_pos": float(mean_pos),
                "mean_neg": float(mean_neg),
                "effect_size": float(effect),
            }

        return results
