# certum/validation/halueval_summary_diagnostics.py

import numpy as np
import pandas as pd
from tqdm import tqdm

from certum.utils.text_utils import split_into_paragraphs, split_into_sentences


class HaluEvalSummaryDiagnostics:

    def __init__(self, energy_computer, embedder):
        self.energy_computer = energy_computer
        self.embedder = embedder

    def evaluate(self, samples, out_csv_path=None):

        rows = []

        for s in tqdm(samples, desc="Processing summaries"):

            document_text = s["evidence"][0]
            paragraphs = split_into_paragraphs(document_text)
            evidence_vecs = self.embedder.embed(paragraphs)

            sentences = split_into_sentences(s["claim"])
            if not sentences:
                continue

            sentence_energies = []
            sentence_coverages = []
            sentence_sim_top1 = []
            sentence_sim_margin = []

            for sent in sentences:
                sent_vec = self.embedder.embed([sent])[0]

                result = self.energy_computer.compute(
                    claim_vec=sent_vec,
                    evidence_vecs=evidence_vecs,
                )

                sentence_energies.append(result.energy)

                sims = evidence_vecs @ sent_vec
                sims_sorted = np.sort(sims)[::-1]

                coverage = float(np.mean(sims > 0.3))
                sentence_coverages.append(coverage)

                if len(sims_sorted) >= 2:
                    sentence_sim_top1.append(float(sims_sorted[0]))
                    sentence_sim_margin.append(float(sims_sorted[0] - sims_sorted[1]))
                elif len(sims_sorted) == 1:
                    sentence_sim_top1.append(float(sims_sorted[0]))
                    sentence_sim_margin.append(0.0)
                else:
                    sentence_sim_top1.append(0.0)
                    sentence_sim_margin.append(0.0)

            sentence_energies = np.array(sentence_energies)
            sentence_coverages = np.array(sentence_coverages)
            sentence_sim_top1 = np.array(sentence_sim_top1)
            sentence_sim_margin = np.array(sentence_sim_margin)

            row = {
                "label": s["label"],
                "sentence_count": len(sentences),
                "paragraph_count": len(paragraphs),
                "summary_length_chars": len(s["claim"]),
                "doc_length_chars": len(document_text),

                # Energy aggregates
                "max_energy": float(np.max(sentence_energies)),
                "mean_energy": float(np.mean(sentence_energies)),
                "p90_energy": float(np.percentile(sentence_energies, 90)),
                "frac_above_05": float(np.mean(sentence_energies > 0.5)),

                # Coverage aggregates
                "mean_coverage": float(np.mean(sentence_coverages)),
                "min_coverage": float(np.min(sentence_coverages)),

                # Similarity aggregates
                "mean_sim_top1": float(np.mean(sentence_sim_top1)),
                "min_sim_top1": float(np.min(sentence_sim_top1)),
                "mean_sim_margin": float(np.mean(sentence_sim_margin)),
                "min_sim_margin": float(np.min(sentence_sim_margin)),
            }

            rows.append(row)

        df = pd.DataFrame(rows)

        if out_csv_path:
            df.to_csv(out_csv_path, index=False)

        return df
