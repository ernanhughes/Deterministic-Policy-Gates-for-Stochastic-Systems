# src/certum/evaluation/runner.py

import argparse
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from certum.embedding.hf_embedder import HFEmbedder
from certum.embedding.sqlite_embedding_backend import SQLiteEmbeddingBackend
from certum.geometry.claim_evidence import ClaimEvidenceGeometry
from certum.geometry.nli_wrapper import EntailmentModel
from certum.geometry.sentence_support import SentenceSupportAnalyzer
from certum.orchestration.summarization_runner import SummarizationRunner

from .experiments import run_experiment, run_ablation
from .modeling import run_model_cv
from .plots import plot_roc, plot_precision_recall, plot_calibration
from .metrics import bootstrap_auc
from .feature_builder import extract_dataframe_from_results
from .modeling import run_xgb_model, run_xgb_model_cv
import numpy as np
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# =========================================================
# Evaluation Runner (Certum-style aligned)
# =========================================================

class EvaluationRunner:

    def run(
        self,
        *,
        input_jsonl: Path,
        embedding_model: str,
        embedding_db: Path,
        nli_model: str,
        top_k: int,
        limit: int,
        seed: int,
        dataset_name: str,
        out_dir: Path,
        entailment_db: str = "entailment_cache.db",
    ) -> None:

        run_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = out_dir / timestamp
        run_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting evaluation run {run_id}")
        logger.info(f"Run folder: {run_folder}")

        # =====================================================
        # 1️⃣ Load dataset
        # =====================================================

        samples = self._load_jsonl(input_jsonl)

        if limit:
            samples = samples[:limit]

        logger.info(f"Loaded {len(samples)} samples.")

        # =====================================================
        # 2️⃣ Build core objects (no hardcoding)
        # =====================================================

        backend = SQLiteEmbeddingBackend(str(embedding_db))
        print(f"Using embedding backend with DB: {embedding_db}")
        embedder = HFEmbedder(
            embedding_model,
            backend=backend,
        )

        energy_computer = ClaimEvidenceGeometry(
            top_k=1000,
            rank_r=32,
        )

        entailment_model = EntailmentModel(
            model_name=nli_model,
            batch_size=32,
            db_path=str(entailment_db),
        )

        support_analyzer = SentenceSupportAnalyzer(
            embedder=embedder,
            energy_computer=energy_computer,
            entailment_model=entailment_model,
            top_k=top_k,
        )

        summarization_runner = SummarizationRunner(
            support_analyzer=support_analyzer
        )

        results_path = run_folder / "summary_results.jsonl"

        logger.info("Running summarization pipeline...")
        results = summarization_runner.run(
            samples,
            out_path=results_path,
        )

        # =====================================================
        # 3️⃣ Feature Extraction
        # =====================================================

        df = extract_dataframe_from_results(results)

        logger.info(f"Extracted feature dataframe shape: {df.shape}")

        # =====================================================
        # 4️⃣ Write Config
        # =====================================================

        config = {
            "run_id": run_id,
            "dataset": dataset_name,
            "input_jsonl": str(input_jsonl),
            "n_samples": len(df),
            "embedding_model": embedding_model,
            "embedding_db": str(embedding_db),
            "nli_model": nli_model,
            "top_k": top_k,
            "limit": limit,
            "seed": seed,
        }

        self._write_json(run_folder / "config.json", config)

        # =====================================================
        # 5️⃣ Correlation Matrix
        # =====================================================

        corr = df.corr(numeric_only=True)
        corr.to_csv(run_folder / "feature_correlation.csv")

        # =====================================================
        # 6️⃣ Feature Sets
        # =====================================================

        feature_sets = self._build_feature_sets()

        results_summary = {}

        # =====================================================
        # 7️⃣ Run Experiments
        # =====================================================

        for name, features in feature_sets.items():

            logger.info(f"Running experiment: {name}")

            result = run_experiment(df, features, name)

            mean_boot, ci_low, ci_high = bootstrap_auc(
                result["y_test"],
                result["probs"],
            )

            mean_cv, std_cv = run_model_cv(df, features)

            print("Unique labels:", np.unique(result["y_test"]))

            # plot_roc(
            #     df,
            #     features,
            #     run_folder / f"roc_{name}.png",
            # )

            if name == "full":
                ap = plot_precision_recall(
                    result["y_test"],
                    result["probs"],
                    run_folder / "precision_recall.png",
                )

                plot_calibration(
                    result["y_test"],
                    result["probs"],
                    run_folder / "calibration.png",
                )
            else:
                ap = None

            results_summary[name] = {
                "auc": result["auc"],
                "bootstrap_mean_auc": mean_boot,
                "bootstrap_ci": [ci_low, ci_high],
                "cv_mean_auc": mean_cv,
                "cv_std_auc": std_cv,
                "average_precision": ap,
                "coefficients": result["coefficients"],
            }

        # =====================================================
        # 8️⃣ Ablation (Full Model)
        # =====================================================

        ablation = run_ablation(
            df,
            feature_sets["full"],
            remove_sets=[
                ["energy_gap"],
                ["high_energy_count"],
                ["energy_gap", "high_energy_count"],
            ],
        )

        results_summary["ablation"] = ablation


        xgb_auc, xgb_importance, y_test, probs = run_xgb_model(df, feature_sets["full"])

        xgb_cv_mean, xgb_cv_std = run_xgb_model_cv(df, feature_sets["full"])

        print(f"\nXGBoost AUC: {xgb_auc:.4f}")
        print(f"XGBoost CV AUC: {xgb_cv_mean:.4f} ± {xgb_cv_std:.4f}")


        # =====================================================
        # 9️⃣ Write Final Report
        # =====================================================

        report = {
            "run_id": run_id,
            "dataset": dataset_name,
            "n_samples": len(df),
            "results": results_summary,
            "xgboost_auc": xgb_auc,
            "xgboost_cv_mean": xgb_cv_mean,
            "xgboost_cv_std": xgb_cv_std,
            "xgboost_feature_importance": xgb_importance,
        }

        self._write_json(run_folder / "report.json", report)

        logger.info("Evaluation complete.")

    # =====================================================
    # Utilities
    # =====================================================

    def _write_json(self, path: Path, obj: dict):

        def convert(o):
            import numpy as np

            if isinstance(o, (np.float32, np.float64)):
                return float(o)
            if isinstance(o, (np.int32, np.int64)):
                return int(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError

        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=convert)

    def _load_jsonl(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def _build_feature_sets(self):

        features_geometry = [
            "mean_sim_top1",
            "min_sim_top1",
            "mean_sim_margin",
            "min_sim_margin",
            "mean_coverage",
            "min_coverage",
            "max_energy",
            "mean_energy",
            "p90_energy",
            "frac_above_threshold",
            "min_energy",
            "energy_gap",
            "high_energy_count",
        ]

        features_entailment = [
            "max_entailment",
            "mean_entailment",
            "min_entailment",
            "entailment_gap",
        ]

        features_full = (
            features_geometry
            + features_entailment
            + ["sentence_count", "paragraph_count"]
        )

        return {
            "geometry": features_geometry,
            "entailment": features_entailment,
            "full": features_full,
        }


# =========================================================
# CLI ENTRYPOINT
# =========================================================

def main():

    ap = argparse.ArgumentParser(description="Certum Summarization Evaluation")

    ap.add_argument("--input_jsonl", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--dataset_name", type=str, default="unknown")
    ap.add_argument("--embedding_model", type=str, required=True)
    ap.add_argument("--embedding_db", type=Path, required=True)
    ap.add_argument("--nli_model", type=str, required=True)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--entailment_db", type=Path, default=Path("entailment_cache.db"))

    args = ap.parse_args()

    runner = EvaluationRunner()

    runner.run(
        input_jsonl=args.input_jsonl,
        embedding_model=args.embedding_model,
        embedding_db=args.embedding_db,
        nli_model=args.nli_model,
        top_k=args.top_k,
        limit=args.limit,
        seed=args.seed,
        dataset_name=args.dataset_name,
        out_dir=args.out_dir,
        entailment_db=args.entailment_db,
    )


if __name__ == "__main__":
    main()
