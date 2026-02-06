# src/verity_gate/embedder.py
from __future__ import annotations

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


class HFEmbedder:
    def __init__(self, model_name: str, device: str = "cpu"):
        log.info("Loading embedding model: %s (device=%s)", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        log.info("Embedding dim: %d", self.dim)

    def embed_one(self, text: str) -> np.ndarray:
        v = self.model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(v, dtype=np.float32)

    def embed_many(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        M = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(M, dtype=np.float32)
