from __future__ import annotations

import os
from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_ID = os.getenv("MODEL_ID", "google/embeddinggemma-300m")

# Map HF_TOKEN -> HUGGINGFACE_HUB_TOKEN if only HF_TOKEN is set
if os.getenv("HF_TOKEN") and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]


class EmbeddingModel:
    def __init__(self) -> None:
        # Ensure HF token is picked up from env if provided (HF_TOKEN)
        # sentence-transformers/transformers will use it automatically.
        self.model = SentenceTransformer(MODEL_ID, device="cpu")

    @staticmethod
    def _truncate_and_renorm(emb: np.ndarray, dim: int) -> np.ndarray:
        vec = emb[..., :dim]
        norms = np.linalg.norm(vec, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vec / norms

    def embed_query(self, inputs: Sequence[str], dimension: int) -> np.ndarray:
        embeddings = self.model.encode(list(inputs), prompt_name="query")
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
        return self._truncate_and_renorm(embeddings, dimension)

    def embed_document(self, inputs: Sequence[str], dimension: int) -> np.ndarray:
        embeddings = self.model.encode(list(inputs), prompt_name="document")
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
        return self._truncate_and_renorm(embeddings, dimension)


_embedding_model: EmbeddingModel | None = None


def get_embedding_model() -> EmbeddingModel:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model


