from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


MODEL_ID = os.getenv("MODEL_ID", "google/embeddinggemma-300m")

# Map HF_TOKEN -> HUGGINGFACE_HUB_TOKEN if only HF_TOKEN is set
if os.getenv("HF_TOKEN") and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


ENCODE_BATCH_SIZE = _env_int("ENCODE_BATCH_SIZE", 4)
MAX_SEQ_LENGTH = _env_int("MAX_SEQ_LENGTH", 512)
TORCH_NUM_THREADS = _env_int("TORCH_NUM_THREADS", 2)


class EmbeddingModel:
    def __init__(self) -> None:
        torch.set_num_threads(TORCH_NUM_THREADS)
        self.model = SentenceTransformer(MODEL_ID, device="cpu")
        # Default 2048 + encode batch_size=32 can spike CPU RAM into many GB; cap both.
        self.model.max_seq_length = MAX_SEQ_LENGTH
        self._encode_kwargs: dict[str, Any] = {
            "batch_size": ENCODE_BATCH_SIZE,
            "show_progress_bar": False,
        }

    @staticmethod
    def _truncate_and_renorm(emb: np.ndarray, dim: int) -> np.ndarray:
        vec = emb[..., :dim]
        norms = np.linalg.norm(vec, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vec / norms

    def embed_query(self, inputs: Sequence[str], dimension: int) -> np.ndarray:
        embeddings = self.model.encode(list(inputs), prompt_name="query", **self._encode_kwargs)
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
        return self._truncate_and_renorm(embeddings, dimension)

    def embed_document(self, inputs: Sequence[str], dimension: int) -> np.ndarray:
        embeddings = self.model.encode(list(inputs), prompt_name="document", **self._encode_kwargs)
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
        return self._truncate_and_renorm(embeddings, dimension)


_embedding_model: EmbeddingModel | None = None


def get_embedding_model() -> EmbeddingModel:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model
