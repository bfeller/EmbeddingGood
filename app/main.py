from __future__ import annotations

from typing import List

from fastapi import Depends, FastAPI

from .auth import require_api_key
from .model import MODEL_ID, get_embedding_model
from .schemas import EmbedRequest, EmbedResponse, HealthResponse


app = FastAPI(title="EmbeddingGemma API", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model=MODEL_ID)


@app.post("/embed", response_model=EmbedResponse, dependencies=[Depends(require_api_key)])
def embed(body: EmbedRequest) -> EmbedResponse:
    texts: List[str] = [body.input] if isinstance(body.input, str) else list(body.input)
    model = get_embedding_model()
    if body.type == body.type.query:
        vectors = model.embed_query(texts, body.dimension)
    else:
        vectors = model.embed_document(texts, body.dimension)

    return EmbedResponse(embeddings=vectors.tolist(), dimension=body.dimension)


