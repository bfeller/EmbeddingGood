from __future__ import annotations

from typing import List

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status

from .auth import require_api_key
from .ratelimit import rate_limiter
from .model import MODEL_ID, get_embedding_model
from .schemas import EmbedRequest, EmbedResponse, HealthResponse


app = FastAPI(title="EmbeddingGemma API", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model=MODEL_ID)


@app.post("/embed", response_model=EmbedResponse, dependencies=[Depends(require_api_key)])
def embed(body: EmbedRequest, response: Response, x_api_key: str | None = Header(default=None)) -> EmbedResponse:
    # No-store headers to prevent intermediary caching
    response.headers["Cache-Control"] = "no-store"

    # Rate limiting per API key
    key = x_api_key or ""
    ok, reset_sec, limit_per_min, remaining = rate_limiter.allow(key)
    response.headers["X-RateLimit-Limit"] = str(limit_per_min)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(int(reset_sec))
    if not ok:
        response.headers["Retry-After"] = str(max(1, int(reset_sec)))
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded. Try again later.")

    texts: List[str] = [body.input] if isinstance(body.input, str) else list(body.input)

    # Validate length per EmbeddingGemma constraints (<= 2000 chars per input)
    for t in texts:
        if len(t) > 2000:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Each input must be ≤ 2000 characters for EmbeddingGemma.")
    model = get_embedding_model()
    if body.type == body.type.query:
        vectors = model.embed_query(texts, body.dimension)
    else:
        vectors = model.embed_document(texts, body.dimension)

    return EmbedResponse(embeddings=vectors.tolist(), dimension=body.dimension)


