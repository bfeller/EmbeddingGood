# EmbeddingGemma API (Dockerized)

FastAPI service for `google/embeddinggemma-300m` embeddings.

Reference: [EmbeddingGemma on Hugging Face](https://huggingface.co/google/embeddinggemma-300m)

## Endpoints
- GET `/health` → `{ status, model }`
- POST `/embed` (requires `x-api-key`)
  - Body: `{ "input": string|string[], "type": "query"|"document", "dimension": 128|256|512|768 }`
  - Response: `{ embeddings: number[][], dimension: number }`

## Auth
- Header `x-api-key` must match one in `API_KEYS` (comma-separated env).

## Environment variables
- **API_KEYS**: Comma-separated list of allowed API keys used for `x-api-key` auth. Example: `API_KEYS="key1,key2"`.
- **HF_TOKEN**: Your Hugging Face access token (Required to pull `google/embeddinggemma-300m`, which is gated). This is automatically mapped to `HUGGINGFACE_HUB_TOKEN` at runtime.
- **MODEL_ID**: Optional model id. Default: `google/embeddinggemma-300m`.
- **TRANSFORMERS_CACHE / HF_HOME**: Optional cache locations for Hugging Face files. The Docker run command mounts a named volume at `/root/.cache/huggingface` so weights persist.
- **RL_REQUESTS_PER_MINUTE**: Per-key refill rate (default 120).
- **RL_BURST**: Per-key burst capacity (default 60).
- **ENCODE_BATCH_SIZE**: Sentence-transformers encode batch size (default 4). The library default of 32 can spike CPU RAM on long inputs.
- **MAX_SEQ_LENGTH**: Token cap passed to the model (default 512). Sufficient for the API's 2000-character limit without reserving 2048-token activations.
- **TORCH_NUM_THREADS** / **OMP_NUM_THREADS** / **MKL_NUM_THREADS**: CPU thread limits (default 2 in Docker).

### Getting a Hugging Face token and model access
1. Create/sign in to a Hugging Face account.
2. Visit the model page and accept the license/terms: [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m).
3. Generate a User Access Token: Profile → Settings → Access Tokens → New Token (read scope is sufficient). Copy the token.
4. Put the token string into `huggingface_temp_token.md` locally (kept out of git), or export it as an environment variable when running Docker.

### Generate API keys (Linux)
```
openssl rand -hex 32
# multiple keys example
API_KEYS="$(openssl rand -hex 32),$(openssl rand -hex 32)"
```

## Build and run (Compose)

```
cp .env.example .env   # set HF_TOKEN and API_KEYS
DOCKER_BUILDKIT=1 docker compose build
docker compose up -d
```

Published on host port **8010** by default (`8010:8000`).

## Build (standalone)
```
DOCKER_BUILDKIT=1 docker build -t embeddinggemma-api .
```

## Run (standalone)
```
docker volume create embedding_hf_cache
TOKEN=$(cat huggingface_temp_token.md)

docker run -p 8000:8000 --memory=3g -e API_KEYS="dev-key" -e HF_TOKEN="$TOKEN" -e MODEL_ID=google/embeddinggemma-300m -v embedding_hf_cache:/root/.cache/huggingface --name embeddinggemma-api embeddinggemma-api
```

## Test
```
# Health
curl http://localhost:8000/health

# Embed (document)
curl -X POST http://localhost:8000/embed -H 'content-type: application/json' -H 'x-api-key: dev-key' -d '{"input":"hello world","type":"document","dimension":768}'

# Embed (query 256d)
curl -X POST http://localhost:8000/embed -H 'content-type: application/json' -H 'x-api-key: dev-key' -d '{"input":["find red planet"],"type":"query","dimension":256}'
```

## Notes
- Float16 is not supported by EmbeddingGemma (we use float32).
- `HF_TOKEN` is mapped to `HUGGINGFACE_HUB_TOKEN` automatically.
- The service does not store requests or results. Intermediary caching is discouraged via `Cache-Control: no-store`.
- To fully clear model files, remove the HF cache volume: `docker volume rm embedding_hf_cache`.

