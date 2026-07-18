# FastAPI RAG Boilerplate

A pluggable RAG starter: swap the vector database or embedding provider from config without touching application code. Ships with a FastAPI API and a Streamlit UI over the same core.

## Project layout

```
config.yaml            # all non-secret settings
.env                   # secrets only (OPENROUTER_API_KEY, MONGO_URI)
main.py                # FastAPI entrypoint (uvicorn main:app)
streamlit_app.py       # Streamlit UI
docker-compose.yml     # qdrant + chroma + mongo services
rag/
  api.py               # FastAPI routes
  bootstrap.py         # wires everything together (AppContainer)
  config.py            # loads config.yaml + .env into Settings
  chunking.py, schemas.py
  embeddings/          # pluggable embedding providers (base + factory + one file each)
  vector_stores/       # pluggable vector DBs   (base + factory + one file each)
  services/            # ingestion, retrieval, parsing, prompt, Mongo registry
  routers/             # thin orchestration over the services
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Add your OpenRouter API key — embeddings are hosted, so this is **required** to ingest or query:
   ```bash
   cp sample.env .env     # then set OPENROUTER_API_KEY=sk-or-...
   ```
3. *(Optional)* start the backing services:
   ```bash
   docker compose up -d   # qdrant + chroma + mongo
   ```
4. Run either interface:
   ```bash
   streamlit run streamlit_app.py   # UI  -> http://localhost:8501
   uvicorn main:app --reload        # API -> http://127.0.0.1:8000/docs
   ```

Everything else is tuned in `config.yaml`.

## Configuration

Configuration is split in two:

- **`config.yaml`** — all non-secret settings (model ids, chunk sizes, vector DB backend + paths/hosts, embedding model, Mongo db/collection). Edit this file directly.
- **`.env`** — secrets only: `OPENROUTER_API_KEY` (required) and `MONGO_URI` (optional; enables the document registry). Copy `sample.env` to `.env` and fill in.

Values are read from `config.yaml`; the two secrets always come from the environment (`.env`), never from the YAML.

## Embeddings

Embeddings are generated via the **OpenRouter** hosted API (OpenAI-compatible) — no local embedding model is downloaded or loaded. Set `OPENROUTER_API_KEY` in `.env` and pick the model in `config.yaml`:

```yaml
embedding_backend: openrouter
openrouter_embedding_model: openai/text-embedding-3-small
embedding_dimension: 1536
```

`embedding_dimension` **must match the model's output** (`text-embedding-3-small` = 1536, `text-embedding-3-large` = 3072); a mismatch raises an error telling you the correct value. Changing the embedding model requires a fresh `collection_name` and a re-ingest — a collection is fixed to the dimension it was created with, and vectors from different models are not comparable.

## Vector database backends

The vector store is a small `VectorStore` interface (`rag/vector_stores/base.py`) with one implementation per backend, wired by a registry-based factory (`rag/vector_stores/factory.py`). Select one in `config.yaml`:

```yaml
vector_db_backend: qdrant   # qdrant | chroma | faiss
```

| Backend | `config.yaml` keys | Modes |
|---|---|---|
| `qdrant` | `qdrant_path`, `qdrant_url` | embedded file **or** server |
| `chroma` | `chroma_path`, `chroma_host`, `chroma_port` | embedded file **or** server |
| `faiss` | `faiss_path` | embedded only (a library, not a server) |

Each backend keeps its own independent storage, so switching backends doesn't affect data ingested under another — re-ingest to compare them side by side.

### Embedded vs server mode

- **Embedded** (default): leave `qdrant_url` / `chroma_host` empty. Persists to a local folder, no Docker needed. Note that embedded Qdrant takes an exclusive lock on its folder — only **one process at a time** (you can't run Streamlit and uvicorn together).
- **Server**: `docker compose up -d`, then point the config at it. Allows concurrent access, and Qdrant adds a web dashboard at <http://localhost:6333/dashboard> for browsing stored chunks.

  ```yaml
  qdrant_url: "http://localhost:6333"   # Qdrant server
  chroma_host: "localhost"              # Chroma server (chroma_port: 8001)
  ```

Chroma is mapped to host port **8001** (container 8000) to avoid colliding with uvicorn on 8000.

## Document registry (MongoDB)

Optional. When `MONGO_URI` is set in `.env` (`docker compose up -d mongo`), every ingested document is recorded with its file name, type, size, content hash, embedding model, and chunk count.

Deduplication key is **`(doc_hash, embedding_model, collection)`**:
- Same file + same embedding model → **skipped** (no re-embedding, no duplicate vectors).
- Same file + **different** embedding model → re-ingested, since the vectors must be recomputed.

Leave `MONGO_URI` empty to disable tracking; the app still runs. The Streamlit **Documents** tab lists everything in the registry.

## LLM

Answer generation uses a local Hugging Face seq2seq model (default `google/flan-t5-base`), configured in `config.yaml` via `local_model_id`, `local_llm_max_new_tokens`, and `local_llm_temperature`. The model downloads on first use (~1GB) and is cached locally; no API key needed.

## Endpoints

- `GET /health`
- `POST /ingest/text`
- `POST /ingest/file`
- `POST /query`

### Raw text ingestion — `POST /ingest/text`

Send raw text as JSON. The app chunks it, embeds each chunk via OpenRouter, and stores the chunks in the configured vector DB (recording the doc in Mongo if enabled).

```bash
curl -X POST "http://127.0.0.1:8000/ingest/text" \
  -H "Content-Type: application/json" \
  -d "{\"source_id\":\"doc-1\",\"raw_text\":\"RAG stands for Retrieval Augmented Generation.\"}"
```

### File ingestion — `POST /ingest/file`

Upload a file as multipart form data. The app parses it to text, chunks, embeds, and stores it.

Supported types: `.pdf` (via `pypdf`), `.docx` (via `python-docx`), `.txt` (UTF-8).

```bash
curl -X POST "http://127.0.0.1:8000/ingest/file" \
  -F "file=@sample.pdf" \
  -F "source_id=annual-report"
```

Both ingestion responses include `embedding_model`, `doc_hash`, and `skipped`.

### Query — `POST /query`

The app embeds the question with the same OpenRouter model, searches the configured vector DB for the top matches, builds a grounded prompt, and sends it to the local Hugging Face model. The response reports the `embedding_model` used, so you can spot a docs-vs-query model mismatch.

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What does RAG stand for?\"}"
```

## Extending it

The vector DB and embedding layers use the same registry/factory pattern, so adding a provider is a single new file — no changes to existing code:

- **New vector DB:** add `rag/vector_stores/<name>_store.py`, subclass `VectorStore`, decorate a builder with `@register("<name>")`, then import it in `rag/vector_stores/__init__.py`. Select it via `vector_db_backend`.
- **New embedding provider:** add `rag/embeddings/<name>.py`, subclass `Embedding`, decorate with `@register("<name>")`, then import it in `rag/embeddings/__init__.py`. Select it via `embedding_backend`.

## Notes

- The upload parser supports `.pdf`, `.docx`, and `.txt`. Old binary `.doc` files are not supported.
- Streamlit's dev file-watcher walks `transformers` and logs a wall of harmless `torchvision` import warnings. Silence them with `streamlit run streamlit_app.py --server.fileWatcherType none` (disables auto-reload on save).
