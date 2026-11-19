<div align="center">

# DocFlow

**Upload a document. Get answers you can check.**

An asynchronous document-intelligence API: secure uploads, background AI processing, and
questions about your PDFs answered with page-level citations that are verified against the text.

[![Python](https://img.shields.io/badge/python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-4169E1?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Celery](https://img.shields.io/badge/Celery-5.5-37814A?logo=celery&logoColor=white)](https://docs.celeryq.dev/)
[![Docling](https://img.shields.io/badge/parser-Docling-6E56CF)](https://github.com/docling-project/docling)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-not%20specified-lightgrey)](#-license)

[Quick start](#-quick-start) ·
[Features](#-features) ·
[Architecture](#-architecture) ·
[Configuration](#-configuration) ·
[API reference](#-api-reference) ·
[Contributing](#-contributing)

</div>

---

> [!NOTE]
> DocFlow is under active development. Read the [known limitations](#-known-limitations) before
> using it with production or sensitive data.

## 💡 Why DocFlow?

Organizations receive valuable information in documents, but turning those files into usable data
is slow and repetitive. Every workflow needs the same plumbing: authenticated uploads, durable
storage, text extraction, long-running AI jobs, status tracking and result retrieval.

DocFlow puts that plumbing behind one API. Clients upload quickly while parsing, OCR and AI work
run in the background, and every answer about a document points to the page it came from, so it can
be checked instead of trusted.

## ✨ Features

| | |
|---|---|
| 📄 **Any document** | PDF, DOCX, JPEG, PNG and TIFF, read by [Docling](https://github.com/docling-project/docling) with layout, tables and OCR for scanned pages |
| ⚙️ **Background processing** | Summaries, invoice extraction and contract metadata as Celery jobs with status, retries and email notification |
| 🔎 **Grounded Q&A** | Ask questions about a PDF; every citation (page + quote) is checked against the page text |
| 🧠 **Retrieval for long documents** | Optional RAG with pgvector and Gemini embeddings for documents of hundreds of pages |
| 💬 **Multi-provider chat** | Streaming chat with conversations, OpenRouter, Gemini and Groq with automatic fallback |
| 🎙️ **Voice and images** | Speech to text (Whisper on Groq) and questions about images (Gemini) |
| 🛡️ **Secure uploads** | OWASP-style checks, magic-byte validation, EXIF stripping and optional ClamAV virus scanning |
| 🔐 **Accounts** | JWT auth, password reset by email, rate limits, per-user data isolation and daily token budgets |
| 📈 **Observability** | JSON logs with request ids, Sentry errors, traces and logs, `/health` for every dependency |
| ☁️ **S3 storage** | Original files in any S3-compatible store (MinIO locally) through boto3 |

## 🚀 Quick start

**Requirements:** Python 3.12+, Docker with Docker Compose, and an OpenAI API key.

```bash
git clone git@github.com:tuyishimehope/DOC-FLOW.git
cd DOC-FLOW
cp .env.example .env            # then fill in OPENAI_API_KEY and secret_key
```

**Option A: everything in Docker**

```bash
docker compose up --build -d
docker compose logs -f api worker
```

**Option B: dependencies in Docker, the app on your machine**

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

docker compose up -d postgres redis minio
alembic upgrade head

uvicorn app.main:app --env-file .env --reload          # terminal 1
celery -A app.tasks.celery_app worker --loglevel=INFO  # terminal 2
```

Check that it is up:

```bash
curl http://localhost:8000/health
# {"status":"ok","checks":{"database":"ok","redis":"ok","storage":"ok"}}
```

| Service | URL |
| --- | --- |
| API | http://localhost:8000 |
| Swagger UI / ReDoc | http://localhost:8000/docs · http://localhost:8000/redoc (not in production) |
| MinIO console | http://localhost:9001 |
| PostgreSQL / Redis | `localhost:5433` / `localhost:6379` |

Generate a signing secret with `python -c 'import secrets; print(secrets.token_hex(32))'`.

<details>
<summary><b>First run: Docling models</b></summary>

<br>

Docling downloads its layout, table and OCR models on first use (a few hundred MB, cached in
`~/.cache/docling`). The Docker image already contains them. To fetch them ahead of time locally:

```bash
docling-tools models download layout tableformer rapidocr
```

Set `DOCUMENT_OCR_ENABLED=false` to read only the text layer of PDFs (faster). No system OCR tools
are needed.

</details>

## 🏗 Architecture

```text
                 ┌──────────────► PostgreSQL + pgvector
                 │                users, documents, jobs, pages, chunks, conversations
Client ──► FastAPI
                 ├──────────────► S3 / MinIO (boto3)
                 │                original files
                 │
                 └──► Redis ──► Celery worker
                                  ├─ read the file with Docling
                                  ├─ summarize / extract with OpenAI
                                  ├─ embed chunks with Gemini (retrieval)
                                  └─ save status and result, email the owner
```

A processing request moves through `PENDING → QUEUED → PROCESSING → COMPLETED`, or ends in
`FAILED` / `CANCELLED`.

<details>
<summary><b>Tech stack</b></summary>

<br>

| Component | Role |
| --- | --- |
| FastAPI | HTTP API and OpenAPI documentation |
| SQLAlchemy + asyncpg | Async database access |
| PostgreSQL 16 + pgvector | Application data, page text and embeddings |
| Alembic | Database migrations |
| boto3 + MinIO | Object storage for uploaded files (any S3-compatible service works) |
| Redis | Celery broker and result backend, rate-limit counters |
| Celery | Background document processing and indexing |
| OpenAI | Summaries, invoice and contract extraction |
| Gemini, Groq, OpenRouter | Chat, document Q&A, vision, embeddings, transcription |
| Docling + RapidOCR | Reading PDF, DOCX and images, OCR for scanned pages |
| ClamAV (optional) | Virus scanning of uploads |
| Sentry (optional) | Errors, performance traces and logs |

</details>

<details>
<summary><b>Project structure</b></summary>

<br>

```text
app/
├── main.py              # FastAPI application and lifecycle
├── api/v1/              # Versioned HTTP routes (auth, chat, document, file, processing_request)
├── core/                # Settings, S3 client, logging, docs, Sentry, dependency checks
├── db/                  # Engines, sessions and dependencies
├── models/schema.py     # SQLAlchemy models
├── service/             # Auth, chat, documents, files, LLM clients
├── tasks/               # Celery app and tasks
├── templates/           # Email templates
└── utils/               # Docling parser, upload checks, virus scan, email helpers
migrations/              # Alembic migrations
evals/                   # Chat and document eval cases, saved runs
scripts/                 # Tools run by hand: run_evals.py, run_doc_evals.py
tests/                   # API tests and sample assets
```

</details>

## 📖 Usage

Sign up, log in and keep the token:

```bash
curl -X POST http://localhost:8000/api/v1/users/signup \
  -H 'Content-Type: application/json' \
  -d '{"first_name":"Demo","last_name":"User","email":"demo@example.com","password":"ExamplePassword123!"}'

TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/users/token \
  --data-urlencode 'username=demo@example.com' \
  --data-urlencode 'password=ExamplePassword123!' | jq -r .access_token)
```

Store a PDF and ask about it:

```bash
curl -X POST http://localhost:8000/api/v1/documents/qa \
  -H "Authorization: Bearer $TOKEN" -F 'file=@contract.pdf;type=application/pdf'
# {"document_id":5,"file_id":9,"name":"contract.pdf","pages":12}

curl -X POST http://localhost:8000/api/v1/documents/5/ask \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"question":"How long is the notice period?"}'
# {"answer":"30 days.","grounded":true,"citations":[{"page":4,"quote":"The notice period is 30 days."}], ...}
```

<details>
<summary><b>Process a document in the background (summary, invoice, contract)</b></summary>

<br>

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Authorization: Bearer $TOKEN" \
  -F 'file=@tests/assets/sample-pdf-invoice.pdf;type=application/pdf' \
  -F 'processing_type=INVOICE_EXTRACTION' \
  -F 'instructions=Use the totals from the last page.'      # optional
# {"document_id":1,"processing_request_id":1,"status":"QUEUED"}

curl "http://localhost:8000/api/v1/processing-requests/status/1" -H "Authorization: Bearer $TOKEN"
curl "http://localhost:8000/api/v1/processing-requests/result/1" -H "Authorization: Bearer $TOKEN"
```

Modes: `DOCUMENT_SUMMARY`, `INVOICE_EXTRACTION`, `CONTRACT_METADATA`.

Summary results are text. Invoice results are a validated object (dates, parties, currency, totals,
line items), contract results likewise (parties, dates, governing law, renewal and termination,
obligations). Missing scalar fields are `null`, missing lists are empty. `confidence_score` is
always `null`, because the model gives no calibrated confidence. A result that is not ready is 404.

```json
{"result":{"invoice_number":"INV-001","invoice_date":"2026-04-15","due_date":null,"vendor_name":"Example Ltd","customer_name":"Buyer Inc","currency":"USD","subtotal":100,"tax":10,"total":110,"line_items":[]},"confidence_score":null}
```

Download the original with `GET /api/v1/files/{file_id}` (`--output file.pdf`).

</details>

<details>
<summary><b>Ask about a PDF: follow-ups, one-off questions and citations</b></summary>

<br>

Send the `conversation_id` of an earlier answer to ask a follow-up. The last few questions and
answers go to the model with the pages. Leave it out to start a new conversation.

```bash
curl -X POST http://localhost:8000/api/v1/documents/5/ask \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"question":"And who pays during it?","conversation_id":8}'

curl http://localhost:8000/api/v1/documents/5/file \
  -H "Authorization: Bearer $TOKEN" --output contract.pdf   # the original, for a viewer
```

One question about one file, keeping nothing but the answer:

```bash
curl -X POST http://localhost:8000/api/v1/chat/document \
  -H "Authorization: Bearer $TOKEN" \
  -F 'file=@contract.pdf;type=application/pdf' \
  -F 'question=How long is the notice period?'
```

- A cited `page` is the number a PDF viewer shows, starting at 1. Only citations whose quote is
  really on that page are returned; `grounded` is `false` when none is.
- `GET /api/v1/chat/conversations/8` returns each answer with its `citations` and `grounded`.
- A document uploaded with `POST /api/v1/documents` has no stored pages and cannot be asked about
  (409); upload it again with `/documents/qa`.
- Deleting a conversation keeps the document; deleting a document keeps the conversations, but new
  questions about it get 404.
- With retrieval on, add `"mode": "chunks"` (or leave `auto`) once `GET /api/v1/documents/5` shows
  `"index_status": "READY"`. Start indexing by hand with `POST /api/v1/documents/5/index`.

</details>

<details>
<summary><b>Chat, voice and images</b></summary>

<br>

```bash
# Streamed chat (Server-Sent Events), remembered as a conversation
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"message":"What is the capital of France?"}'

# Speech to text, then send the text to the chat
curl -X POST http://localhost:8000/api/v1/chat/transcribe \
  -H "Authorization: Bearer $TOKEN" -F 'file=@question.ogg;type=audio/ogg'
# {"id":1,"text":"What is the capital of France?","model":"whisper-large-v3-turbo", ...}

# A question about an image
curl -X POST http://localhost:8000/api/v1/chat/image \
  -H "Authorization: Bearer $TOKEN" \
  -F 'file=@photo.jpg;type=image/jpeg' -F 'question=What is on the desk?'
```

- Audio formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm (checked by extension). A chat
  message is at most 4,000 characters, so show the transcript and send it when the user confirms.
- An image question and its answer are saved as a new conversation and count towards the daily
  token budget. Send the image again to ask about it again.
- Browsers need `fetch` with a stream reader for `/chat/stream`, because `EventSource` only
  supports GET.

</details>

## 🔧 Configuration

All settings are read from `.env` (see [`.env.example`](.env.example)). The essentials:

| Setting | Purpose |
| --- | --- |
| `secret_key` | Signs the JWTs. Use a long random value |
| `OPENAI_API_KEY` | Background processing (summaries, extraction) |
| `GEMINI_AI_API_KEY`, `GROQ_AI_API_KEY`, `OPENROUTER_API_KEY` | Chat, document Q&A, vision, embeddings, transcription. Empty = that feature is off |
| `DATABASE_*`, `broker_host`, `broker_backend` | PostgreSQL and Redis |
| `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET`, `MINIO_SECURE` | S3 storage. The endpoint is `host:port` or a full URL |
| `ENVIRONMENT` | `development`, `staging` or `production` (default) |

Inside Compose, use service names: `DATABASE_HOST=postgres`, `DATABASE_PORT=5432`,
`broker_host=redis://redis:6379/0`, `MINIO_ENDPOINT=minio:9000`.

<details>
<summary><b>Environments</b></summary>

<br>

`ENVIRONMENT` is `development`, `staging` or `production` (the default). Case does not matter, and
any other value stops the app at startup.

| | development | staging | production |
|---|---|---|---|
| API docs (`/docs`, `/redoc`, `/openapi.json`) | served | served | 404 |
| Values of failed SQL statements in errors | shown | hidden | hidden |

Every JSON log line carries the value as `"environment"`, and Sentry reports under it unless
`SENTRY_ENVIRONMENT` is set. SQL values are hidden outside development because they contain users'
messages and transcripts.

</details>

<details>
<summary><b>AI models and limits</b></summary>

<br>

```dotenv
OPENAI_MODEL=gpt-5.5
OPENAI_TIMEOUT_SECONDS=45
OPENAI_MAX_INPUT_CHARS=100000     # extracted text sent for processing
OPENAI_MAX_OUTPUT_TOKENS=2048

CHAT_MODEL=openrouter/free
CHAT_GEMINI_MODEL=gemini-3.1-flash-lite
CHAT_GROQ_MODEL=openai/gpt-oss-20b
CHAT_DAILY_TOKEN_BUDGET=200000    # per user per UTC day, all AI features; 0 = no limit
CHAT_RATE_LIMIT_PER_MINUTE=20
CHAT_MAX_CONCURRENT_STREAMS=2
```

The character and token limits bound request size but are not a fixed billing ceiling.

**Speech to text** (Groq Whisper): `TRANSCRIBE_MODEL`, `TRANSCRIBE_MAX_FILE_MB` (at most 25),
`TRANSCRIBE_TIMEOUT_SECONDS` (at most 120). The transcript is kept, the audio is not.

**Image questions** (Gemini): `VISION_MODEL`, `VISION_MAX_IMAGE_MB` (at most 14, because base64
adds a third and Gemini accepts 20 MB per request), `VISION_TIMEOUT_SECONDS`. Also limit the request
size in your reverse proxy (for example `client_max_body_size` in nginx).

**Document questions** (Gemini): `DOCUMENT_MODEL`, `DOCUMENT_QA_MAX_MB` (at most 14),
`DOCUMENT_QA_MAX_PAGES` (default 40, the model reads all of them for every question),
`DOCUMENT_QA_TIMEOUT_SECONDS` (at most 120). Encrypted, damaged or scanned PDFs are refused with 422.

> [!IMPORTANT]
> Text, images and audio are sent to hosted providers. On Gemini's free tier Google may use what
> you send to improve its products. Send only content you may share with a third party.

</details>

<details>
<summary><b>Long documents: retrieval with pgvector</b></summary>

<br>

With `DOCUMENT_RETRIEVAL_ENABLED=true` a stored PDF is cut into chunks of about 400 tokens (never
across a page), each embedded by Gemini. A question only sends the `DOCUMENT_RETRIEVAL_TOP_K`
nearest chunks to the model, so long PDFs become affordable (`DOCUMENT_RAG_MAX_PAGES` instead of
`DOCUMENT_QA_MAX_PAGES`), at the risk that the right chunk is not among the nearest.

```dotenv
DOCUMENT_RETRIEVAL_ENABLED=false
DOCUMENT_RETRIEVAL_TOP_K=5
DOCUMENT_RAG_MAX_PAGES=300
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_BATCH_SIZE=50
EMBEDDING_TIMEOUT_SECONDS=60
```

- **Needs pgvector.** The migration runs `CREATE EXTENSION IF NOT EXISTS vector`. Compose uses
  `pgvector/pgvector:pg16-trixie` (same Debian as `postgres:16`, so an existing volume keeps its
  collation; avoid the plain `pg16` tag). On Neon, run `alembic upgrade head` as the database
  owner. The test database needs the extension too.
- **Needs the worker.** Storing a PDF queues `index_document_task`; `index_status` goes `NONE`,
  `PROCESSING`, `READY` or `FAILED`. Index older or failed documents with `POST /{id}/index`.
- **`mode` on `/ask`.** `auto` (default) uses chunks when the document is `READY`, the whole text
  otherwise; `pages` and `chunks` force one way. A document longer than `DOCUMENT_QA_MAX_PAGES`
  can only be asked in `chunks` mode (409 until it is `READY`).
- **Vectors belong to a model.** After changing `EMBEDDING_MODEL`, index every document again.
- **Privacy.** Logs and saved messages carry chunk ids, pages and distances, never the text. Every
  chunk carries the owner's `user_id` and retrieval filters on it.

Measure before relying on it: compare both modes on a PDF of your own (60 to 100 pages) and about 15
questions, three of them unanswerable (`evals/documents.example.json`, `evals/README.md`):

```bash
# server: DOCUMENT_QA_MAX_PAGES=120 DOCUMENT_RETRIEVAL_ENABLED=true, worker running
python -m scripts.run_doc_evals --email you@example.com --password '...' --cases evals/documents.json
```

</details>

<details>
<summary><b>Upload security and virus scanning</b></summary>

<br>

Every upload is checked before use, following the OWASP File Upload Cheat Sheet: an extension
allowlist, a size limit while reading, magic bytes that must match the extension, a 40-megapixel
limit for images, and images are decoded and written again (which strips EXIF, text chunks and
hidden trailing bytes). The browser's content type is ignored. The checks live in
`app/utils/upload.py` and `app/utils/image.py`. Not done yet: removing active content from PDF or
Office files.

Optional [ClamAV](https://www.clamav.net/) scanning runs before a file is parsed or stored:

```dotenv
CLAMAV_HOST=clamav      # service name in Compose; localhost if the API runs outside Docker
CLAMAV_PORT=3310
CLAMAV_TIMEOUT_SECONDS=30
```

```bash
docker compose --profile scan up -d clamav   # first start downloads the database (a few minutes)
```

- Empty `CLAMAV_HOST` = off (one warning at the first upload).
- Once set, the scan is required: an unreachable scanner gives 503, never an unscanned upload.
- An infected file gets 422; the signature is logged, not returned.
- ClamAV needs 3 to 4 GiB of memory, too much for the smallest servers. Files are streamed with
  `INSTREAM`, never written to disk.

</details>

<details>
<summary><b>Logging</b></summary>

<br>

The API and the worker write one JSON line per record to **stdout**, so Docker collects them:

```bash
docker compose logs -f api worker
docker compose logs api | jq 'select(.request_id == "<id>")'   # one request from start to end
```

| Setting | Default | Meaning |
|---|---|---|
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |
| `LOG_QUEUE` | `true` | `false` writes directly, useful when debugging a crash |
| `LOG_QUEUE_SIZE` | `10000` | Lines that may wait; beyond that the newest are dropped and counted |

- **A queue keeps logging off the request path.** A slow or stuck stdout cannot block requests.
  The Celery worker writes directly, because its forked children would not have the writer thread.
- **Every request has an id**, from the `X-Request-ID` header or generated, returned in the
  response and attached to every line as `request_id`.
- **One line per request** (`"Request finished"`) with `method`, the route template as `path`,
  `status` and `latency_ms`. It replaces `uvicorn.access`, which logs full URLs.
- **Only whitelisted fields** from `extra` are printed (`context_fields` in
  `app/core/logging_config.py`), so questions, answers and file names never reach the logs.
- Every line carries `environment`.

</details>

<details>
<summary><b>Monitoring with Sentry</b></summary>

<br>

Off while `SENTRY_DSN` is empty. To turn it on, create a Python project in Sentry and set:

```dotenv
SENTRY_DSN=https://<key>@<org>.ingest.sentry.io/<project>
SENTRY_ENVIRONMENT=        # empty = the same as ENVIRONMENT
SENTRY_TRACES_SAMPLE_RATE=0.1
SENTRY_ENABLE_LOGS=true
SENTRY_LOGS_LEVEL=INFO
```

- **Reported:** unhandled exceptions and HTTP 500 from the API and the worker, and "every chat
  provider failed". Provider outages answered with 502/503/504 and automatic failovers are not.
- **Logs:** application logs go to Sentry Logs with the same fields as the JSON line; ERROR logs
  also become events.
- **Never sent:** request bodies, local variables, prompts and answers, cookies, the
  `Authorization` header, email and IP addresses. Only the numeric user id is attached.
- Nothing arriving? Set `SENTRY_DEBUG=true`; the startup log says `Sentry is on` or `Sentry is off`.

</details>

<details>
<summary><b>Rate limits, emails and evals</b></summary>

<br>

- Login allows 10 attempts per email every 5 minutes and forgot-password 3 emails per 15 minutes
  (429 beyond that). Counters live in Redis; `rate_limit_enabled=false` turns them off.
- When processing finishes or finally fails, the owner gets an email (`processing_email_enabled`).
  SMTP is set with `mail_server`, `mail_port`, `mail_username`, `mail_password`, `mail_from`,
  `mail_use_tls`.
- Every streamed answer stores `provider`, token counts and `latency_ms`; `GET /api/v1/chat/stats`
  sums them per day. An empty value means the provider did not report it, not zero.

After changing a prompt (`app/prompts/chat_*.md`) or a model setting, run the evals against a
running API and compare with the previous run:

```bash
python -m scripts.run_evals --email you@example.com --password '...'   # or --token / EVALS_TOKEN
python -m scripts.run_evals --compare evals/results/<old>.json evals/results/<new>.json
```

`--model groq` tests one provider, `--min-score 0.9` fails below 90%. About ten real model calls per
run.

</details>

## 📚 API reference

The full interactive schema is at [`/docs`](http://localhost:8000/docs) outside production. List
endpoints take `skip` and `limit` (at most 50) and return newest first.

<details>
<summary><b>System and accounts</b></summary>

<br>

| Method | Endpoint | Auth | Description |
| --- | --- | --- | --- |
| GET | `/`, `/start` | No | Liveness |
| GET | `/health` | No | PostgreSQL, Redis and storage; 503 if any is down |
| POST | `/api/v1/users/signup` | No | Register |
| POST | `/api/v1/users/token` | No | Log in (OAuth2 form: the email goes in `username`) |
| GET | `/api/v1/users/me` | Yes | The current user |
| GET | `/api/v1/users/me/stats` | Yes | Dashboard counts: documents, requests by status and type, recent failures |
| PATCH | `/api/v1/users/{id}` | Yes | Update your profile |
| DELETE | `/api/v1/users/{id}` | Yes | Delete your account |
| POST | `/api/v1/users/forgot-password` | No | Request a reset email |
| POST | `/api/v1/users/reset-password` | No | Reset with a token |
| PATCH | `/api/v1/users/me/password` | Yes | Change your password |

</details>

<details>
<summary><b>Documents and files</b></summary>

<br>

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/api/v1/documents` | Upload and queue a document for processing |
| GET | `/api/v1/documents` | List; filter with `q` (name) and `status` (`ACTIVE` / `ARCHIVED`) |
| GET | `/api/v1/documents/{id}` | Metadata, including `index_status` |
| PATCH | `/api/v1/documents/{id}` | Rename and/or archive: `{"name": "..."}`, `{"status": "ARCHIVED"}` |
| POST | `/api/v1/documents/{id}/process` | Process again: `{"processing_type": "...", "instructions": "..."}` |
| DELETE | `/api/v1/documents/{id}` | Soft-delete; removes the stored file and page text |
| GET | `/api/v1/documents/{id}/jobs` | Processing attempts, with a readable `failure_reason` |
| GET | `/api/v1/documents/{id}/processing-requests` | The document's requests, newest first |
| POST | `/api/v1/documents/qa` | Store a PDF for questions → `{"document_id", "file_id", "name", "pages", "index_status"}`. 413 too big or too many pages, 415 wrong type, 422 empty, encrypted, damaged, scanned or infected |
| POST | `/api/v1/documents/{id}/ask` | Ask a stored PDF: `{"question", "conversation_id"?, "mode"?}` (`auto`, `pages`, `chunks`). 404 not yours, 409 not stored / not `READY` / other embedding model, 429 limited or over budget, 502/503/504 provider problems |
| POST | `/api/v1/documents/{id}/index` | Chunk and embed in the background (202). 409 when retrieval is off, nothing is stored or it is already `PROCESSING` |
| GET | `/api/v1/documents/{id}/file` | The original file, owner only |
| GET | `/api/v1/files` | List your files |
| GET | `/api/v1/files/{id}` | Download a file |
| DELETE | `/api/v1/files/{id}` | Soft-delete the file and its document, then remove the object |

All require authentication.

</details>

<details>
<summary><b>Processing requests</b></summary>

<br>

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/api/v1/processing-requests/{id}` | A processing request |
| GET | `/api/v1/processing-requests/status/{id}` | Its status |
| GET | `/api/v1/processing-requests/result/{id}` | Its result (404 until ready) |
| GET | `/api/v1/processing-requests/{id}/export?format=json\|csv` | Download the result |
| POST | `/api/v1/processing-requests/{id}/cancel` | Cancel one that has not started (409 otherwise) |
| POST | `/api/v1/processing-requests/{id}/retry` | Re-queue a failed or cancelled one (409 otherwise) |

All require authentication.

</details>

<details>
<summary><b>Chat, voice and vision</b></summary>

<br>

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/api/v1/chat` | One question: `{"message", "model"?}` (1 to 4,000 characters). 422 invalid, 429 limited, 503/504 provider problems |
| GET | `/api/v1/chat/models` | Accepted `model` values: `auto` (tries every configured provider) plus each configured provider |
| POST | `/api/v1/chat/stream` | Streamed and remembered: `{"message", "conversation_id"?, "model"?}`. SSE events `conversation`, `token`, then `done` or `error` |
| GET | `/api/v1/chat/stats` | Your usage over `days` (1 to 90, default 7): tokens per day, p50/p95 latency, error rate |
| POST | `/api/v1/chat/transcribe` | Speech to text → `{"id", "text", "model", "created_at"}`. 413 too big, 415 wrong extension, 422 empty |
| GET | `/api/v1/chat/transcribe/{id}` | One of your transcriptions |
| POST | `/api/v1/chat/image` | Ask about an image (jpg, jpeg, png, webp) → `{"conversation_id", "answer", "model", tokens}` |
| POST | `/api/v1/chat/document` | Ask about a PDF without keeping it → `{"answer", "found", "grounded", "citations", "pages", ...}` |
| POST | `/api/v1/chat/conversations` | Start an empty conversation: `{"title"?}` |
| GET | `/api/v1/chat/conversations` | Your conversations, most recently active first |
| GET | `/api/v1/chat/conversations/{id}` | One conversation with its messages, citations and `document_id` |
| PATCH | `/api/v1/chat/conversations/{id}` | Rename: `{"title": "..."}` |
| DELETE | `/api/v1/chat/conversations/{id}` | Delete it and its messages (204) |

All require authentication.

</details>

## 🛠 Development

<details>
<summary><b>Code style</b></summary>

<br>

[PEP 8](https://peps.python.org/pep-0008/), checked and formatted by
[Ruff](https://docs.astral.sh/ruff/) (pinned in `requirements-dev.txt`, configured in
`pyproject.toml`):

```bash
pip install -r requirements-dev.txt
ruff check .              # add --fix to apply the safe fixes
ruff format .
ruff format --check .     # what CI would run
```

Lines are at most 99 characters. `migrations/` is excluded because Alembic generates it. Class names
such as `Processing_Request` in `app/models/schema.py` break CapWords on purpose: Postgres names its
enum types after them, so renaming needs a migration.

</details>

<details>
<summary><b>Database migrations</b></summary>

<br>

```bash
alembic current
alembic upgrade head
alembic revision --autogenerate -m "describe the schema change"   # after changing a model
```

Always review autogenerated migrations before applying or committing them.

</details>

<details>
<summary><b>Docker Compose</b></summary>

<br>

```bash
docker compose up --build -d
docker compose logs -f api worker
docker compose down              # keeps data; --volumes deletes the Postgres and MinIO volumes
```

The API entrypoint waits for PostgreSQL, Redis and storage (at most
`DEPENDENCY_STARTUP_TIMEOUT_SECONDS`, 60 by default) before running migrations and starting
Uvicorn. The image bakes in Docling's models and uses CPU-only PyTorch.

</details>

<details>
<summary><b>Testing</b></summary>

<br>

pytest with HTTPX, AnyIO and a real PostgreSQL test database. Storage, task submission and AI
providers are mocked.

> [!WARNING]
> Use a separate test database. The fixture creates and drops every table, and refuses to run
> against the application database.

```bash
# .env or the environment; the database needs the pgvector extension
DATABASE_URL_TEST=postgresql+asyncpg://postgres:postgres@localhost:5433/test_docflow

python -m pytest -q
python -m pytest -m slow      # also run the real Docling (downloads models on the first run)
```

</details>

## 🤝 Contributing

Contributions are welcome. Before opening a pull request:

1. Work on a focused branch.
2. Keep routes, services and persistence in their existing layers.
3. Add an Alembic migration for every model change.
4. Add or update tests, and run `pytest` against a separate test database.
5. Run `ruff check .` and `ruff format --check .`.
6. Update this README when configuration, endpoints or requirements change.
7. Explain the motivation, the behavior change and how you verified it.

Never commit `.env`, API keys, SMTP credentials or uploaded documents. Report security issues to the
repository owner privately, not in a public issue.

## 🚧 Known limitations

- Replace every placeholder secret in `.env.example` before deploying.
- Input and output limits bound AI requests but do not guarantee a fixed cost.
- Uploads default to a 10 MiB maximum (`MAX_UPLOAD_SIZE_BYTES`).
- Large or complex PDFs can need a lot of worker memory and time, especially with OCR.
- Active content in PDF and Office files is not removed yet.
- Celery retries a failed task up to three times; every attempt is recorded.

## 📄 License

No license file is included yet. Until the owner adds one, treat the project as all rights
reserved: public source does not grant permission to use, modify or redistribute it.

---

<div align="center">

Built with [FastAPI](https://fastapi.tiangolo.com/), [Docling](https://github.com/docling-project/docling)
and [pgvector](https://github.com/pgvector/pgvector).

</div>
