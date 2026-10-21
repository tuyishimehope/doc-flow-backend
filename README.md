# DocFlow Backend

An asynchronous document-processing API for uploading files and turning their contents into useful AI-generated output.

DocFlow combines FastAPI, PostgreSQL, MinIO, Redis, Celery, and OpenAI to provide authenticated document storage and background processing. It currently supports document summaries, invoice extraction, and contract metadata extraction from PDF, DOCX, JPEG, PNG, and TIFF files.

> [!NOTE]
> This repository is under active development. The current implementation is best suited to local development and evaluation; review the [known limitations](#known-limitations) before using it with production or sensitive data.

## Table of contents

- [Motivation](#motivation)
- [Features](#features)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [API reference](#api-reference)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [Known limitations](#known-limitations)
- [License](#license)

## Motivation

Organizations receive valuable information in documents, but turning those files into usable data is often slow and repetitive. Each workflow needs the same supporting infrastructure: authenticated uploads, durable storage, text extraction, long-running AI jobs, status tracking, and result retrieval. Building that foundation separately for every application adds complexity before the actual document use case can even be addressed.

DocFlow centralizes those concerns behind one API. Applications can submit documents quickly while OCR and AI processing continue asynchronously in the background, avoiding long-running HTTP requests and giving clients a consistent way to monitor progress and retrieve results.

The project is intended to be a practical, extensible foundation for workflows such as summarizing reports, extracting invoice fields, identifying contract metadata, and adding new document-processing strategies without rebuilding the surrounding storage and job infrastructure.

## Features

- JWT-based signup, login, profile management, and password changes
- Email-based password reset flow
- Per-user document and file access
- Multipart uploads backed by MinIO object storage
- PDF and DOCX text extraction
- OCR for JPEG, PNG, and TIFF images with Tesseract
- Background processing through Celery and Redis
- PostgreSQL persistence for users, documents, requests, jobs, and results
- Three processing modes: `DOCUMENT_SUMMARY`, `INVOICE_EXTRACTION`, and `CONTRACT_METADATA`
- Interactive OpenAPI documentation through Swagger UI and ReDoc

## How it works

```text
Client
  |
  v
FastAPI --------> PostgreSQL
  |               users, documents, requests, jobs, results
  |
  +-------------> MinIO
  |               original files
  |
  `-------------> Redis queue -----> Celery worker
                                      |-- extract text / run OCR
                                      |-- send content to OpenAI
                                      `-- persist status and result
```

Uploading a file creates the file, document, and processing-request records before placing a task on the queue. A successful request normally moves through:

```text
PENDING -> QUEUED -> PROCESSING -> COMPLETED
```

Errors move the request to `FAILED`. `CANCELLED` exists in the data model, but there is currently no cancellation endpoint.

## Tech stack

| Component | Role |
| --- | --- |
| FastAPI | HTTP API and OpenAPI documentation |
| SQLAlchemy + asyncpg | Async database access |
| PostgreSQL | Application and workflow data |
| Alembic | Database migrations |
| MinIO | Uploaded file storage |
| Redis | Celery message broker and result backend |
| Celery | Background document processing |
| OpenAI API | Summarization and extraction |
| pypdf / python-docx | PDF and DOCX text extraction |
| Tesseract | Image OCR |

## Quick Start

### Prerequisites

- Python 3.12 or newer
- Docker and Docker Compose
- Tesseract OCR
- An OpenAI API key with access to the configured model
- SMTP credentials if you want to test password-reset email delivery

The Docker image uses Python 3.12 and installs Tesseract and Poppler for OCR. For local development, install those system tools on the host as well.

### 1. Create the environment

```bash
git clone <repository-url>
cd doc_flow_backend

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install Tesseract and Poppler if they are not already available:

```bash
# macOS
brew install tesseract poppler

# Debian / Ubuntu
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils
```

### 2. Configure the application

Create `.env` in the repository root. The checked-in `.env.example` is not yet complete, so use the full example in [Configuration](#configuration).

### 3. Start the dependencies

```bash
docker compose up -d postgres redis minio
docker compose exec postgres pg_isready -U postgres -d docflow
alembic upgrade head
```

### 4. Start the API and worker

Run the API:

```bash
uvicorn app.main:app --env-file .env --host 0.0.0.0 --port 8000 --reload
```

In another terminal, activate the same virtual environment and run the worker:

```bash
source .venv/bin/activate
celery -A app.tasks.celery_app worker --loglevel=INFO
```

Verify the service:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok","service":"doc-flow-backend","message":"App is running"}
```

| Service | URL |
| --- | --- |
| API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| MinIO API | http://localhost:9000 |
| MinIO console | http://localhost:9001 |

PostgreSQL is exposed on `localhost:5433`; Redis is exposed on `localhost:6379`.

## Configuration

Create a `.env` file with the following settings:

```dotenv
app_name=DocFlow
secret_key=replace-with-a-random-secret
algorithm=HS256
access_token_expire_minutes=30

broker_host=redis://localhost:6379/0
broker_backend=redis://localhost:6379/1
OPENAI_API_KEY=replace-with-your-api-key
OPENAI_MODEL=gpt-5.5
OPENAI_TIMEOUT_SECONDS=45
OPENAI_MAX_INPUT_CHARS=100000
OPENAI_MAX_OUTPUT_TOKENS=2048

DATABASE_NAME=docflow
DATABASE_USER=postgres
DATABASE_PASSWORD=postgres
DATABASE_HOST=localhost
DATABASE_PORT=5433

MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_BUCKET=docflow
MINIO_SECURE=false
MAX_UPLOAD_SIZE_BYTES=10485760

DATABASE_URL_TEST=postgresql+asyncpg://docflow_user:your_password@localhost:5432/test_docflow

reset_token_expire_minutes=60
mail_server=localhost
mail_port=587
mail_username=
mail_password=
mail_from=noreply@example.com
mail_use_tls=true
frontend_url=http://localhost:3000
cors_origins=["http://localhost:3000"]
rate_limit_enabled=true
processing_email_enabled=true
```

Generate a signing secret with:

```bash
python -c 'import secrets; print(secrets.token_hex(32))'
```

Keep `.env` and real credentials out of version control. `MINIO_ENDPOINT` must be a `host:port` value without a URL scheme.

`OPENAI_MODEL` selects the model, `OPENAI_TIMEOUT_SECONDS` bounds each API request, `OPENAI_MAX_INPUT_CHARS` limits extracted document text sent for processing, and `OPENAI_MAX_OUTPUT_TOKENS` caps generated output. The character and token limits help control request size but do not impose a fixed billing ceiling.

### Usage numbers and evals

Every streamed answer is stored with what it cost: `provider`, `input_tokens`, `output_tokens`, `cached_tokens`, `reasoning_tokens` and `latency_ms` are columns of `message`, and each one is empty when the provider did not report it (an empty value means "unknown", not zero). `GET /api/v1/chat/stats` sums them per day for the logged-in user.

After every change of the prompt (`app/prompts/chat_*.md`) or of the model settings, run the evals against a running API and compare with the run before:

```bash
python -m scripts.run_evals --email you@example.com --password '...'   # or --token / EVALS_TOKEN
python -m scripts.run_evals --compare evals/results/<old>.json evals/results/<new>.json
```

Each run is saved in `evals/results/` (not committed) with its score, so two runs can be compared. `--model groq` tests one provider, `--min-score 0.9` makes the command fail below 90%. It makes about ten real model calls per run. The case format is in `evals/README.md`.

### Monitoring with Sentry

Error and performance monitoring is optional and stays off while `SENTRY_DSN` is empty. To turn it on, create a Python project in Sentry, copy its DSN into `.env`, and restart the API and the worker:

```dotenv
SENTRY_DSN=https://<key>@<org>.ingest.sentry.io/<project>
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=0.1
```

What it reports: unhandled exceptions and HTTP 500 errors from the API and the Celery worker, plus an error when every chat provider has failed. Provider outages that were answered with 502, 503 or 504, and automatic failovers between providers, are expected and are not reported. Errors carry the numeric user id and the chat model choice and provider as tags.

Logs: everything the app writes through the configured logger (`LOG_LEVEL`, default INFO) is also sent to Sentry Logs, with the same extra fields as the JSON line (`request_id`, `user_id`, `provider`, `status`, `latency_ms`, ...). ERROR logs also become error events. `uvicorn.access` is excluded because it prints full URLs with query strings. Set `SENTRY_ENABLE_LOGS=false` to stop sending logs, or `SENTRY_LOGS_LEVEL=WARNING` to send fewer (default `INFO`; the worker runs at DEBUG, which would be very noisy). If nothing shows up in Sentry, set `SENTRY_DEBUG=true`: the SDK then prints why events are dropped or cannot be sent, and the startup log says `Sentry is on` or `Sentry is off: SENTRY_DSN is empty`.

What it never receives: request bodies (chat messages), local variables, prompts and answers, cookies, the `Authorization` header, e-mail addresses and IP addresses. `SENTRY_TRACES_SAMPLE_RATE` is the share of requests that also send performance traces.

If you run the API and worker inside Compose, use service names instead of host addresses:

```dotenv
DATABASE_HOST=postgres
DATABASE_PORT=5432
broker_host=redis://redis:6379/0
broker_backend=redis://redis:6379/1
MINIO_ENDPOINT=minio:9000
```

### Logging

The API and the worker write one JSON line per log record to **stdout** and nothing else. In Docker, Docker collects that output, so there are no log files to manage inside the container:

```bash
docker compose logs -f api worker          # follow both
docker compose logs api | jq 'select(.request_id == "<id>")'   # one request from start to end
```

`docker-compose.yml` keeps at most 3 files of 10 MB per container (`max-size`, `max-file`). Without that Docker never rotates its log file and it can fill the disk.

| Setting | Default | Meaning |
|---|---|---|
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |
| `LOG_QUEUE` | `true` | `false` writes directly instead of through the queue (useful when debugging a crash: the last lines then appear immediately) |
| `LOG_QUEUE_SIZE` | `10000` | How many lines may wait. When it is full, the newest lines are dropped and counted; the count is printed to stderr at shutdown |

How it works:
- **A queue keeps logging from slowing requests down.** In the API, `logger.info(...)` only puts the line in memory; a background thread writes it. If stdout is slow or stuck (a full pipe, a slow log collector), requests keep running. The Celery worker writes directly, because its pool forks child processes and the writer thread would not exist in them.
- **Every request has an id.** It is read from the `X-Request-ID` header if the caller sent one (8 to 64 letters, digits, `.`, `_`, `-`), otherwise it is generated, and it is returned in the `X-Request-ID` response header. Every log line written while the request is handled carries it as `request_id`.
- **One line per request**, `"message": "Request finished"`, with `method`, `path`, `status` and `latency_ms`. `path` is the route template (`/api/v1/chat/conversations/{id}`), never the real URL, so ids and query strings are not logged. It replaces the `uvicorn.access` log, which is switched off. `/health` is logged at DEBUG while it succeeds; 5xx answers are WARNING, not ERROR, so a provider outage does not become a Sentry event.
- **Only a fixed list of fields is printed** from `extra` (see `context_fields` in `app/core/logging_config.py`). Add a field there on purpose; questions, answers and file names are not on the list.
- `httpx`, `httpcore` and `urllib3` are limited to WARNING, because they log every call with its URL.

### Speech to text

The audio-to-text endpoint uses Groq's hosted Whisper model with the same `GROQ_AI_API_KEY` as chat:

```dotenv
GROQ_AI_API_KEY=replace-with-your-groq-key
TRANSCRIBE_MODEL=whisper-large-v3-turbo
TRANSCRIBE_MAX_FILE_MB=10
TRANSCRIBE_TIMEOUT_SECONDS=60
ENVIRONMENT=production
```

`TRANSCRIBE_MAX_FILE_MB` is at most 25 (the provider's free-tier limit) and `TRANSCRIBE_TIMEOUT_SECONDS` at most 120. `ENVIRONMENT=development` shows the values of failed SQL statements in error messages, which helps while coding; any other value (the default is `production`) hides them, because they contain users' messages and transcripts.

What is kept: the transcript text, plus the model, the size of the audio in bytes and how long the provider took. What is not kept: the audio itself. Neither the transcript nor the file name is written to the logs, and Sentry never receives request bodies. The audio is sent to Groq, a hosted service, so do not send recordings you are not allowed to share with a third party.

### Image questions

Questions about an uploaded image go to Gemini through its OpenAI-compatible endpoint, with the same `GEMINI_AI_API_KEY` and `GEMINI_BASE_URL` as the chat fallback:

```dotenv
VISION_MODEL=gemini-3.1-flash-lite
VISION_MAX_IMAGE_MB=5
VISION_TIMEOUT_SECONDS=60
```

`VISION_MAX_IMAGE_MB` is at most 14: base64 makes the file a third bigger and Gemini accepts 20 MB per request. Also set a request size limit in the reverse proxy (for example `client_max_body_size` in nginx), because the server has read the whole upload before the API can refuse it.

Every upload is checked before it is used, following the OWASP File Upload Cheat Sheet: the extension must be on an allowlist (jpg, jpeg, png, webp), the size is limited while reading, the first bytes must match the extension, the picture is refused above 40 million pixels, and the image is then decoded and written again, which removes EXIF data (GPS position, camera, time), text chunks and bytes hidden after the image. The browser's content type is ignored. What is kept: the question, the answer, the token counts, and for the image only its size, type, pixel size and SHA-256. The image itself is not stored, and neither the question, the answer nor the file name is logged. The checks live in `app/utils/upload.py` (any feature) and `app/utils/image.py` (images), so a document feature can reuse them with its own list of types. Not done yet: removing active content from PDF or Office files.

#### Virus scanning

Image and document uploads can be scanned with [ClamAV](https://www.clamav.net/) before they are decoded, parsed or stored. It is off when `CLAMAV_HOST` is empty (the API logs one warning at the first upload). Once `CLAMAV_HOST` is set the scan is required: if the scanner cannot be reached or does not answer within `CLAMAV_TIMEOUT_SECONDS`, the upload is refused with 503 instead of being let through. A file with a virus is refused with 422 and the signature name is logged, not returned.

```
CLAMAV_HOST=clamav      # service name in docker compose; use localhost if the API runs outside Docker
CLAMAV_PORT=3310
CLAMAV_TIMEOUT_SECONDS=30
```

Start the scanner with `docker compose --profile scan up -d clamav`. The first start downloads the virus database and takes a few minutes. ClamAV needs at least 3 GiB of memory (4 GiB is better), so it is too heavy for the smallest servers; there, keep it off and rely on the other checks, or run it on a separate machine. The scan uses clamd's `INSTREAM` command, so no file is written to disk. Scanning does not replace the other checks, it adds to them.

On Gemini's free tier Google may use what you send to improve its products, and human reviewers may read it. Send only images you may share with a third party.

## Usage

The examples below use the sample invoice at `tests/assets/sample-pdf-invoice.pdf`.

### Register and authenticate

```bash
curl -X POST http://localhost:8000/api/v1/users/signup \
  -H 'Content-Type: application/json' \
  -d '{"first_name":"Demo","last_name":"User","email":"demo@example.com","password":"ExamplePassword123!"}'

curl -X POST http://localhost:8000/api/v1/users/token \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode 'username=demo@example.com' \
  --data-urlencode 'password=ExamplePassword123!'
```

Login uses OAuth2 form fields, so the email address is passed as `username`. Copy the returned token:

```bash
TOKEN='paste-access-token-here'
curl http://localhost:8000/api/v1/users/me \
  -H "Authorization: Bearer $TOKEN"
```

### Upload and process a document

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Authorization: Bearer $TOKEN" \
  -F 'file=@tests/assets/sample-pdf-invoice.pdf;type=application/pdf' \
  -F 'processing_type=DOCUMENT_SUMMARY' \
  -F 'instructions=Summarize the key information in this document.'
```

Example response:

```json
{"document_id":1,"processing_request_id":1,"status":"QUEUED"}
```

`instructions` is optional.

Valid processing modes are `DOCUMENT_SUMMARY`, `INVOICE_EXTRACTION`, and `CONTRACT_METADATA`.

### Check status and retrieve the result

Use the processing request ID returned by upload in the route path:

```bash
REQUEST_ID=1
curl "http://localhost:8000/api/v1/processing-requests/status/$REQUEST_ID" \
  -H "Authorization: Bearer $TOKEN"

curl "http://localhost:8000/api/v1/processing-requests/result/$REQUEST_ID" \
  -H "Authorization: Bearer $TOKEN"
```

Summary results contain a text string. Invoice results use a validated object with invoice dates, parties, currency, totals, and line items; contract results use a validated object with parties, dates, governing law, renewal and termination terms, and obligations. Responses include `confidence_score: null` because the model does not provide a calibrated confidence measure. A result that is not available returns `404`.

For example, an invoice response is shaped like `{"result":{"invoice_number":"INV-001","invoice_date":"2026-04-15","due_date":null,"vendor_name":"Example Ltd","customer_name":"Buyer Inc","currency":"USD","subtotal":100,"tax":10,"total":110,"line_items":[]},"confidence_score":null}`. Missing scalar fields are represented as `null`; missing collections are empty arrays.

### Download the original file

```bash
DOCUMENT_ID=1
curl "http://localhost:8000/api/v1/documents/$DOCUMENT_ID" \
  -H "Authorization: Bearer $TOKEN"

FILE_ID=1
curl "http://localhost:8000/api/v1/files/$FILE_ID" \
  -H "Authorization: Bearer $TOKEN" \
  --output downloaded-document.pdf
```

### Ask a question by voice

Two requests: the first turns the recording into text, the second sends that text to the chat.

```bash
curl -X POST http://localhost:8000/api/v1/chat/transcribe \
  -H "Authorization: Bearer $TOKEN" \
  -F 'file=@question.ogg;type=audio/ogg'
# {"id":1,"text":"What is the capital of France?","model":"whisper-large-v3-turbo","created_at":"2026-10-20T14:30:44Z"}

curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is the capital of France?"}'
```

Accepted formats are flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav and webm, checked by file extension only. A chat message is limited to 4,000 characters, so a long recording can be transcribed and then be refused by the chat with a 422. A client should show the text first and send it only when the user confirms. `GET /api/v1/chat/transcribe/{id}` returns the stored transcript again.

### Ask a question about an image

```bash
curl -X POST http://localhost:8000/api/v1/chat/image \
  -H "Authorization: Bearer $TOKEN" \
  -F 'file=@photo.jpg;type=image/jpeg' \
  -F 'question=What is on the desk?'
# {"conversation_id":7,"answer":"A laptop and a cup of coffee.","model":"gemini-3.1-flash-lite","input_tokens":300,"output_tokens":20}
```

The question and the answer are saved as a new conversation (see `GET /api/v1/chat/conversations/7`), and their tokens count towards the daily budget and `/api/v1/chat/stats`. Each question is about one image; to ask about the image again, send it again.

## API reference

The complete, interactive schema is available at `/docs` while the API is running.

| Method | Endpoint | Auth | Description |
| --- | --- | --- | --- |
| GET | `/`, `/start` | No | Application liveness |
| GET | `/health` | No | Checks PostgreSQL, Redis, and object storage; 503 if any is down |
| POST | `/api/v1/users/signup` | No | Register a user |
| POST | `/api/v1/users/token` | No | Obtain an access token |
| GET | `/api/v1/users/me` | Yes | Get the current user |
| GET | `/api/v1/users/me/stats` | Yes | Dashboard counts: documents, requests by status and type, recent failures |
| PATCH | `/api/v1/users/{id}` | Yes | Update your profile |
| DELETE | `/api/v1/users/{id}` | Yes | Delete your account |
| POST | `/api/v1/users/forgot-password` | No | Request a reset email |
| POST | `/api/v1/users/reset-password` | No | Reset with a token |
| PATCH | `/api/v1/users/me/password` | Yes | Change the current password |
| POST | `/api/v1/documents` | Yes | Upload and enqueue a document |
| GET | `/api/v1/documents` | Yes | List your documents; filter with `q` (name search) and `status` (`ACTIVE`/`ARCHIVED`) |
| GET | `/api/v1/documents/{id}` | Yes | Get document metadata |
| PATCH | `/api/v1/documents/{id}` | Yes | Rename and/or archive: `{"name": "..."}`, `{"status": "ARCHIVED"}` or `{"status": "ACTIVE"}` |
| POST | `/api/v1/documents/{id}/process` | Yes | Process an uploaded document again: `{"processing_type": "...", "instructions": "..."}` |
| DELETE | `/api/v1/documents/{id}` | Yes | Soft-delete a document and remove its stored file |
| GET | `/api/v1/documents/{id}/jobs` | Yes | List processing attempts |
| GET | `/api/v1/documents/{id}/processing-requests` | Yes | List a document's processing requests, newest first |
| GET | `/api/v1/files` | Yes | List your files |
| GET | `/api/v1/files/{id}` | Yes | Download a file |
| DELETE | `/api/v1/files/{id}` | Yes | Soft-delete its file and parent document, then remove the stored object |
| GET | `/api/v1/processing-requests/{id}` | Yes | Get a processing request |
| GET | `/api/v1/processing-requests/status/{id}` | Yes | Get processing status |
| GET | `/api/v1/processing-requests/result/{id}` | Yes | Get processing output |
| GET | `/api/v1/processing-requests/{id}/export?format=json\|csv` | Yes | Download the result as JSON or CSV |
| POST | `/api/v1/processing-requests/{id}/cancel` | Yes | Cancel a request that has not started (409 otherwise) |
| POST | `/api/v1/processing-requests/{id}/retry` | Yes | Queue a new request with the same settings as a failed or cancelled one (409 otherwise) |
| POST | `/api/v1/chat` | Yes | Ask the assistant one question: body `{"message": "...", "model": "auto"}` (message 1-4,000 characters after trimming; `model` optional, see `GET /api/v1/chat/models`). 422 for invalid input, 429 when rate limited, 503/504 when the AI provider is unavailable or slow |
| GET | `/api/v1/chat/models` | Yes | The values the `model` field accepts: `auto` (default, tries every configured provider in turn) plus each provider that has an API key. A chosen provider is tried first and the others still back it up; 400 if it is not configured, 422 for any name not on the list |
| POST | `/api/v1/chat/stream` | Yes | Same message rules as `/api/v1/chat`, answered as Server-Sent Events and remembered: body `{"message": "...", "conversation_id": 12, "model": "groq"}` (`conversation_id` and `model` optional; omit it to start a new conversation). Events: `conversation` (`{"conversation_id"}`), `token` (`{"text"}`), then `done` (provider, model, token counts, timings) or `error` (`code`, `message`). Failures before the first token use normal HTTP status codes, 404 for a conversation that is not yours. Browsers need `fetch` with a stream reader, because `EventSource` only supports GET |
| GET | `/api/v1/chat/stats` | Yes | Your own usage over the last `days` days (1-90, default 7): answers, input/output/cached/reasoning tokens in total and per day (days without answers are zeros), p50/p95 latency of finished answers, and the error rate (errors / answers; answers you cut off are counted separately). Only streamed conversations are stored, so only they appear here |
| POST | `/api/v1/chat/transcribe` | Yes | Speech to text: multipart field `file` (flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav or webm, at most `TRANSCRIBE_MAX_FILE_MB` MB). Returns `{"id", "text", "model", "created_at"}` and stores the text, not the audio. 413 too big, 415 wrong extension, 422 empty file, 429 when limited, 502/503/504 for provider problems |
| GET | `/api/v1/chat/transcribe/{id}` | Yes | One of your transcriptions; 404 if it does not exist or is not yours |
| POST | `/api/v1/chat/image` | Yes | Ask about one image: multipart fields `file` (jpg, jpeg, png or webp, at most `VISION_MAX_IMAGE_MB` MB) and `question` (1-4,000 characters). Returns `{"conversation_id", "answer", "model", "input_tokens", "output_tokens"}` and saves the question and answer, not the image. 413 too big or too many pixels, 415 wrong type or content that does not match, 422 empty or invalid file or bad question, 429 when limited, 502/503/504 for provider problems |
| POST | `/api/v1/chat/conversations` | Yes | Start an empty conversation: body `{"title": "..."}` (optional, 1-100 characters; otherwise named after the first message) |
| GET | `/api/v1/chat/conversations` | Yes | Your conversations, most recently active first, with `skip` and `limit` |
| GET | `/api/v1/chat/conversations/{id}` | Yes | One conversation with all its messages, oldest first (404 if it is not yours) |
| PATCH | `/api/v1/chat/conversations/{id}` | Yes | Rename: body `{"title": "..."}` |
| DELETE | `/api/v1/chat/conversations/{id}` | Yes | Delete the conversation and its messages (204) |

Collection endpoints use `skip` and `limit`, return newest first, and cap `limit` at 50.

Login allows 10 attempts per email every 5 minutes and forgot-password 3 emails per 15 minutes (HTTP 429 beyond that). Counters live in the broker Redis; set `rate_limit_enabled=false` to turn this off.

When processing finishes or finally fails, the document owner gets an email. Set `processing_email_enabled=false` to turn this off. Failed attempts list a readable `failure_reason` in `GET /api/v1/documents/{id}/jobs`.

## Development

### Database migrations

```bash
alembic current
alembic upgrade head

# After changing a persistence model:
alembic revision --autogenerate -m "describe the schema change"
```

Always review autogenerated migrations before applying or committing them.

### Project structure

```text
app/
├── main.py                  # FastAPI application and lifecycle
├── api/v1/                 # Versioned HTTP routes
├── core/                   # Settings and MinIO client
├── db/                     # Engines, sessions, and dependencies
├── models/schema.py        # SQLAlchemy persistence models
├── service/                # Auth, document, file, and OpenAI logic
├── tasks/                  # Celery configuration and tasks
└── utils/                  # Text extraction and email helpers
migrations/                 # Alembic migrations
evals/                      # Chat test questions (chat.json) and saved eval runs
scripts/                    # Tools you run by hand, e.g. run_evals.py (not part of the server)
templates/email/            # Password-reset email template
tests/                      # API tests and sample assets
```

### Docker Compose

Start the full development stack with:

```bash
docker compose up --build -d
docker compose logs -f api worker
```

The API entrypoint waits for PostgreSQL, Redis, and MinIO before applying migrations and starting Uvicorn. Compose also gates the API and worker on PostgreSQL and Redis health. The wait is bounded by `DEPENDENCY_STARTUP_TIMEOUT_SECONDS` (60 seconds by default).

Use `docker compose down` to stop the stack while preserving data. Adding `--volumes` removes the named PostgreSQL and MinIO volumes and their data.

## Testing

The tests use pytest, HTTPX, AnyIO, and a real PostgreSQL test database. MinIO calls and task submission are mocked in upload tests; the suite does not perform end-to-end OCR or OpenAI processing.

> [!WARNING]
> Use an isolated test database. The fixture creates and drops all application tables.

Set `DATABASE_URL_TEST` in `.env` (or the process environment) to the isolated database used by the fixture:

```text
postgresql+asyncpg://docflow_user:your_password@localhost/test_docflow
```

The fixture now uses this setting and refuses to reset the configured application database. Keep `DATABASE_URL_TEST` pointed at a separate database dedicated to tests, and provision that database and user before running the suite. The Compose database is published on host port 5433; `.env.example` uses that port and a separate `test_docflow` database.

```bash
python -m pytest -q
```

## Contributing

Contributions are welcome. Before opening a pull request:

1. Create a focused branch for the change.
2. Keep route handlers, service logic, and persistence concerns in their existing layers.
3. Add an Alembic migration for every database-model change.
4. Add or update tests for changed behavior.
5. Run `python -m pytest -q` against an isolated test database.
6. Update this README when configuration, endpoints, or operating requirements change.
7. Explain the motivation, behavior change, and verification performed in the pull request.

Never commit `.env`, API keys, SMTP credentials, uploaded documents, or other secrets. For security issues, avoid publishing sensitive details in a public issue; contact the repository owner privately instead.

## Known limitations

- The sample `.env.example` documents the required application settings; replace its placeholder secrets and credentials before deploying.
- The OpenAI call is bounded by configurable input-character and output-token limits, but those limits do not guarantee a fixed monetary cost.
- Uploads default to a 10 MiB maximum, configurable with `MAX_UPLOAD_SIZE_BYTES`; the MIME type and file signature or container are checked before storage.
- PDFs without embedded text use Tesseract OCR through Poppler. Large or unusually complex PDFs can still require substantial worker memory and processing time.
- Celery task failures are retried up to three times, and job attempts record retry and failure state.
- Worker logs include document, request, task, and attempt identifiers but omit extracted document text.

## License

No license file is currently included. Unless the repository owner adds one, the project should be treated as all rights reserved; public source availability alone does not grant permission to use, modify, or redistribute it.
