## Chat

Legend: `[x]` done and tested, `[ ]` not done yet.

<details open> 
<summary>Phase 0. Setup</summary>

### Phase 0 · Setup (half a day)
- [x] Pick the use case + write a system prompt
  - [x] Prompt lives in `app/prompts/chat_v1.md` (versioned by file name)
- [x] Write 10 test questions with expected answers (evals)
  - [x] `evals/chat.json`: 10 cases, each with `id`, `category`, `message`, `checks`, `why`
  - [x] `evals/README.md` explains the checks (`contains_all`, `contains_any`, `not_contains`, `max_words`)
- [x] FastAPI project, API keys in `.env`, `GET /health`
  - [x] Settings in `app/core/config.py`, keys never logged, `.env.example` kept up to date
  - [x] `/health` checks the database, MinIO and Redis
  - [ ] 
</details>

<details>
<summary>Phase 1 · One question, one answer</summary>

### Phase 1 · One question, one answer
- [x] `POST /api/v1/chat` receives `{ "message": "..." }`
- [x] Validate: trimmed, not empty, max 4,000 characters
  - [x] Whitespace-only and over-long messages return 422
- [x] `LLMClient` interface with ONE provider (OpenRouter)
  - [x] `LLMClient` Protocol with `complete()` and `stream()`, so a provider can be swapped without touching the routes
  - [x] One client class for every OpenAI-compatible provider (`OpenAICompatibleClient`)
- [x] Send system prompt + message → return the answer
  - [x] Answer, model and provider returned in the response
- [x] Handle failures: timeout (30s), rate limit (429) → retry, then a clear error
  - [x] Own retry with backoff and jitter, honours `Retry-After` (the SDK's retries are turned off)
  - [x] 30s per attempt and 45s for all attempts together
  - [x] Clear errors: 502 provider failed, 503 chat unavailable (bad key / no credit), 504 timeout
  - [x] Per-user rate limit (`CHAT_RATE_LIMIT_PER_MINUTE`) and `max_tokens` cap (`CHAT_MAX_OUTPUT_TOKENS`)
- [x] Log every call: model, latency, tokens in/out, status
  - [x] JSON log with `provider`, `model`, `status`, `latency_ms`, `input_tokens`, `output_tokens`, `attempts`
  - [x] Message text is never logged
- [x] Tests: validation + LLM mocked

</details>

<details>
<summary>Phase 2 · Streaming</summary>

### Phase 2 · Streaming
- [x] Send the answer word by word (Server-Sent Events)
  - [x] `POST /api/v1/chat/stream` with events `conversation`, `message`, `done`, `error`
  - [x] Errors before the first piece are normal HTTP errors; later errors become an `error` event
  - [x] Time limit for the whole stream (`CHAT_STREAM_TIMEOUT_SECONDS`)
- [x] Handle the user closing the page mid-answer
  - [x] The server notices the disconnect, stops the provider call and does not keep paying for tokens
  - [x] The partial answer is saved as `cancelled` (see Phase 3)
- [x] Tests: stream order, error mid-stream, disconnect

</details>

<details>
<summary>Phase 3 · Memory (conversations)</summary>

### Phase 3 · Memory (conversations)
- [x] Schema: `conversations` + `messages`
  - [x] Alembic migration, indexes on `user_id` and `updated_at`
  - [x] Message `status`: `complete`, `error`, `cancelled`
- [x] Send the last N messages to the LLM, so it remembers
  - [x] `CHAT_HISTORY_MESSAGES` and `CHAT_HISTORY_MAX_CHARS` trim the history
  - [x] The user's message is saved first, so it is never lost
- [x] Create, list, get, rename, delete a conversation
  - [x] `POST/GET /conversations`, `GET/PATCH/DELETE /conversations/{id}`, list is paginated
  - [x] Every query is scoped to the user; someone else's conversation is a 404
- [x] Auto-title from the first message
- [x] Save the AI's message after the stream finishes
  - [x] Saved with a fresh database session, before the `done` event
  - [x] A cut-off or failed stream saves the partial answer with its status
- [x] Tests: ownership, trimming, saving on error and cancel

</details>

<details>
<summary>Phase 4 · More models + fallback</summary>

### Phase 4 · More models + fallback
- [x] Second provider behind the same `LLMClient`
  - [x] Gemini, Groq and OpenAI added through the registry; keys and models in settings
- [x] If model A fails → try model B
  - [x] Order: OpenRouter → Gemini → Groq → OpenAI (`gpt-3.5-turbo`, last resort, cheap)
  - [x] One shared time budget (`CHAT_FALLBACK_TOTAL_TIMEOUT_SECONDS`)
  - [x] A stream only falls back before its first piece
  - [x] A provider without a key is skipped
  - [x] Checked live against all four providers
- [x] Choose the model per request
  - [x] `model` field with an allowlist: `auto`, `openrouter`, `gemini`, `groq`, `openai`
  - [x] `GET /api/v1/chat/models` lists what is available
- [x] Tests: fallback order, no fallback after the first piece, unknown model → 422

</details>

<details open>
<summary>Phase 5 · Observability</summary>

### Phase 5 · Observability
Phase 5a is done: errors, logs and traces go to Sentry. Phase 5b is the data you query yourself.

<details>
<summary>5a · Sentry (done)</summary>

#### 5a · Sentry (done)
- [x] Add Sentry (`app/core/observability.py`), off while `SENTRY_DSN` is empty
  - [x] FastAPI, Starlette, Celery, logging and OpenAI integrations
  - [x] Only HTTP 500, unhandled exceptions and "every provider failed" become issues; provider outages (502/503/504) do not
- [x] Privacy: no request bodies, local variables, prompts, cookies, auth header, e-mail or IP; only the numeric user id
- [x] Send logs from the logger config to Sentry Logs (`SENTRY_LOGS_LEVEL`, `uvicorn.access` excluded)
- [x] Tag errors with `llm.provider` and `chat.model_choice`
- [x] `SENTRY_DEBUG` switch and a startup line "Sentry is on / off"
- [x] Tests with a fake transport, README section

</details>

<details open>
<summary>5b · Your own numbers (done)</summary>

#### 5b · Your own numbers (done)
- [x] Store token usage on every assistant message
  - [x] Nullable columns `input_tokens`, `output_tokens`, `cached_tokens`, `reasoning_tokens`, `provider`, `latency_ms` (`model` was already there)
  - [x] Nullable because not every provider reports every kind; a provider that sends no usage still saves
  - [x] Read safely from `usage.prompt_tokens_details` and `usage.completion_tokens_details`, in both `complete()` and `stream()`
  - [x] Alembic migration `c4e8a1d6f293`, copies the old numbers out of the `metadata` JSON (run on PostgreSQL 16 with messy rows)
  - [x] Tests: usage reading, saving with and without usage, cut-off answers
  - [x] The plain `/chat` endpoint saves no messages, so only streamed conversations are counted
- [x] Track: p50/p95 latency, tokens per day, error rate
  - [x] Three small queries: usage per day, answers by status, latencies (percentiles are worked out from the sorted latencies, so it also runs on SQLite in tests)
  - [x] `GET /api/v1/chat/stats?days=7` (1-90): the user's own numbers. An all-users view needs an admin role, which the app does not have yet
  - [x] Tests: per-day sums, p50/p95, error rate, other users' and old answers left out, bad `days`
- [x] Rerun evals after every prompt or model change
  - [x] `python -m scripts.run_evals` reads `evals/chat.json`, calls the real `POST /api/v1/chat`, prints pass/fail per case
  - [x] Saves a scored run in `evals/results/`; `--compare OLD NEW` shows which cases got worse or better; `--min-score` makes it usable in CI
  - [ ] Optional: an LLM-as-judge check for answers that keyword checks judge unfairly (not built)
  - [ ] Run it once against the live API and keep the first result as the baseline

</details>

</details>

<details open>
<summary>Phase 6 · Cost and abuse control</summary>

### Phase 6 · Cost and abuse control
- [ ] Daily token budget per user
  - [ ] Setting `CHAT_DAILY_TOKEN_BUDGET` (needs 5b token columns)
  - [ ] Over budget → 429 with a clear message and when it resets
  - [ ] Counts the partial answer of a cancelled stream
- [ ] Limit what goes to the model
  - [ ] Token-based limit on history, not only characters
  - [ ] Reject a message that is too big for the model's window
- [ ] Keep the rate limit honest
  - [ ] Count streams that are still running per user (max 2 at once)
  - [ ] Test that the Redis-down case fails open and is logged
- [ ] Protect other users' data
  - [ ] Test that a user cannot read, rename, delete or continue another user's conversation
  - [ ] Test that the system prompt cannot be changed through the `message` field
- [ ] Alert when spend is unusual
  - [ ] Sentry alert on a sudden jump in tokens per hour

</details>

<details>
<summary>Phase 7 · Reliability</summary>

### Phase 7 · Reliability
- [ ] Circuit breaker per provider
  - [ ] After 3 failures in a row, skip that provider for 60 seconds
  - [ ] Half-open: let one request through to test it again
  - [ ] State in Redis, so every API process shares it
  - [ ] Test: a dead provider costs the next request no waiting time
- [ ] Provider health
  - [ ] `GET /health` shows which chat providers are usable (no key, open breaker, ok)
  - [ ] Never show key values
- [ ] Shut down gracefully
  - [ ] On deploy, running streams finish or save their partial answer
- [ ] Load test
  - [ ] 20 users at once on the stream endpoint; check p95, memory and database connections
  - [ ] Write the result in the README

</details>

<details>
<summary>Phase 8 · Better memory</summary>

### Phase 8 · Better memory
- [ ] Paginate the messages of a conversation
- [ ] Summarise old messages instead of dropping them
  - [ ] When history is over the limit, ask the model for a short summary and store it on the conversation
  - [ ] Send the summary + the last N messages
  - [ ] Test that a fact from message 1 is still known at message 30
- [ ] Search inside a user's conversations by title

</details>

<details>
<summary>Phase 9 · Feedback and control</summary>

### Phase 9 · Feedback and control
- [ ] Thumbs up / down on an answer
  - [ ] `PUT /conversations/{id}/messages/{message_id}/feedback` with a value and an optional reason
  - [ ] Counts shown in `/chat/stats`; bad answers can be added to the evals
- [ ] Regenerate the last answer
- [ ] Edit a user message and answer again
  - [ ] Later messages are removed or kept as a branch (decide first)
- [ ] Stop a running stream from another request
  - [ ] `POST /chat/stream/{id}/stop` for clients that cannot close the connection
- [ ] Tests for every endpoint, including "not your message → 404"

</details>

<details>
<summary>Phase 10 · Quality gate</summary>

### Phase 10 · Quality gate
- [ ] Run the evals in CI
  - [ ] A small pinned model, or mocked answers, so CI is cheap and repeatable
  - [ ] The build fails if the score drops by more than an agreed amount
- [ ] Remember which prompt wrote each answer
  - [ ] `prompt_version` column on `messages`
  - [ ] Compare scores per prompt version
- [ ] Review the prompt on a schedule
  - [ ] Add the 3 worst answers of the month as new eval cases
- [ ] Final check of the whole chat
  - [ ] README has: setup, settings, endpoints, SSE example, monitoring, how to run evals
  - [ ] `openapi.json` regenerated and committed

</details>
