## Chat

### Phase 0 · Setup (half a day)
- [x] Pick the use case + write a system prompt
- [x] Write 10 test questions with expected answers (evals)
- [x] FastAPI project, API keys in `.env`, `GET /health`

### Phase 1 · One question, one answer
- [x] `POST /chat` receives `{ "message": "..." }`
- [x] Validate: trimmed, not empty, max 4,000 characters
- [x] `LLMClient` interface with ONE provider (OpenRouter)
- [x] Send system prompt + message → return the answer
- [x] Handle failures: timeout (30s), rate limit (429) → retry, then a clear error
- [x] Log every call: model, latency, tokens in/out, status
- [x] Tests: validation + LLM mocked

### Phase 2 · Streaming
- [x] Send the answer word by word (Server-Sent Events)
- [x] Handle the user closing the page mid-answer

### Phase 3 · Memory (conversations)
- [x] Schema: `conversations` + `messages`
- [x] Send the last N messages to the LLM, so it remembers
- [x] Create, list, get, rename, delete a conversation
- [x] Auto-title from the first message
- [x] Save the AI's message after the stream finishes

### Phase 4 · More models + fallback
- [x] Second provider behind the same `LLMClient`
- [x] If model A fails → try model B
- [x] Choose the model per request

### Phase 5 · Observability
- [x] Store every call in an `llm_calls` table
- [x] Track: p50/p95 latency, tokens per day, error rate
- [x] Rerun evals after every prompt or model change