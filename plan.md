## Chat

### Phase 0 · Setup (half a day)
- [x] Pick the use case + write a system prompt
- [x] Write 10 test questions with expected answers (evals)
- [x] FastAPI project, API keys in `.env`, `GET /health`

### Phase 1 · One question, one answer
- [ ] `POST /chat` receives `{ "message": "..." }`
- [ ] Validate: trimmed, not empty, max 4,000 characters
- [ ] `LLMClient` interface with ONE provider (OpenRouter)
- [ ] Send system prompt + message → return the answer
- [ ] Handle failures: timeout (30s), rate limit (429) → retry, then a clear error
- [ ] Log every call: model, latency, tokens in/out, status
- [ ] Tests: validation + LLM mocked

### Phase 2 · Streaming
- [ ] Send the answer word by word (Server-Sent Events)
- [ ] Handle the user closing the page mid-answer

### Phase 3 · Memory (conversations)
- [ ] Schema: `conversations` + `messages`
- [ ] Send the last N messages to the LLM, so it remembers
- [ ] Create, list, get, rename, delete a conversation
- [ ] Auto-title from the first message
- [ ] Save the AI's message after the stream finishes

### Phase 4 · More models + fallback
- [ ] Second provider behind the same `LLMClient`
- [ ] If model A fails → try model B
- [ ] Choose the model per request

### Phase 5 · Observability
- [ ] Store every call in an `llm_calls` table
- [ ] Track: p50/p95 latency, tokens per day, error rate
- [ ] Rerun evals after every prompt or model change