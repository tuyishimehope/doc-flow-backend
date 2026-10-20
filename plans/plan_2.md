## Voice (Part 2): audio in, answer out. Smallest version first

Legend: `[x]` done and tested, `[~]` built but no tests yet, `[ ]` not done yet.

Goal: a user sends a short voice message, the API turns it into text, stores the text, logs the call,
and the existing chat answers it.

How to work: finish Part A in order. Each step is small and ends with a check you can run by hand.
Part B is a menu of extra steps; take **one at a time, only when the "when" column is true for you**.
Nothing in Part B is needed for a working demo.

### Decisions already made
- **Model:** Groq `whisper-large-v3-turbo` (free tier, hosted).
  - Same OpenAI-style API your chat already uses: `POST {GROQ_BASE_URL}/audio/transcriptions`.
  - `GROQ_AI_API_KEY` and `GROQ_BASE_URL` are already in `app/core/config.py`.
  - Formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm. Free-tier file limit: 25 MB.
    Anything shorter than 10 s is billed as 10 s. (Source: Groq docs, checked 2026-10-20.)
  - Free-tier rate limits: read the Rate Limits page in your Groq console and write the numbers here: 20 RPM, 2K RPD
- **Audio is never stored.** It is sent to the model and dropped. Only the transcript is stored.
- **Test with your own or public audio only.** Do not send real recordings of other people to a free hosted
  service. For real casework audio see B10 (local model) and ask your team for its data-handling rules first.
- **Languages:** start with English. Whether Whisper handles Kinyarwanda well is not confirmed here; B5 tests it.

### Already built from plan_1 (carry-over)
- [~] Daily token budget per user (`CHAT_DAILY_TOKEN_BUDGET`)
- [~] Max 2 running streams per user (`CHAT_MAX_CONCURRENT_STREAMS`)
- [~] `prompt_version` column on messages
- [ ] Write the decision "edit a message: remove or branch" in `plans/decisions.md` (only text, no code)

## Part A · The smallest working version

### A1 · Try the model by hand (no app code)
- [x] Get a 10 to 30 second clip you recorded yourself (English). Keep it outside the repo.
- [x] Call Groq with `curl` and a file, and read the text that comes back
  - [x] Check: the transcript is printed and roughly right
- [x] Write down the response fields you see (`text`, anything else) and how long it took

### A2 · Settings
- [x] `TRANSCRIBE_MODEL=whisper-large-v3-turbo`
- [x] `TRANSCRIBE_MAX_FILE_MB=10` (your own limit, below the provider's 25 MB)
- [x] `TRANSCRIBE_TIMEOUT_SECONDS=60`
- [x] Added to `app/core/config.py` and `.env.example`; the key is reused, never logged

### A3 · Send: `POST /api/v1/transcribe`
- [x] Multipart upload, signed-in users only
- [x] Reject before calling the provider (checked by file extension only, the smallest way)
  - [x] Too big → 413
  - [x] Extension not in the list → 415
  - [x] Empty file → 422
- [x] Send the bytes to Groq with the OpenAI SDK pointed at `GROQ_BASE_URL`, like the chat client does
- [x] Failures: provider error → 502, timeout → 504, with a clear message (no retries yet)

### A4 · Store the transcript
- [~] New table `transcription`: `id`, `user_id`, `text`, `model`, `audio_bytes`, `latency_ms`, `created_at`, with a migration (`ccff25f8d428`)
- [~] `POST /transcribe` saves a row and returns `{ "id", "text", "model", "created_at" }`
- [~] `GET /api/v1/transcribe/{id}` returns your own row; someone else's id → 404
- [~] Check: the row is in the database, and the audio is not saved anywhere (no file, no column)
- Not stored on purpose: token counts. Groq's transcription answer has none (only `text` and `x_groq.id`); it bills by audio hours. Audio length in seconds is B7.

### A5 · Log
- [~] One JSON log line per call: `provider`, `model`, `status`, `latency_ms`, `audio_bytes`
- [~] The transcript text and the file name are never logged (add the new fields to the log whitelist)
- [~] Check: make one good call and one failing call, then read the log lines

### A6 · Audio → answer, with no new chat code
Backend: nothing to build. The frontend makes the two calls (`/transcribe`, then `/chat/stream` with the returned `text`).
- [~] The client sends the audio to `/transcribe`, then sends the returned `text` as `message` to the existing `/chat/stream`
- [~] Check: a spoken question gets a streamed answer
- [~] A text longer than the chat limit gets the chat's normal 422

### A7 · Done when
- [~] You can record a 10 s question and get an answer with two `curl` commands
- [~] README has those two commands, the three settings, and one line on what is kept (transcript) and what is not (audio)

## Part B · Extra steps (one at a time, only when needed)

| Step | Do this | When you need it |
|---|---|---|
| B1 | `POST /chat/voice`: transcribe and answer in one request, with a `transcript` event first | A client wants one call instead of two |
| B2 | Check the file's first bytes, not only its extension | Before anyone else uses it |
| B3 | Retry once with backoff on 429 / 5xx from the provider | The log shows provider errors |
| B4 | Tests: 413 / 415, provider mocked, another user's id → 404, not signed in → 401 | Before you show it to anyone |
| B5 | `language` hint, 10 test clips, a word error rate number; test Kinyarwanda clips if your users speak it | Users speak other languages |
| B6 | Daily audio-minutes limit and a rate limit on uploads (reuse the Redis helpers) | More than a few users, or the free tier says stop |
| B7 | `source` and `audio_seconds` on messages; a `chat_v2` prompt that says "this came from speech and may contain mistakes" | You build B1 |
| B8 | Long recordings through a Celery job (`queued`, `running`, `done`, `failed`) | You need files over the size limit |
| B9 | Audio minutes per day and transcription p50 / p95 in `/chat/stats` | You want to see usage |
| B10 | Local model (for example faster-whisper on your own machine) so audio never leaves it | Before using any real recording of people |
| B11 | Run the voice evals live and keep the first result as a baseline | You change the model or the prompt |
