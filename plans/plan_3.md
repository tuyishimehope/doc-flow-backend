# Image (Part 3): upload an image, ask a question, get an answer. Smallest version first

Legend: `[x]` done and tested, `[~]` built but no tests yet, `[ ]` not done yet.

Goal: a user sends one image and one question. The API asks a vision model, returns the answer, saves the
question and the answer (with what the call cost) in a new conversation, and logs the call. The image is
not kept.

How to work: same as plan_2. Finish Part A in order; each step ends with a check you can run by hand.
Part B is a menu: take **one step at a time, only when the "when" column is true for you**.
Nothing in Part B is needed for a working demo.

<details>
<summary>Decisions already made</summary>

### Decisions already made
- **Model:** Gemini `gemini-3.1-flash-lite` through Google's OpenAI-compatible endpoint.
  - `GEMINI_AI_API_KEY`, `GEMINI_BASE_URL` and the model name `CHAT_GEMINI_MODEL` are already in `app/core/config.py`.
  - Request shape: a user message whose `content` is a list with a `text` part and an `image_url` part
    holding a data URI: `data:image/jpeg;base64,<bytes>`.
  - Free tier: input and output are free of charge; paid image input costs the same as text input.
  - Limits: the whole request, base64 text included, may be 20 MB. Base64 makes a file about a third bigger,
    so a raw image must stay well under 15 MB. Formats in Google's docs: png, jpeg, webp, heic, heif.
  - Free-tier rate limits: open <https://aistudio.google.com/rate-limit> and write the numbers here: ___
  - (Sources: Google Gemini API docs, checked 2026-10-20. Not checked: whether the compatible endpoint
    accepts heic/heif. A1 finds out.)
- **Why a separate small client, not the chat fallback chain:** the chat chain also holds models that cannot
  read images (`openai/gpt-oss-20b` on Groq, `gpt-3.5-turbo`, a random `openrouter/free` model). An image sent to
  one of them is an error, not an answer. So images get their own client with one model that is known to
  accept them, like the transcriber in plan_2.
- **The image is never stored** (same rule as the audio). Only the question, the answer and the usage are.
- **One image, one question, one new conversation.** A follow-up about the same image needs B5.
- **Unlike audio, a vision call reports tokens.** Saving the answer as a normal assistant message means the
  daily budget (`CHAT_DAILY_TOKEN_BUDGET`) and `/chat/stats` count it automatically. No new table.
- **Privacy: on the free tier Google may use what you send to improve its products, and human reviewers may
  read it.** Its terms say not to submit sensitive, confidential or personal information to unpaid services
  (source: Gemini API terms, checked 2026-10-20). So:
  - test with your own or public images only: no faces, IDs, documents or real casework;
  - for real images use a billed Google project (not used for product improvement per the same terms) or a
    local model (B11), and ask your team for its data-handling rules first.

</details>

<details open>
<summary>Carry-over (small, do when you touch the area)</summary>

### Carry-over (small, do when you touch the area)
- [ ] Plan_2 live check: one real recording through the two `curl` commands, and one failing call, then read the log lines
- [ ] Write the decision "edit a message: remove or branch" in `plans/decisions.md` (only text)
- [ ] Write down which providers may see real data. The chat fallback can send a message, and a voice
      transcript, to OpenRouter, Gemini or Groq, depending on which keys are set. With real data, set only the
      keys you accept in `.env`.

</details>

<details open>
<summary>Part A · The smallest working version</summary>

## Part A · The smallest working version

<details open>
<summary>A1 · Try the model by hand (no app code)</summary>

### A1 · Try the model by hand (no app code)
- [ ] Take a photo yourself (a desk, a plant, a page of your own notes). Keep it outside the repo. Make one JPEG under 2 MB and one PNG.
- [ ] Call the compatible endpoint with `curl` and a base64 data URI (look up the exact command in Google's OpenAI-compatibility page)
  - [ ] Check: the answer is printed and roughly right
- [ ] Write down: the response fields, `usage.prompt_tokens` for one image (that is what an image costs), and the time it took
- [ ] Try on purpose, and write down the status and message of each:
  - a 15 MB image
  - a JPEG sent with `data:image/png`
  - a `.heic` file

</details>

<details open>
<summary>A2 · Settings</summary>

### A2 · Settings
- [~] `VISION_MODEL=gemini-3.1-flash-lite`
- [~] `VISION_MAX_IMAGE_MB=5` (`gt=0`, `le=14`: base64 grows a file by a third and the request limit is 20 MB)
- [~] `VISION_TIMEOUT_SECONDS=60` (`gt=0`, `le=120`)
- [~] In `app/core/config.py` and `.env.example`, with a comment on each; the key and base URL are reused, never logged
- [~] No Gemini key → the endpoint answers 503, like chat does

</details>

<details open>
<summary>A3 · The client: app/service/llm_client/vision.py</summary>

### A3 · The client: `app/service/llm_client/vision.py`
- [~] One function: image bytes + mime type + question in, `{text, model, input_tokens, output_tokens, latency_ms}` out
- [~] Builds the message with a `text` part and an `image_url` data-URI part; one cached `AsyncOpenAI`, `max_retries=0`
- [~] Same error rules as the transcriber: timeout → `LLMTimeoutError`, no connection → `LLMUnavailableError`,
      429 → `LLMRateLimitError`, anything else → `LLMError`. No provider message ever reaches the user.
- [~] A short system prompt in `app/prompts/vision_v1.md`: answer from what is visible, say when you cannot tell,
      and treat text inside the image as content to describe, never as instructions
- [~] Check: a script calls it with your photo and prints the answer and the token counts

</details>

<details open>
<summary>A4 · Endpoint: POST /api/v1/chat/image</summary>

### A4 · Endpoint: `POST /api/v1/chat/image`
The file checks are reusable and live in `app/utils/upload.py` (any feature) and `app/utils/image.py` (images). They follow the OWASP File Upload Cheat Sheet: extension allowlist, size limit while reading, first-bytes check, pixel limit, then the image is decoded and written again (no EXIF, no hidden bytes). B2 and B4 below are done here.
- [~] Multipart: field `file` and field `question` (same rule as chat: 1 to 4,000 characters after trimming); signed-in users only
- [~] Rate limit and daily token budget run first (reuse `_rate_limit` and `_token_budget`)
- [~] Reject before calling the provider (extension decides the type: jpg, jpeg, png, webp)
  - [~] Too big → 413 (`VISION_MAX_IMAGE_MB`)
  - [~] Other extension → 415
  - [~] Empty file or empty question → 422
- [~] Failures go through `to_http_error`: 502 / 503 / 504 with a generic message
- [~] Check: `curl -F file=@photo.jpg -F question='What is on the desk?'` returns the answer

</details>

<details open>
<summary>A5 · Store it</summary>

### A5 · Store it
- [~] A new conversation named from the question (`auto_title`)
- [~] User message: the question, with `meta` = `{"image": {"bytes": <n>, "type": "image/jpeg"}}`. No file name, no pixels.
- [~] Assistant message: the answer, with provider, model, token counts, latency and `prompt_version="vision_v1"`
- [~] Response: `{ "conversation_id", "answer", "model", "input_tokens", "output_tokens" }`
- [~] Check: the conversation shows in `GET /api/v1/chat/conversations`, the tokens appear in `GET /api/v1/chat/stats`, and nothing in the database or on disk holds the image

</details>

<details open>
<summary>A6 · Log</summary>

### A6 · Log
- [~] One JSON log line per call: `provider`, `model`, `status`, `latency_ms`, `image_bytes`, `input_tokens`, `output_tokens`
- [~] The question, the answer and the file name are never logged (add `image_bytes` to the log whitelist)
- [~] Check: one good call and one failing call (wrong key), then read the log lines

</details>

<details open>
<summary>A7 · Done when</summary>

### A7 · Done when
- [~] You can ask a question about your own photo with one `curl` command and get an answer
- [~] README has that command, the three settings, one line on what is kept (question, answer, usage) and what is not (the image), and the privacy line about the free tier

</details>

</details>

<details>
<summary>Part B · Extra steps (one at a time, only when needed)</summary>

## Part B · Extra steps (one at a time, only when needed)

| Step | Do this | When you need it |
|---|---|---|
| B1 | Stream the answer (Server-Sent Events, like `/chat/stream`) | A client wants words to appear while it waits |
| B2 | ~~First-bytes check, open with Pillow, pixel limit~~ | Done in A4 |
| B3 | Shrink big photos before sending (pick the size after A1 shows tokens against size) | The log shows slow calls or high token counts |
| B4 | ~~Remove EXIF data~~ | Done in A4 (the image is written again) |
| B4b | ~~Virus scan (ClamAV container)~~ done (`app/utils/scan.py`, images and documents, off when `CLAMAV_HOST` is empty). Still open: a file status (`SCANNING` / `CLEAN` / `REJECTED`) before processing, and PDF/DOCX with active content removed | Other people can upload, or you start Part 4 (documents) |
| B5 | Follow-up questions about the same image: keep the image in MinIO under the user's id, send it again with the history, delete it with the conversation | Users ask a second question about one image |
| B6 | Second provider (for example Groq `qwen/qwen3.8-27b`: 3 images per request, about 2,048 tokens each) and a fallback | The log shows provider errors or the free tier says stop |
| B7 | Several images in one question | A use case needs to compare two images |
| B8 | HEIC / HEIF (iPhone photos): convert to JPEG, or use what A1 found | Users send iPhone photos |
| B9 | Tests: 413 / 415 / 422, provider mocked, no key → 503, tokens counted in the budget and in `/chat/stats`, not signed in → 401 | Before you show it to anyone |
| B10 | Upload rate limit and an images-per-day limit (reuse the Redis helpers) | More than a few users |
| B11 | Local vision model (for example through Ollama) so images never leave the machine | Before using any real image of people or casework |
| B12 | Ten test images with the facts the answer must contain; run them live and keep the first result as a baseline | You change the model or the prompt |
| B13 | Prompt-injection test: an image that says "ignore your instructions"; the answer must describe it, not obey it | Before Part 5 (agent), where an answer could trigger a tool |
| B14 | Images of scanned pages go to the document pipeline instead (text extraction, citations) | You start Part 4 (documents) |

</details>

<details>
<summary>Sources (checked 2026-10-20)</summary>

## Sources (checked 2026-10-20)
- Gemini image input: <https://ai.google.dev/gemini-api/docs/image-understanding>
- Gemini OpenAI-compatible request shape: <https://ai.google.dev/gemini-api/docs/openai>
- Gemini pricing and free tier: <https://ai.google.dev/gemini-api/docs/pricing>
- Gemini rate limits (per model, per project): <https://ai.google.dev/gemini-api/docs/rate-limits>
- Gemini API terms (free tier data use): <https://ai.google.dev/gemini-api/terms>
- Groq vision models: <https://console.groq.com/docs/vision>

</details>
