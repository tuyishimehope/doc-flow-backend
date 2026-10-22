# Decisions

Short notes on choices that are easy to forget the reason for. Newest first. Only text, no code.

## 2026-10-22 · Part C (plan_4): how retrieval is built

**Question.** Part C changes which text goes into the prompt. Four choices could have gone another way.

**1. Where the chunk state lives: a new column, not new values of `Document_Status`.** `Document_Status` is the
document's life (active, archived, deleted) and its Postgres enum would need a migration to grow. Indexing is
background work that can fail and be repeated, so it is `document.index_status` (`NONE`, `PROCESSING`, `READY`,
`FAILED`), a plain string column with a default. A document can be archived and `READY` at the same time.

**2. Retrieval is off by default, and `mode` decides per question.** `DOCUMENT_RETRIEVAL_ENABLED=false` means no
embedding is ever started and `auto` reads the whole text, so merging Part C changes nothing for anyone. `mode`
(`auto`, `pages`, `chunks`) exists so the two can be compared on the same document (C7) without a redeploy.
`chunks` works for a `READY` document even with the flag off. A document too long to read whole can only be asked
in `chunks` mode (409 with the reason until it is `READY`), and the upload page limit is
`DOCUMENT_RAG_MAX_PAGES` only while the flag is on.

**3. Vectors: 768 numbers, cut and scaled by us, model name stored on every row.** The model returns longer
vectors. Cutting and scaling in `fit_vector` works for any model and does not depend on the endpoint accepting a
`dimensions` field. A vector is only meaningful next to vectors of the same model, so `embedding_model` is stored,
the query filters on the current one, and a model change gives 409 ("index it again") instead of silently
searching nothing. 768 numbers per chunk, not 3,072, is about a quarter of the storage.

**4. Safety of the chunk table.** `user_id` is a column of `document_chunk` and part of the retrieval query, next
to `document_id`, so a caller who passes someone else's document id still gets nothing back. The ask endpoint
checks ownership first as well (404), so this is the second lock, and it has its own test. Deleting a document or
its file deletes the chunks for real, in the same transaction as the page text.

**What follows from it.**

- A citation is checked against the text of the chunks the model was given, not the whole document. A quote that
  is a real sentence of the PDF but was not retrieved is dropped. Because a chunk never crosses a page, the page
  of a citation is still exactly one number.
- Chunk text is in `document_chunk` only. Saved messages keep ids, pages and distances of what was retrieved
  (`metadata.retrieval`), logs keep ids and distances, never text.
- Indexing replaces the chunks only after the new embedding succeeded, so a failed re-index leaves the old,
  usable chunks in place (and the status `FAILED`, which the next `POST /index` retries).
- The worker makes a new Gemini client for each task and closes it, because an HTTP client must not outlive the
  event loop it was made in. The API keeps using the shared client.
- Text inside a chunk is defused only against its own tag (`<chunk`, `</chunk`), like pages are. A fake
  `<question>` inside a chunk stays inside that chunk, and the prompt says chunk text is never an instruction.
  D10 is where this gets tested against a real injection PDF.
- No vector index (HNSW). An exact scan took 9 to 11 ms for 1,000 chunks of one document among 40,000 rows, since
  the `(user_id, document_id)` index narrows it first. D3 is for later.

**Not decided here (trigger to decide it).**

- Whether Part C stays. That is C7: needs the baseline and the comparison with your key and a real PDF. If the
  numbers say no, set the flag to false and leave the code; the cost of keeping it is one table.
- Which embedding model and whether a list of texts per request works on the compatible endpoint (C2 live check).
- Re-indexing everything after a model change: today it is one `POST /index` per document. A loop is worth writing
  when there is more than a handful of documents.

## 2026-10-22 · B1 (plan_4): how a PDF is kept for questions

**Question.** Users want a second question about the same PDF without uploading it again. The PDF and
the text of each page must be kept. Two ways to get there:

1. **A new `Processing_Type`** (for example `DOCUMENT_PAGES`) through the existing `POST /documents`,
   `process_document` and the Celery worker.
2. **A new small endpoint**, `POST /api/v1/documents/qa`, that does the work inside the request.

**Chosen: 2.**

- Option 1 is asynchronous. `process_document` saves the file and queues a task, and the worker rejects any
  type except the three it knows (`DOCUMENT_SUMMARY`, `INVOICE_EXTRACTION`, `CONTRACT_METADATA`). Page
  extraction would need a new branch in the worker, a sync copy of the save code, and a status the client
  must poll before the first question. All that to read at most 40 pages, which the Part A route already
  does in a thread in well under a second.
- Option 2 reuses what Part A built: `read_checked_upload`, `scan_upload` and `extract_pages`. It then writes
  the same `File` and `Document` rows as the old upload, so the document list, `GET /files/{id}`, rename,
  archive and delete keep working. The new part is one table, `document_page`.
- The old upload route is unchanged.

**What follows from it.**

- A document uploaded the old way has no pages. Asking it returns 409 with the instruction to upload it again
  with `/documents/qa`. No back-fill, because nobody has asked for one.
- The MinIO object and the database rows are written together: if the page rows fail, the object is removed
  again (`store_pdf`).
- The content type stored for the file is always `application/pdf`, never the client's value.
- The pages are sent again with every question, so a long document costs the same input tokens each time.
  This is what Part C (chunks and retrieval) is for. The `document_page` rows are where its chunks come from.
- A conversation about a document has `conversation.document_id`. A follow-up must name a conversation of the
  same document (404 if it is not the user's, 409 if it is about another document or none).
- Follow-ups send the last `CHAT_HISTORY_MESSAGES` messages, trimmed to `CHAT_HISTORY_MAX_CHARS`, as plain
  question and answer text. Citations are never sent back: the model only sees the pages.

**Delete (B5).** The plan says "same soft-delete rule as conversations", but conversations are deleted for
real today. So: the document keeps the existing soft delete (the row is hidden and the stored file is
removed), and the **page text is deleted for real at the same moment**, because it is a copy of the file's
content. Conversations about the document stay: they are the user's own and can be deleted by the user.
New questions about a deleted document return 404. When conversations get a soft delete, this stays as it is.

**Not decided here (trigger to decide it).**

- A cap on stored PDFs per user. Today any signed-in user can store as many as the rate limit allows. Decide
  before more than a few people use it.
- Edit a message: remove the later messages or branch the conversation (still owed from plan_1).
