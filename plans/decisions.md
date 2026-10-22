# Decisions

Short notes on choices that are easy to forget the reason for. Newest first. Only text, no code.

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
