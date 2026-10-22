# Evals

`chat.json` holds test cases for the chat prompt in `app/prompts/chat_v1.md`.

Each case: `id`, `category`, `message` (what the user sends), `checks`, `why`.

Checks (all that are present must pass; text comparison ignores case):
- `contains_all`: every string must appear in the answer
- `contains_any`: at least one string must appear
- `not_contains`: none of these may appear
- `max_words`: the answer has at most this many words

Keyword checks are cheap and predictable but brittle. When a correct answer fails
only because of wording, widen `contains_any` or move that case to an LLM judge later.
Change the prompt -> bump the version (`chat_v2.md`) and rerun every case.

Run them with `python -m scripts.run_evals` (see the main README). Every run is saved in
`results/` with its score; `--compare OLD NEW` lists the cases that got worse or better.

## Document evals (`documents.json`)

`documents.example.json` shows the format of the questions for `python -m scripts.run_doc_evals`.
Copy it to `documents.json` and write your own: it needs a real PDF, so none is checked in.

- `document`: the PDF, relative to the cases file (or pass `--pdf`, or `--document-id` for one
  that is already stored).
- Each case: `id`, `question`, the page(s) of the answer, `checks` (the same checks as above),
  `why`.
  - `"page": 12` or `"pages": [18, 19]` (any of them counts): the answer is in the document.
    The case passes when the model says it found it, the checks pass, **and** a citation that the
    API verified is on one of those pages.
  - `"page": null`: the answer is **not** in the document. The case passes only when the model
    says it could not find it and cites nothing. Write at least three of these: they catch a
    model that makes something up, and a retrieval that returns the wrong chunk with confidence.
- Page numbers are the ones a PDF viewer shows, starting at 1.

Run the baseline (`--modes pages`) before you change anything, then both modes after. The run is
saved as `results/docs-<time>.json`; the printed table is the C0/C7 comparison: right answers,
right pages, refusals, input tokens per question, seconds. The rule it prints (tokens down at
least 50% and the score not worse) is a starting point, not a verdict: read the failed cases.
