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
