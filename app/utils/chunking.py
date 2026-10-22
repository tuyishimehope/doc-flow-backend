"""Split the text of a PDF into chunks that are small enough to embed and to quote from.

Rules (plan_4, C3):

* A chunk never crosses a page, so every citation has exactly one page.
* Paragraphs first (a blank line), then sentences when a paragraph is too long, then words
  when a sentence is too long. A chunk is never cut inside a word; the one exception is a
  single "word" longer than a whole chunk (a long URL, a line of noise), which is cut.
* Neighbouring chunks of a page share a little text (the overlap), so a sentence that falls
  on a boundary is whole in at least one chunk.

There is no tokenizer here. A token is taken as ``CHARS_PER_TOKEN`` characters, which is about
right for English; text in other languages uses more tokens per character, so such chunks are
a bit bigger in tokens than asked. All limits are far below what the embedding model accepts.
"""

import re
from dataclasses import dataclass

from app.utils.pdf import Page

CHARS_PER_TOKEN = 4
DEFAULT_TARGET_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 50

_PARAGRAPH = re.compile(r"\n[ \t]*\n\s*")  # a blank line, with the whitespace after it
_SENTENCE = re.compile(r"(?<=[.!?])\s+")  # the whitespace after . ! or ?
_WORD = re.compile(r"\S+\s*|\s+")  # a word with the whitespace after it


@dataclass(frozen=True)
class Chunk:
    page: int  # the page number it comes from, starting at 1
    position: int  # 0, 1, 2 ... over the whole document, in reading order
    text: str


def chunk_pages(
    pages: list[Page],
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunks of at most ``target_tokens`` tokens, in page order. Blank pages give none."""
    if target_tokens < 1:
        raise ValueError("target_tokens must be at least 1")
    if not 0 <= overlap < target_tokens:
        raise ValueError("overlap must be at least 0 and less than target_tokens")
    max_chars = target_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap * CHARS_PER_TOKEN

    chunks: list[Chunk] = []
    for page in pages:
        for text in _pack(_pieces(page.text, max_chars), max_chars, overlap_chars):
            chunks.append(Chunk(page=page.number, position=len(chunks), text=text))
    return chunks


def _split_after(pattern: re.Pattern[str], text: str) -> list[str]:
    """Cut ``text`` after every match, so the pieces joined together are the text again."""
    pieces: list[str] = []
    position = 0
    for match in pattern.finditer(text):
        pieces.append(text[position : match.end()])
        position = match.end()
    if position < len(text):
        pieces.append(text[position:])
    return pieces


def _pieces(text: str, max_chars: int) -> list[str]:
    """Paragraphs, or sentences, or words: the largest pieces that each fit in a chunk."""
    pieces: list[str] = []
    for paragraph in _split_after(_PARAGRAPH, text):
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
            continue
        for sentence in _split_after(_SENTENCE, paragraph):
            if len(sentence) <= max_chars:
                pieces.append(sentence)
            else:
                pieces.extend(_split_words(sentence, max_chars))
    return pieces


def _split_words(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for word in _WORD.findall(text):
        if len(word) > max_chars:  # one word longer than a chunk: the only place we cut a word
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(word[i : i + max_chars] for i in range(0, len(word), max_chars))
        elif len(current) + len(word) > max_chars:
            pieces.append(current)
            current = word
        else:
            current += word
    if current:
        pieces.append(current)
    return pieces


def _pack(pieces: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    """Join neighbouring pieces up to ``max_chars``; each new chunk starts with the last pieces
    of the one before, as many as fit in the overlap."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for piece in pieces:
        if current and size + len(piece) > max_chars:
            chunks.append("".join(current))
            current = _tail(current, min(overlap_chars, max_chars - len(piece)))
            size = sum(len(item) for item in current)
        current.append(piece)
        size += len(piece)
    if current:
        chunks.append("".join(current))
    return [text.strip() for text in chunks if text.strip()]


def _tail(pieces: list[str], limit: int) -> list[str]:
    """The last pieces whose total length is at most ``limit``."""
    kept: list[str] = []
    total = 0
    for piece in reversed(pieces):
        if total + len(piece) > limit:
            break
        kept.append(piece)
        total += len(piece)
    kept.reverse()
    return kept
