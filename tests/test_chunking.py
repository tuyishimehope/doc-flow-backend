"""Chunking is a pure function, so these tests need no database and no network."""

import pytest

from app.utils.chunking import CHARS_PER_TOKEN, Chunk, chunk_pages
from app.utils.pdf import Page

SENTENCE = "The notice period is thirty days and it starts on the day of the letter. "


def normal(text: str) -> str:
    return " ".join(text.split())


def long_page(number: int = 1, sentences: int = 120) -> Page:
    return Page(number, "".join(f"Sentence {i}: {SENTENCE}" for i in range(sentences)))


def test_a_short_page_is_one_chunk_with_its_page_and_position():
    chunks = chunk_pages([Page(1, "Hello there."), Page(2, "Second page.")])

    assert chunks == [Chunk(1, 0, "Hello there."), Chunk(2, 1, "Second page.")]


def test_blank_pages_give_no_chunks_and_do_not_use_a_position():
    chunks = chunk_pages([Page(1, ""), Page(2, "  \n \n"), Page(3, "Text.")])

    assert chunks == [Chunk(3, 0, "Text.")]


@pytest.mark.parametrize(("target", "overlap"), [(100, 0), (100, 20), (400, 50), (30, 10)])
def test_no_chunk_is_over_the_limit(target, overlap):
    chunks = chunk_pages([long_page(1), long_page(2)], target_tokens=target, overlap=overlap)

    assert len(chunks) > 2
    assert max(len(chunk.text) for chunk in chunks) <= target * CHARS_PER_TOKEN


def test_a_chunk_never_crosses_a_page():
    pages = [Page(1, "Alpha beta gamma. " * 80), Page(2, "Delta epsilon zeta. " * 80)]

    chunks = chunk_pages(pages, target_tokens=50, overlap=5)

    assert {chunk.page for chunk in chunks} == {1, 2}
    for chunk in chunks:
        other = "Delta" if chunk.page == 1 else "Alpha"
        assert other not in chunk.text
    assert [chunk.position for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.page for chunk in chunks] == sorted(chunk.page for chunk in chunks)


def test_without_overlap_the_chunks_put_together_give_back_the_page():
    page = long_page(1, 60)

    chunks = chunk_pages([page], target_tokens=80, overlap=0)

    assert normal(" ".join(chunk.text for chunk in chunks)) == normal(page.text)


def test_with_overlap_every_chunk_is_text_of_the_page_and_nothing_is_lost():
    page = long_page(1, 60)
    whole = normal(page.text)

    chunks = chunk_pages([page], target_tokens=80, overlap=20)

    covered = 0
    previous_start = -1
    for chunk in chunks:
        text = normal(chunk.text)
        start = whole.find(text, previous_start + 1)
        assert start != -1, "a chunk is not a stretch of the page"
        # only whitespace may lie between the end of one chunk and the start of the next
        assert whole[covered:start].strip() == "", "text between two chunks is missing"
        covered = max(covered, start + len(text))
        previous_start = start
    assert covered == len(whole)


def test_neighbouring_chunks_share_text_when_there_is_overlap():
    chunks = chunk_pages([long_page(1, 60)], target_tokens=80, overlap=20)

    shared = [
        a.text.split()[-3:] == b.text.split()[:3]
        or set(a.text.split()[-6:]) & set(b.text.split()[:12])
        for a, b in zip(chunks, chunks[1:], strict=False)
    ]
    assert all(shared)


def test_paragraphs_are_kept_whole_when_they_fit():
    paragraph_a = "First paragraph, short."
    paragraph_b = "Second paragraph, also short."
    # the two do not fit together in 8 tokens (32 characters), so each is a chunk of its own
    chunks = chunk_pages([Page(1, f"{paragraph_a}\n\n{paragraph_b}")], target_tokens=8, overlap=0)

    assert [chunk.text for chunk in chunks] == [paragraph_a, paragraph_b]


def test_a_long_paragraph_is_split_at_sentences_not_in_the_middle_of_one():
    sentences = [f"This is sentence number {i} of the paragraph." for i in range(30)]
    chunks = chunk_pages([Page(1, " ".join(sentences))], target_tokens=40, overlap=0)

    assert len(chunks) > 3
    for chunk in chunks:
        assert chunk.text.endswith("paragraph."), chunk.text
        assert chunk.text.startswith("This is sentence")


def test_a_sentence_longer_than_a_chunk_is_split_between_words():
    words = [f"word{i}" for i in range(400)]
    chunks = chunk_pages([Page(1, " ".join(words))], target_tokens=20, overlap=0)

    assert len(chunks) > 4
    seen = [word for chunk in chunks for word in chunk.text.split()]
    assert seen == words  # every word whole, in order, none lost


def test_one_word_longer_than_a_chunk_is_the_only_thing_that_gets_cut():
    chunks = chunk_pages([Page(1, "start " + "x" * 100 + " end")], target_tokens=10, overlap=0)

    assert all(len(chunk.text) <= 40 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks).replace(" ", "") == "start" + "x" * 100 + "end"


def test_text_without_any_spaces_or_punctuation_still_ends():
    chunks = chunk_pages([Page(1, "a" * 1000)], target_tokens=25, overlap=5)

    assert len(chunks) == 10
    assert all(len(chunk.text) == 100 for chunk in chunks)


@pytest.mark.parametrize(("target", "overlap"), [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)])
def test_impossible_limits_are_refused(target, overlap):
    with pytest.raises(ValueError):
        chunk_pages([Page(1, "text")], target_tokens=target, overlap=overlap)


def test_the_default_size_makes_about_as_many_chunks_as_expected():
    # 60 pages of about 3,000 characters: around 750 tokens a page, so two or three chunks each
    page_text = "".join(f"Fact {i}: {SENTENCE}" for i in range(36))
    chunks = chunk_pages([Page(n, page_text) for n in range(1, 61)])

    per_page = len(chunks) / 60
    assert 2 <= per_page <= 4


def test_the_overlap_never_pushes_a_chunk_over_the_limit_when_a_long_sentence_follows():
    short = ("word " * 19).strip() + "."  # about 100 characters
    long = (
        "word " * 75
    ).strip() + "."  # about 375 characters: fits alone, not after the short one
    chunks = chunk_pages([Page(1, f"{short} {long}")], target_tokens=100, overlap=50)

    assert [chunk.text for chunk in chunks] == [short, long]  # the overlap had to give way
    assert max(len(chunk.text) for chunk in chunks) <= 100 * CHARS_PER_TOKEN
