import pytest
from app.ingestion.chunker import Chunker, Tokenizer


class DummyTokenizer(Tokenizer):
    def count_tokens(self, text: str) -> int:
        return len(text.split())


def get_test_chunker():
    return Chunker(
        tokenizer=DummyTokenizer(),
        target_tokens=10,
        max_tokens=15,
        overlap_tokens=2,
    )


def test_deterministic_output():
    chunker = get_test_chunker()
    text = "One sentence. Another sentence. Third sentence."

    out1 = chunker.chunk_page(1, text)
    out2 = chunker.chunk_page(1, text)

    assert out1 == out2


def test_page_boundaries():
    chunker = get_test_chunker()
    pages = [
        {"page_number": 1, "text": "Page one sentence. Another one."},
        {"page_number": 2, "text": "Page two sentence. Another one."},
    ]

    chunks = chunker.chunk_document(pages)

    assert all(c["page_number"] == 1 for c in chunks[:1])
    assert all(c["page_number"] == 2 for c in chunks[1:])


def test_token_limits_respected():
    chunker = get_test_chunker()
    text = " ".join(["long sentence"] * 20)

    chunks = chunker.chunk_page(1, text)

    for c in chunks:
        assert chunker.tokenizer.count_tokens(c["text"]) <= 15


def test_overlap_semantics():
    chunker = get_test_chunker()
    text = "One. Two. Three. Four. Five. Six."

    chunks = chunker.chunk_page(1, text)

    if len(chunks) > 1:
        first_words = set(chunks[0]["text"].replace(".", "").split())
        second_words = set(chunks[1]["text"].replace(".", "").split())

        assert first_words & second_words


def test_output_schema():
    chunker = get_test_chunker()
    chunks = chunker.chunk_page(1, "Simple sentence.")

    chunk = chunks[0]
    assert set(chunk.keys()) == {"chunk_id", "text", "page_number", "position"}