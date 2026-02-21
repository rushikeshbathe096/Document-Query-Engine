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
        overlap_sentences=2,
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
    text = "A one. B two. C three. D four. E five. F six."

    chunks = chunker.chunk_page(1, text)

    if len(chunks) > 1:
        first = chunks[0]["text"]
        second = chunks[1]["text"]

        overlap = first.split(".")[-3:-1]
        for part in overlap:
            assert part.strip() in second


def test_output_schema():
    chunker = get_test_chunker()
    chunks = chunker.chunk_page(1, "Simple sentence.")

    chunk = chunks[0]
    assert set(chunk.keys()) == {"chunk_id", "text", "page_number", "position"}