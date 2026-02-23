from app.reasoning.answer_generator import generate_answer
from app.llm import gemini


def test_refuses_on_empty_evidence(monkeypatch):
    def fake_llm(_):
        return '{"answer":"x","citations":["c1"],"confidence":"high"}'

    monkeypatch.setattr(gemini, "generate_with_gemini", fake_llm)

    result = generate_answer(
        query="anything",
        query_analysis={},
        evidence_chunks=[]
    )

    assert result["answer"] == "INSUFFICIENT_EVIDENCE"


def test_refuses_on_invalid_citation(monkeypatch):
    def fake_llm(_):
        return '{"answer":"ok","citations":["fake_id"],"confidence":"high"}'

    monkeypatch.setattr(gemini, "generate_with_gemini", fake_llm)

    result = generate_answer(
        query="q",
        query_analysis={},
        evidence_chunks=[{"chunk_id": "real_id", "text": "some text"}]
    )

    assert result["answer"] == "INSUFFICIENT_EVIDENCE"


def test_accepts_valid_answer(monkeypatch):
    def fake_llm(_):
        return '{"answer":"ok","citations":["real_id"],"confidence":"high"}'

    monkeypatch.setattr(gemini, "generate_with_gemini", fake_llm)

    result = generate_answer(
        query="q",
        query_analysis={},
        evidence_chunks=[{"chunk_id": "real_id", "text": "some text"}]
    )

    assert result["answer"] == "ok"
    assert result["citations"] == ["real_id"]