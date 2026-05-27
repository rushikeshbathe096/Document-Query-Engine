from fastapi.testclient import TestClient

from app.main import app
from app.persistence.database import SessionLocal
from app.persistence import repository
from app.ingestion.chunker import get_chunker


client = TestClient(app)


class _FakeFAISSIndex:
    def __init__(self):
        self.index = type("Index", (), {"ntotal": 0})()
        self.chunk_ids = []


class _FakeSemanticIndex:
    def __init__(self, chunk_id: str):
        class _Index:
            ntotal = 1

            def search(self, query_vec, top_k):
                return [[0.12]], [[0]]

        self.index = _Index()
        self.chunk_ids = [chunk_id]


def _create_resume_document():
    db = SessionLocal()
    try:
        document = repository.create_document(db, "resume.pdf", "resume-hash")

        pages = [
            {
                "page_number": 1,
                "text": (
                    "Candidate Name: Rushikesh Bathe. "
                    "Educational Qualification: Bachelor of Engineering. "
                    "Technical Skills: Python, FastAPI, FAISS, BM25, PostgreSQL. "
                    "Projects: Document Query Engine, Resume Parser."
                ),
            }
        ]

        chunks = get_chunker().chunk_document(pages)
        repository.create_chunks(db, document.id, chunks)
        return document.id, chunks
    finally:
        db.close()


def _patch_query_flow(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.query_analyzer.analyze_query",
        lambda query: {"query_type": "unknown", "keywords": [], "expected_sections": []},
    )

    def fake_generate_answer(query, query_analysis, evidence_chunks):
        lowered_query = query.lower()

        if "name" in lowered_query:
            answer = "Rushikesh Bathe"
        elif "skill" in lowered_query:
            answer = "Python, FastAPI, FAISS, BM25, PostgreSQL"
        elif "education" in lowered_query:
            answer = "Bachelor of Engineering"
        else:
            answer = "Document Query Engine"

        citation = next(
            (
                chunk["chunk_id"]
                for chunk in evidence_chunks
                if answer.lower() in chunk["text"].lower()
            ),
            evidence_chunks[0]["chunk_id"],
        )

        return {
            "answer": answer,
            "citations": [citation],
            "confidence": "high",
        }

    monkeypatch.setattr("app.api.routes.generate_answer", fake_generate_answer)
    monkeypatch.setattr("app.retrieval.hybrid_search.FAISSIndex.load", lambda name: _FakeFAISSIndex())


def test_resume_queries_return_answers_and_debug_results(monkeypatch):
    document_id, _ = _create_resume_document()
    _patch_query_flow(monkeypatch)

    queries = {
        "What is the candidate name?": "Rushikesh Bathe",
        "What technical skills does the candidate have?": "Python, FastAPI, FAISS, BM25, PostgreSQL",
        "What is the educational qualification?": "Bachelor of Engineering",
        "What projects has the candidate worked on?": "Document Query Engine",
    }

    for query, expected_answer in queries.items():
        response = client.post(f"/documents/{document_id}/query", json={"query": query})
        assert response.status_code == 200
        payload = response.json()

        assert payload["answer"] != "Not Found"
        assert expected_answer.lower() in payload["answer"].lower()
        assert payload["confidence"] > 0

        debug_response = client.get(f"/documents/{document_id}/debug", params={"query": query})
        assert debug_response.status_code == 200
        debug_payload = debug_response.json()

        assert debug_payload["query"] == query
        assert debug_payload["query_type"] == "unknown"
        assert debug_payload["bm25_results"]
        assert debug_payload["final_results"]
        assert all("preview" in result for result in debug_payload["final_results"])


def test_debug_endpoint_includes_semantic_results_shape(monkeypatch):
    document_id, chunks = _create_resume_document()
    _patch_query_flow(monkeypatch)

    monkeypatch.setattr(
        "app.retrieval.hybrid_search.FAISSIndex.load",
        lambda name: _FakeSemanticIndex(chunks[0]["chunk_id"]),
    )

    response = client.get(
        f"/documents/{document_id}/debug",
        params={"query": "What is the candidate name?"},
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["semantic_results"]
    semantic_item = payload["semantic_results"][0]
    assert set(semantic_item.keys()) == {"chunk_id", "score", "preview"}
    assert semantic_item["chunk_id"] == chunks[0]["chunk_id"]
    assert isinstance(semantic_item["score"], float)
    assert semantic_item["preview"]