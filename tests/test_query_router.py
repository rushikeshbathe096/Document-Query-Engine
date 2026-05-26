from app.reasoning.router import route_query


def test_definition_query_routing():
    route = route_query({"query_type": "definition"})
    assert route["strategy"] == "hybrid"
    assert route["bm25_k"] > 0
    assert route["faiss_k"] > 0


def test_lookup_query_uses_bm25_only():
    route = route_query({"query_type": "lookup"})
    assert route["faiss_k"] == 0
    assert route["bm25_k"] > 0


def test_conceptual_prefers_faiss():
    route = route_query({"query_type": "conceptual"})
    assert route["faiss_k"] > route["bm25_k"]


def test_unknown_fallback_is_safe():
    route = route_query({})
    assert route["strategy"] == "hybrid"
    assert route["bm25_k"] > 0
    assert route["faiss_k"] > 0
