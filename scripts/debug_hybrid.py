from app.retrieval.hybrid_search import HybridSearch

# Fake BM25 output (higher = better)
bm25_results = [
    {"chunk_id": "c1", "score": 3.0},
    {"chunk_id": "c2", "score": 1.5},
    {"chunk_id": "c3", "score": 0.1},
]

# Fake FAISS output (lower distance = better)
faiss_results = [
    {"chunk_id": "c1", "score": 0.2},
    {"chunk_id": "c4", "score": 0.4},
    {"chunk_id": "c2", "score": 1.2},
]

hybrid = HybridSearch()

results = hybrid.merge_results(
    keyword_results=bm25_results,
    semantic_results=faiss_results
)

for r in results:
    print(r)