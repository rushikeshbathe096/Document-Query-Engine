from typing import List, Dict, Any
import numpy as np

from app.persistence import repository
from app.retrieval.bm25_index import BM25Index
from app.retrieval.faiss_index import FAISSIndex
from app.retrieval.embeddings import Embedder
from app.retrieval.reranker import Reranker


def hybrid_search(
    document_id: int,
    query: str,
    bm25_k: int,
    faiss_k: int,
    db: Any,
) -> Dict[str, Any]:
    embedder = Embedder()

    chunks = repository.get_chunks_by_document(db, document_id)
    if not chunks:
        return {"chunks": [], "metrics": {}}

    bm25 = BM25Index()
    bm25.index_chunks(chunks)

    index_name = FAISSIndex.index_name_for_document(document_id)
    try:
        faiss_idx = FAISSIndex.load(index_name)
    except FileNotFoundError:
        return {"chunks": [], "metrics": {}}

    keyword_results = bm25.search(query, top_k=bm25_k)

    query_vec = np.array(embedder.model.encode([query])).astype("float32")

    if faiss_k > 0 and faiss_idx.index.ntotal > 0:
        distances, indices = faiss_idx.index.search(query_vec, faiss_k)
        semantic_results = [
            {"chunk_id": faiss_idx.chunk_ids[i], "score": float(distances[0][j])}
            for j, i in enumerate(indices[0])
            if i < len(faiss_idx.chunk_ids)
        ]
    else:
        semantic_results = []

    merged = _merge_results(keyword_results, semantic_results)

    chunk_map = {c["chunk_id"]: c for c in chunks}
    evidence_chunks = []
    for item in merged:
        if item["chunk_id"] in chunk_map:
            c = chunk_map[item["chunk_id"]].copy()
            c["score"] = item["score"]
            evidence_chunks.append(c)

    metrics = {
        "normalized_score": evidence_chunks[0]["score"] if evidence_chunks else 0.0,
        "bm25_count": len(keyword_results),
        "faiss_count": len(semantic_results),
        "merged_count": len(merged),
    }

    return {"chunks": evidence_chunks, "metrics": metrics}


def _normalize_scores(results: List[Dict[str, Any]], invert: bool = False) -> List[Dict[str, Any]]:
    if not results:
        return []

    scores = [r["score"] for r in results]
    min_s = min(scores)
    max_s = max(scores)
    denom = max_s - min_s

    normalized = []
    for r in results:
        s = r["score"]
        norm_s = 1.0 if denom == 0 else (s - min_s) / denom
        if invert:
            norm_s = 1.0 - norm_s
        normalized.append({"chunk_id": r["chunk_id"], "norm_score": norm_s})
    return normalized


def _merge_results(
    keyword_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    norm_keyword = _normalize_scores(keyword_results, invert=False)
    norm_semantic = _normalize_scores(semantic_results, invert=True)

    merged = {}
    for item in norm_keyword:
        cid = item["chunk_id"]
        if cid not in merged:
            merged[cid] = {"chunk_id": cid, "norm_keyword_score": 0.0, "norm_semantic_score": 0.0}
        merged[cid]["norm_keyword_score"] = item["norm_score"]

    for item in norm_semantic:
        cid = item["chunk_id"]
        if cid not in merged:
            merged[cid] = {"chunk_id": cid, "norm_keyword_score": 0.0, "norm_semantic_score": 0.0}
        merged[cid]["norm_semantic_score"] = item["norm_score"]

    reranker = Reranker()
    reranked = reranker.rerank(list(merged.values()))

    for item in reranked:
        item["score"] = (
            item.get("norm_keyword_score", 0.0) +
            item.get("norm_semantic_score", 0.0)
        ) / 2

    return reranked
