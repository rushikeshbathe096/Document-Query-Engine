import json
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

from app.persistence import repository
from app.retrieval.bm25_index import BM25Index
from app.retrieval.faiss_index import FAISSIndex
from app.retrieval.embeddings import Embedder
from app.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


def _structured_log(event: str, **fields: Any) -> None:
    logger.debug("%s %s", event, json.dumps(fields, default=str, ensure_ascii=False))


def _preview(text: str, limit: int = 150) -> str:
    return (text or "")[:limit]


def _is_short_query(query: str) -> bool:
    return len((query or "").split()) < 6


def _search_weights(query: str) -> Tuple[float, float]:
    if _is_short_query(query):
        return 0.6, 0.4
    return 0.5, 0.5


def _log_results(stage: str, results: List[Dict[str, Any]], chunk_map: Dict[str, Dict[str, Any]]) -> None:
    _structured_log(
        "retrieval_stage",
        stage=stage,
        chunk_count=len(results),
        chunks=[
            {
                "chunk_id": item.get("chunk_id"),
                "score": float(item.get("score", item.get("combined_score", 0.0))),
                "preview": _preview(chunk_map.get(item.get("chunk_id"), {}).get("text", "")),
            }
            for item in results
        ],
    )


def _format_debug_results(results: List[Dict[str, Any]], chunk_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted = []
    for item in results:
        chunk = chunk_map.get(item.get("chunk_id"))
        if not chunk:
            continue
        formatted.append(
            {
                "chunk_id": item["chunk_id"],
                "score": float(item.get("score", item.get("combined_score", 0.0))),
                "preview": _preview(chunk.get("text", ""), 200),
            }
        )
    return formatted


def _prepare_retrieval(
    document_id: int,
    query: str,
    bm25_k: int,
    faiss_k: int,
    db: Any,
    query_type: str = "unknown",
) -> Dict[str, Any]:
    _structured_log(
        "retrieval_query",
        document_id=document_id,
        query=query,
        query_type=query_type,
    )

    chunks = repository.get_chunks_by_document(db, document_id)
    if not chunks:
        return {
            "bm25_results": [],
            "semantic_results": [],
            "hybrid_results": [],
            "final_results": [],
            "metrics": {
                "normalized_score": 0.0,
                "bm25_count": 0,
                "faiss_count": 0,
                "merged_count": 0,
                "supporting_chunks": 0,
            },
            "chunk_map": {},
            "raw_final_results": [],
        }

    chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}

    bm25 = BM25Index()
    bm25.index_chunks(chunks)

    index_name = FAISSIndex.index_name_for_document(document_id)
    try:
        faiss_idx = FAISSIndex.load(index_name)
    except FileNotFoundError:
        faiss_idx = None

    keyword_results = bm25.search(query, top_k=bm25_k)
    _log_results("bm25", keyword_results, chunk_map)

    semantic_results: List[Dict[str, Any]] = []
    if faiss_idx is not None and faiss_k > 0 and faiss_idx.index.ntotal > 0:
        embedder = Embedder()
        query_vec = np.array(embedder.model.encode([query])).astype("float32")
        distances, indices = faiss_idx.index.search(query_vec, faiss_k)
        semantic_results = [
            {"chunk_id": faiss_idx.chunk_ids[i], "score": float(distances[0][j])}
            for j, i in enumerate(indices[0])
            if i < len(faiss_idx.chunk_ids)
        ]

    _log_results("semantic", semantic_results, chunk_map)

    hybrid_results = _merge_results(keyword_results, semantic_results, query=query)
    _log_results("hybrid", hybrid_results, chunk_map)

    final_results = []
    for item in hybrid_results:
        chunk = chunk_map.get(item["chunk_id"])
        if not chunk:
            continue
        enriched = chunk.copy()
        combined_score = float(item.get("combined_score", item.get("score", 0.0)))
        enriched["score"] = combined_score
        enriched["combined_score"] = combined_score
        final_results.append(enriched)

    _log_results("final", final_results, chunk_map)

    metrics = {
        "normalized_score": float(final_results[0]["score"] if final_results else 0.0),
        "bm25_count": len(keyword_results),
        "faiss_count": len(semantic_results),
        "merged_count": len(hybrid_results),
        "supporting_chunks": len(final_results),
    }

    _structured_log(
        "retrieval_selected_evidence",
        document_id=document_id,
        query=query,
        query_type=query_type,
        selected_chunks=[
            {
                "chunk_id": chunk["chunk_id"],
                "score": float(chunk.get("score", 0.0)),
                "preview": _preview(chunk.get("text", "")),
            }
            for chunk in final_results
        ],
    )

    return {
        "bm25_results": _format_debug_results(keyword_results, chunk_map),
        "semantic_results": _format_debug_results(semantic_results, chunk_map),
        "hybrid_results": _format_debug_results(hybrid_results, chunk_map),
        "final_results": _format_debug_results(final_results, chunk_map),
        "metrics": metrics,
        "chunk_map": chunk_map,
        "raw_final_results": final_results,
    }


def hybrid_search(
    document_id: int,
    query: str,
    bm25_k: int,
    faiss_k: int,
    db: Any,
    query_type: str = "unknown",
) -> Dict[str, Any]:
    retrieval = _prepare_retrieval(
        document_id=document_id,
        query=query,
        bm25_k=bm25_k,
        faiss_k=faiss_k,
        db=db,
        query_type=query_type,
    )
    return {"chunks": retrieval["raw_final_results"], "metrics": retrieval["metrics"]}


def debug_retrieval(
    document_id: int,
    query: str,
    db: Any,
    query_type: str = "unknown",
    bm25_k: int = 5,
    faiss_k: int = 5,
) -> Dict[str, Any]:
    retrieval = _prepare_retrieval(
        document_id=document_id,
        query=query,
        bm25_k=bm25_k,
        faiss_k=faiss_k,
        db=db,
        query_type=query_type,
    )

    return {
        "query": query,
        "query_type": query_type,
        "bm25_results": retrieval["bm25_results"],
        "semantic_results": retrieval["semantic_results"],
        "hybrid_results": retrieval["hybrid_results"],
        "final_results": retrieval["final_results"],
    }


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
    semantic_results: List[Dict[str, Any]],
    query: str = "",
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

    semantic_weight = _search_weights(query)[1]
    reranker = Reranker(alpha=semantic_weight)
    reranked = reranker.rerank(list(merged.values()))

    return reranked


def merge_results(
    keyword_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    query: str = "",
) -> List[Dict[str, Any]]:
    return _merge_results(keyword_results, semantic_results, query=query)
