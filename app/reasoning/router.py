from typing import Dict


def route_query(query_analysis: Dict) -> Dict:
    query_type = query_analysis.get("query_type", "unknown")

    if query_type == "definition":
        return {
            "strategy": "hybrid",
            "bm25_weight": 0.7,
            "faiss_weight": 0.3,
            "bm25_k": 10,
            "faiss_k": 5,
        }

    if query_type == "procedural":
        return {
            "strategy": "hybrid",
            "bm25_weight": 0.4,
            "faiss_weight": 0.6,
            "bm25_k": 8,
            "faiss_k": 8,
        }

    if query_type == "conceptual":
        return {
            "strategy": "faiss",
            "bm25_weight": 0.2,
            "faiss_weight": 0.8,
            "bm25_k": 5,
            "faiss_k": 10,
        }

    if query_type == "comparison":
        return {
            "strategy": "hybrid",
            "bm25_weight": 0.5,
            "faiss_weight": 0.5,
            "bm25_k": 8,
            "faiss_k": 8,
        }

    if query_type == "lookup":
        return {
            "strategy": "bm25",
            "bm25_weight": 1.0,
            "faiss_weight": 0.0,
            "bm25_k": 12,
            "faiss_k": 0,
        }

    return {
        "strategy": "hybrid",
        "bm25_weight": 0.5,
        "faiss_weight": 0.5,
        "bm25_k": 5,
        "faiss_k": 5,
    }