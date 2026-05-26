from typing import Dict


def route_query(query_analysis: Dict) -> Dict:
    query_type = query_analysis.get("query_type", "unknown")

    if query_type == "definition":
        return {"strategy": "hybrid", "bm25_k": 10, "faiss_k": 10}

    if query_type == "procedural":
        return {"strategy": "hybrid", "bm25_k": 8, "faiss_k": 8}

    if query_type == "conceptual":
        return {"strategy": "faiss", "bm25_k": 5, "faiss_k": 10}

    if query_type == "comparison":
        return {"strategy": "hybrid", "bm25_k": 8, "faiss_k": 8}

    if query_type == "lookup":
        return {"strategy": "bm25", "bm25_k": 12, "faiss_k": 0}

    if query_type == "factual":
        return {"strategy": "bm25", "bm25_k": 10, "faiss_k": 5}

    if query_type == "summarization":
        return {"strategy": "faiss", "bm25_k": 5, "faiss_k": 10}

    return {"strategy": "hybrid", "bm25_k": 5, "faiss_k": 5}
