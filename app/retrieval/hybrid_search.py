from typing import List, Dict, Any
from app.retrieval.reranker import Reranker

class HybridSearch:
    """
    Orchestrates the merging and normalization of keyword (BM25) and semantic (FAISS) results.
    """
    def __init__(self, reranker: Reranker = None):
        self.reranker = reranker or Reranker()

    def _normalize_scores(self, results: List[Dict[str, Any]], invert: bool = False) -> List[Dict[str, Any]]:
        """
        Applies Min-Max normalization to scores.
        If invert is True (for distances), smaller original values get higher normalized scores.
        """
        if not results:
            return []
        
        scores = [r["score"] for r in results]
        min_s = min(scores)
        max_s = max(scores)
        denom = max_s - min_s
        
        normalized = []
        for r in results:
            s = r["score"]
            if denom == 0:
                # If all scores are equal, assign 1.0
                norm_s = 1.0
            else:
                norm_s = (s - min_s) / denom
            
            if invert:
                norm_s = 1.0 - norm_s
                
            normalized.append({
                "chunk_id": r["chunk_id"],
                "norm_score": norm_s
            })
        return normalized

    def merge_results(
        self, 
        keyword_results: List[Dict[str, Any]], 
        semantic_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Normalizes and merges results from both sources by chunk_id.
        - keyword_results: List of {'chunk_id': str, 'score': float} (BM25 scores)
        - semantic_results: List of {'chunk_id': str, 'score': float} (FAISS distances)
        """
        # 1. Normalize both streams
        norm_keyword = self._normalize_scores(keyword_results, invert=False)
        norm_semantic = self._normalize_scores(semantic_results, invert=True)

        # 2. Merge by chunk_id
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

        # 3. Pass to reranker
        return self.reranker.rerank(list(merged.values()))
