from typing import List, Dict, Any

class Reranker:
    """
    Combines evidence from multiple sources using weighted normalization and thresholds.
    """
    def __init__(
        self, 
        alpha: float = 0.5, 
        threshold: float = 0.1,
        top_k: int = 5
    ):
        """
        Args:
            alpha: Weight for semantic score (0.0 to 1.0). Keyword weight is (1 - alpha).
            threshold: Minimum combined score to keep a chunk.
            top_k: Number of results to return.
        """
        self.alpha = alpha
        self.threshold = threshold
        self.top_k = top_k

    def rerank(self, combined_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies weights and thresholds to return the final ranked evidence.
        Each entry in combined_results should have:
        - chunk_id
        - norm_keyword_score (float, 0-1)
        - norm_semantic_score (float, 0-1)
        """
        final_scores = []
        
        for res in combined_results:
            k_score = res.get("norm_keyword_score", 0.0)
            s_score = res.get("norm_semantic_score", 0.0)
            
            # Weighted combination
            combined_score = (self.alpha * s_score) + ((1 - self.alpha) * k_score)
            
            if combined_score >= self.threshold:
                final_scores.append({
                    "chunk_id": res["chunk_id"],
                    "score": combined_score,
                    # Provenance for debugging/transparency
                    "metadata": {
                        "keyword_norm": k_score,
                        "semantic_norm": s_score
                    }
                })
        
        # Sort by combined score descending
        final_scores.sort(key=lambda x: x["score"], reverse=True)
        
        return final_scores[:self.top_k]
