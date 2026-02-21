import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

class BM25Index:
    """
    In-memory keyword search index using BM25.
    Operates on existing chunks and provides deterministic tokenization.
    """
    def __init__(self):
        self.index = None
        self.chunk_ids = []

    def _tokenize(self, text: str) -> List[str]:
        """
        Word-level tokenization: lowercase and alphanumeric split.
        """
        if not text:
            return []
        return re.findall(r"\w+", text.lower())

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Builds the index from a list of chunks. 
        Maintains alignment between chunk_ids and the BM25 internal corpus.
        """
        if not chunks:
            self.index = None
            self.chunk_ids = []
            return

        self.chunk_ids = [c["chunk_id"] for c in chunks]
        tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]
        self.index = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs keyword search using BM25 selection.
        Returns a list of structured results sorted by score.
        """
        if self.index is None or not query:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.index.get_scores(tokenized_query)

        # Map scores back to chunk identifiers
        results = []
        for i in range(len(self.chunk_ids)):
            results.append({
                "chunk_id": self.chunk_ids[i],
                "score": float(scores[i])
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]
