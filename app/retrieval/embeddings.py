from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np

class Embedder:
    """
    Handles generation of semantic embeddings using SentenceTransformers.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generates embeddings for a list of chunk objects.
        Returns a numpy array of embeddings.
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.array(embeddings).astype("float32")

    def get_dimension(self) -> int:
        """
        Returns the embedding dimension for the current model.
        """
        return self.model.get_sentence_embedding_dimension()
