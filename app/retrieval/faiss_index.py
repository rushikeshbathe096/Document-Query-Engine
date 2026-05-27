import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any

INDEX_DIR = Path("data/indexes")

class FAISSIndex:
    """
    Manages a FAISS IndexFlatL2 for storing and persisting chunk embeddings.
    Ensures strict alignment between vector order and chunk identifiers.
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunk_ids: List[str] = []

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Adds embeddings to the FAISS index and records chunk identifiers.
        Assumes embeddings are in the same order as the chunks.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks and {len(embeddings)} embeddings."
            )
        
        # FAISS expects float32
        vectors = embeddings.astype("float32")
        self.index.add(vectors)
        
        # Maintain order alignment
        for chunk in chunks:
            self.chunk_ids.append(chunk["chunk_id"])

    def save(self, name: str = "semantic_index"):
        """
        Persists the index and the mapping to data/indexes/.
        """
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        
        index_path = INDEX_DIR / f"{name}.faiss"
        mapping_path = INDEX_DIR / f"{name}.json"
        
        faiss.write_index(self.index, str(index_path))
        with open(mapping_path, "w") as f:
            json.dump(self.chunk_ids, f)

    @classmethod
    def load(cls, name: str = "semantic_index") -> "FAISSIndex":
        """
        Loads the index and mapping from disk.
        """
        index_path = INDEX_DIR / f"{name}.faiss"
        mapping_path = INDEX_DIR / f"{name}.json"
        
        if not index_path.exists() or not mapping_path.exists():
            raise FileNotFoundError(f"Could not find index files in {INDEX_DIR}")
            
        index = faiss.read_index(str(index_path))
        with open(mapping_path, "r") as f:
            chunk_ids = json.load(f)
            
        instance = cls(dimension=index.d)
        instance.index = index
        instance.chunk_ids = chunk_ids
        return instance

    @classmethod
    def delete(cls, name: str = "semantic_index") -> None:
        index_path = INDEX_DIR / f"{name}.faiss"
        mapping_path = INDEX_DIR / f"{name}.json"

        if index_path.exists():
            index_path.unlink()
        if mapping_path.exists():
            mapping_path.unlink()

    @staticmethod
    def index_name_for_document(document_id: int) -> str:
        return f"doc_{document_id}"

    def rebuild(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Clears the current index and rebuilds it from the provided chunks.
        """
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunk_ids = []
        self.add_chunks(chunks, embeddings)
