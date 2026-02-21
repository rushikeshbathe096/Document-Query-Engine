from app.retrieval.embeddings import Embedder
from app.retrieval.faiss_index import FAISSIndex

chunks = [
    {
        "chunk_id": "doc1_p1_0",
        "document_id": 1,
        "page_number": 1,
        "chunk_index": 0,
        "text": "Radiation intensity represents power per unit solid angle."
    },
    {
        "chunk_id": "doc1_p2_1",
        "document_id": 1,
        "page_number": 2,
        "chunk_index": 1,
        "text": "Half Power Beam Width is the angular width at half maximum power."
    },
]

# Embeddings
embedder = Embedder()
embeddings = embedder.get_embeddings(chunks)

print("Embeddings shape:", embeddings.shape)
print("Embedding dimension:", embedder.get_dimension())

# FAISS index
index = FAISSIndex(dimension=embedder.get_dimension())
index.add_chunks(chunks, embeddings)

print("FAISS index size:", index.index.ntotal)

# Persist
index.save("semantic_index")
print("Index saved")

# Load
loaded_index = FAISSIndex.load("semantic_index")
print("Loaded index size:", loaded_index.index.ntotal)
print("Loaded chunk ids:", loaded_index.chunk_ids)