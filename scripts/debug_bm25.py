from app.retrieval.bm25_index import BM25Index

chunks = [
    {
        "chunk_id": "c1",
        "text": "Half Power Beam Width is the angular width of the main lobe"
    },
    {
        "chunk_id": "c2",
        "text": "Radiation intensity represents power per unit solid angle"
    },
    {
        "chunk_id": "c3",
        "text": "This chunk talks about something completely unrelated"
    }
]

bm25 = BM25Index()
bm25.index_chunks(chunks)

results = bm25.search("What is HPBW?", top_k=3)

for r in results:
    print(r)