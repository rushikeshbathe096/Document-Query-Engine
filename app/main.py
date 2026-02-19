from fastapi import FastAPI

app = FastAPI(
    title="Document Intelligence API",
    description="Backend-only Document Intelligence system using Hybrid Retrieval (FAISS + BM25) and LLM.",
    version="1.0.0",
)

@app.get("/health")
async def health_check():
    return {"status": "ok"}
from app.core import config
