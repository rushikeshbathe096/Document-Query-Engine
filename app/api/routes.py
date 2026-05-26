from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
import logging

from app.persistence.database import SessionLocal
from app.persistence import repository

from app.ingestion.uploader import save_uploaded_file
from app.ingestion.parser import parse_document
from app.ingestion.chunker import get_chunker

from app.retrieval.embeddings import Embedder
from app.retrieval.faiss_index import FAISSIndex
from app.retrieval.hybrid_search import hybrid_search

from app.reasoning.query_analyzer import QueryAnalyzer
from app.reasoning.router import route_query
from app.reasoning.answer_generator import generate_answer
from app.reasoning.verification import verify_answer
from app.confidence.scorer import score_confidence

logger = logging.getLogger(__name__)

router = APIRouter()
query_analyzer = QueryAnalyzer()
chunker = get_chunker()

# DB Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Upload Endpoint — pre-processes document
@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    filename, file_hash, file_path = save_uploaded_file(file)

    existing_doc = repository.get_document_by_hash(db, file_hash)
    if existing_doc:
        return {"document_id": existing_doc.id, "status": "exists"}

    new_doc = repository.create_document(db, filename, file_hash)

    try:
        pages = parse_document(file_path)
        if not pages:
            raise ValueError("No pages extracted from document")

        chunks = chunker.chunk_document(pages)
        if not chunks:
            raise ValueError("No chunks generated from document")

        embedder = Embedder()
        index_name = FAISSIndex.index_name_for_document(new_doc.id)
        embeddings = embedder.get_embeddings(chunks)
        faiss_idx = FAISSIndex(dimension=embedder.get_dimension())
        faiss_idx.add_chunks(chunks, embeddings)
        faiss_idx.save(index_name)

        repository.create_chunks(db, new_doc.id, chunks)
    except Exception as e:
        logger.error("Pre-processing failed for document %d: %s", new_doc.id, e)
        from pathlib import Path
        file_path_obj = Path(file_path)
        if file_path_obj.exists():
            file_path_obj.unlink()
        repository.delete_chunks_by_document(db, new_doc.id)
        db.delete(new_doc)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Document processing failed: {e}")

    return {"document_id": new_doc.id, "status": "created"}


# Query Models
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[str]


# Query Endpoint
@router.post("/documents/{document_id}/query", response_model=QueryResponse)
def query_document(
    document_id: int,
    payload: QueryRequest,
    db: Session = Depends(get_db)
):
    document = repository.get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    query_row = repository.create_query(db, document_id, payload.query)

    query_analysis = query_analyzer.analyze_query(payload.query)
    route = route_query(query_analysis)

    retrieval = hybrid_search(
        document_id=document_id,
        query=payload.query,
        bm25_k=route["bm25_k"],
        faiss_k=route["faiss_k"],
        db=db,
    )

    evidence_chunks = retrieval.get("chunks", [])
    retrieval_metrics = retrieval.get("metrics", {})

    if not evidence_chunks:
        repository.create_answer(db, query_row.id, "Not Found", 0.0)
        return {"answer": "Not Found", "confidence": 0.0, "sources": []}

    answer_payload = generate_answer(
        query=payload.query,
        query_analysis=query_analysis,
        evidence_chunks=evidence_chunks
    )

    if answer_payload["answer"] == "INSUFFICIENT_EVIDENCE":
        repository.create_answer(db, query_row.id, "Not Found", 0.0)
        return {"answer": "Not Found", "confidence": 0.0, "sources": []}

    verification = verify_answer(
        answer_payload=answer_payload,
        evidence_chunks=evidence_chunks
    )

    if not verification["verified"]:
        repository.create_answer(db, query_row.id, "Not Found", 0.0)
        return {"answer": "Not Found", "confidence": 0.0, "sources": []}

    confidence = score_confidence(
        retrieval_metrics=retrieval_metrics,
        answer_payload=answer_payload,
        verification_result=verification
    )

    response = {
        "answer": answer_payload["answer"],
        "confidence": confidence,
        "sources": answer_payload["citations"]
    }

    repository.create_answer(db, query_row.id, response["answer"], response["confidence"])

    return response
