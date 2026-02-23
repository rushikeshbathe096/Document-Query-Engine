from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List

from app.persistence.database import SessionLocal
from app.persistence import repository
from app.persistence.models import Document, Query, Answer

from app.ingestion.uploader import save_uploaded_file

from app.reasoning.query_analyzer import QueryAnalyzer
from app.reasoning.router import route_query
from app.retrieval.hybrid_search import hybrid_search
from app.reasoning.answer_generator import generate_answer
from app.reasoning.verification import verify_answer
from app.confidence.scorer import score_confidence


router = APIRouter()
query_analyzer = QueryAnalyzer()

# ---------- DB Dependency ----------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------- Upload Endpoint ----------
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
    return {"document_id": new_doc.id, "status": "created"}


# ---------- Query Models ----------
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[str]


# ---------- Phase 14: Query Endpoint ----------
@router.post(
    "/documents/{document_id}/query",
    response_model=QueryResponse
)
def query_document(
    document_id: int,
    payload: QueryRequest,
    db: Session = Depends(get_db)
):
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # 1. Log query
    query_row = Query(
        document_id=document_id,
        query_text=payload.query
    )
    db.add(query_row)
    db.commit()
    db.refresh(query_row)

    # 2. Query analysis
    query_analysis = analyze_query(payload.query)

    # 3. Routing
    route = route_query(query_analysis)

    # 4. Retrieval
    retrieval = hybrid_search(
        document_id=document_id,
        query=payload.query,
        bm25_k=route["bm25_k"],
        faiss_k=route["faiss_k"],
        bm25_weight=route["bm25_weight"],
        faiss_weight=route["faiss_weight"],
    )

    evidence_chunks = retrieval.get("chunks", [])
    retrieval_metrics = retrieval.get("metrics", {})

    # Short-circuit: no evidence
    if not evidence_chunks:
        response = {"answer": "Not Found", "confidence": 0.0, "sources": []}
        db.add(Answer(
            query_id=query_row.id,
            answer_text=response["answer"],
            confidence=response["confidence"]
        ))
        db.commit()
        return response

    # 5. Answer generation
    answer_payload = generate_answer(
        query=payload.query,
        query_analysis=query_analysis,
        evidence_chunks=evidence_chunks
    )

    if answer_payload["answer"] == "INSUFFICIENT_EVIDENCE":
        response = {"answer": "Not Found", "confidence": 0.0, "sources": []}
        db.add(Answer(
            query_id=query_row.id,
            answer_text=response["answer"],
            confidence=response["confidence"]
        ))
        db.commit()
        return response

    # 6. Verification
    verification = verify_answer(
        answer_payload=answer_payload,
        evidence_chunks=evidence_chunks
    )

    if not verification["verified"]:
        response = {"answer": "Not Found", "confidence": 0.0, "sources": []}
        db.add(Answer(
            query_id=query_row.id,
            answer_text=response["answer"],
            confidence=response["confidence"]
        ))
        db.commit()
        return response

    # 7. Confidence scoring
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

    # 8. Persist final answer
    db.add(Answer(
        query_id=query_row.id,
        answer_text=response["answer"],
        confidence=response["confidence"]
    ))
    db.commit()

    return response