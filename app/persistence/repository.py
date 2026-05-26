from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.persistence.models import Document, Chunk, Query, Answer

def create_document(db: Session, filename: str, file_hash: str) -> Document:
    db_document = Document(filename=filename, hash=file_hash)
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document

def get_document_by_hash(db: Session, file_hash: str) -> Document | None:
    return db.query(Document).filter(Document.hash == file_hash).first()

def get_document_by_id(db: Session, document_id: int) -> Document | None:
    return db.query(Document).filter(Document.id == document_id).first()

def create_chunks(db: Session, document_id: int, chunks: List[Dict[str, Any]]):
    for c in chunks:
        db.add(Chunk(
            document_id=document_id,
            chunk_id=c["chunk_id"],
            text=c["text"],
            page_number=c["page_number"],
            position=c["position"]
        ))
    db.commit()

def get_chunks_by_document(db: Session, document_id: int) -> List[Dict[str, Any]]:
    rows = db.query(Chunk).filter(Chunk.document_id == document_id).order_by(Chunk.position).all()
    return [
        {"chunk_id": r.chunk_id, "text": r.text, "page_number": r.page_number, "position": r.position}
        for r in rows
    ]

def delete_chunks_by_document(db: Session, document_id: int):
    db.query(Chunk).filter(Chunk.document_id == document_id).delete()
    db.commit()

def create_query(db: Session, document_id: int, question: str) -> Query:
    db_query = Query(document_id=document_id, question=question)
    db.add(db_query)
    db.commit()
    db.refresh(db_query)
    return db_query

def create_answer(db: Session, query_id: int, answer_text: str, confidence: float) -> Answer:
    db_answer = Answer(query_id=query_id, answer_text=answer_text, confidence=confidence)
    db.add(db_answer)
    db.commit()
    db.refresh(db_answer)
    return db_answer
