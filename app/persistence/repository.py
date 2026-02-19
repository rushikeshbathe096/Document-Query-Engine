from sqlalchemy.orm import Session
from app.persistence.models import Document, Query, Answer

def create_document(db: Session, filename: str, file_hash: str) -> Document:
    db_document = Document(filename=filename, hash=file_hash)
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document

def get_document_by_hash(db: Session, file_hash: str) -> Document | None:
    return db.query(Document).filter(Document.hash == file_hash).first()

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
