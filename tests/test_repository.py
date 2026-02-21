from app.persistence.database import SessionLocal
from app.persistence.repository import (
    create_document,
    get_document_by_hash,
    create_query,
    create_answer,
)


def test_repository_crud_flow():
    db = SessionLocal()
    try:
        doc = create_document(
            db=db,
            filename="test.pdf",
            file_hash="abc123"
        )

        assert doc.id is not None

        fetched = get_document_by_hash(db, "abc123")
        assert fetched is not None
        assert fetched.id == doc.id

        query = create_query(
            db=db,
            document_id=doc.id,
            question="What is this document about?"
        )

        answer = create_answer(
            db=db,
            query_id=query.id,
            answer_text="Test answer",
            confidence=0.87
        )

        assert answer.id is not None
        assert answer.confidence == 0.87

    finally:
        db.close()