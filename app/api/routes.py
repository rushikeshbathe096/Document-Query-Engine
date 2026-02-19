from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session
from app.persistence.database import SessionLocal
from app.persistence import repository
from app.ingestion.uploader import save_uploaded_file

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 1. Save file and get hash
    filename, file_hash, file_path = save_uploaded_file(file)
    
    # 2. Check if document exists
    existing_doc = repository.get_document_by_hash(db, file_hash)
    if existing_doc:
        return {"document_id": existing_doc.id, "status": "exists"}
        
    # 3. Create new document
    new_doc = repository.create_document(db, filename, file_hash)
    
    return {"document_id": new_doc.id, "status": "created"}
