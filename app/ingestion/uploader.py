import hashlib
import shutil
from pathlib import Path
from fastapi import UploadFile

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def save_uploaded_file(file: UploadFile) -> tuple[str, str, str]:
    """
    Saves uploaded file to disk and computes SHA256 hash.
    Returns (filename, file_hash, file_path).
    """
    # Create a temporary path or read into memory to hash? 
    # For large files, reading chunk by chunk is better.
    # We will write to a temp location first or directly to destination if we knew the hash?
    # Since we need the hash to maybe name it or just to return it, we read it.
    
    # We'll save to a temp path first, compute hash, then move/rename if needed
    # or just save to the target filename. 
    # User requirement: "Save uploaded files under data/uploads/"
    # User requirement: "Compute SHA256 hash"
    
    file_path = UPLOAD_DIR / file.filename
    
    # We assume filename availability or overwrite.
    # Computing hash while writing.
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "wb") as buffer:
        while content := file.file.read(4096):
            buffer.write(content)
            sha256_hash.update(content)
            
    # Reset file cursor just in case, though we saved it already.
    file.file.seek(0)
    
    return file.filename, sha256_hash.hexdigest(), str(file_path)
