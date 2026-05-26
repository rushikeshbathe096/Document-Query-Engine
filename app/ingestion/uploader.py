import hashlib
import uuid
from pathlib import Path
from fastapi import UploadFile

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".docx", ".txt"}

def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"File type '{suffix}' not allowed. Allowed: {ALLOWED_EXTENSIONS}")
    return name

def save_uploaded_file(file: UploadFile) -> tuple[str, str, str]:
    safe_name = _safe_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    file_path = UPLOAD_DIR / unique_name

    sha256_hash = hashlib.sha256()
    with open(file_path, "wb") as buffer:
        while content := file.file.read(4096):
            buffer.write(content)
            sha256_hash.update(content)

    return safe_name, sha256_hash.hexdigest(), str(file_path)
