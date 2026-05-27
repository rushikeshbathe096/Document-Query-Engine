import fitz
from pathlib import Path
from typing import List, Dict, Any


class UnsupportedDocumentTypeError(ValueError):
    pass

def parse_document(file_path: str) -> List[Dict[str, Any]]:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [{"page_number": 1, "text": text}]

    if suffix != ".pdf":
        raise UnsupportedDocumentTypeError(f"Unsupported file type: {suffix}")

    doc = fitz.open(file_path)
    pages = []

    try:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            text = page.get_text()
            pages.append({
                "page_number": page_index + 1,
                "text": text
            })
    finally:
        doc.close()

    return pages
