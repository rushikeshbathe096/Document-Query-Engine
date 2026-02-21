import fitz
from typing import List, Dict, Any

def parse_document(file_path: str) -> List[Dict[str, Any]]:
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