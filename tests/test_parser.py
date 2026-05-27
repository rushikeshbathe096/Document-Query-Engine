import fitz
from app.ingestion.parser import parse_document


def test_parse_document_reads_pdf(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello from test PDF")
    doc.save(pdf_path)
    doc.close()

    pages = parse_document(str(pdf_path))

    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert "Hello from test PDF" in pages[0]["text"]
