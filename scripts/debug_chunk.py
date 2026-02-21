from app.ingestion.parser import parse_document
from app.ingestion.chunker import get_chunker

pdf_path = "data/uploads/RS_Exp1_Rushikesh Bathe.pdf"

pages = parse_document(pdf_path)
chunker = get_chunker()

chunks = []
for page in pages:
    chunks.extend(
        chunker.chunk_page(
            page_number=page["page_number"],
            text=page["text"]
        )
    )

print("Total chunks:", len(chunks))
print("=" * 50)

for c in chunks[:5]:
    print("PAGE:", c["page_number"])
    print("POS:", c["position"])
    print("TEXT:", c["text"][:300])
    print("-" * 50)