from app.ingestion.parser import parse_document

pdf_path = "data/uploads/RS_Exp1_Rushikesh Bathe.pdf"

pages = parse_document(pdf_path)

print(type(pages))
print(len(pages))
print(pages[0].keys())
print(pages[0]["page_number"])
print(pages[0]["text"][:500])