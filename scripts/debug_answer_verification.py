from app.reasoning.verification import verify_answer

answer_payload = {
    "answer": "Experiment 1 is designed to study the basic operation of the RS Lab setup.",
    "citations": ["doc1_p1_c0"],
    "confidence": "high",
}

evidence_chunks = [
    {
        "chunk_id": "doc1_p1_c0",
        "text": "The objective of Experiment 1 is to study the basic operation of the RS Lab setup."
    }
]

result = verify_answer(answer_payload, evidence_chunks)
print(result)