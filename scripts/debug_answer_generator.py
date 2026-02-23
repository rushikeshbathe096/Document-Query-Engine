from app.reasoning.answer_generator import generate_answer

query = "What is the objective of Experiment 1?"

query_analysis = {
    "query_type": "definition",
    "keywords": ["objective", "experiment 1"],
    "expected_sections": ["objective"]
}

evidence_chunks = [
    {
        "chunk_id": "doc1_p1_c0",
        "text": "Experiment 1 aims to study the basic operation of the RS Lab setup."
    }
]

result = generate_answer(query, query_analysis, evidence_chunks)
print(result)