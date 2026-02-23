import json
from typing import List, Dict
from app.reasoning.prompt_builder import build_answer_prompt
from app.llm.gemini import generate_with_gemini   # ← USE YOUR FUNCTION

def extract_json(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        lines = [line for line in lines if not line.startswith("```")]
        text = "\n".join(lines)

    return text.strip()

def generate_answer(
    query: str,
    query_analysis: Dict,
    evidence_chunks: List[Dict]
) -> Dict:

    if not evidence_chunks:
        return {
            "answer": "INSUFFICIENT_EVIDENCE",
            "citations": [],
            "confidence": "low"
        }

    prompt = build_answer_prompt(query, query_analysis, evidence_chunks)

    raw_output = generate_with_gemini(prompt)

    print("\n===== RAW GEMINI OUTPUT =====")
    print(raw_output)
    print("===== END RAW OUTPUT =====\n")

    try:
        cleaned = extract_json(raw_output)
        parsed = json.loads(cleaned)
    except Exception:
        return {
            "answer": "INSUFFICIENT_EVIDENCE",
            "citations": [],
            "confidence": "low"
        }

    if set(parsed.keys()) != {"answer", "citations", "confidence"}:
        return {
            "answer": "INSUFFICIENT_EVIDENCE",
            "citations": [],
            "confidence": "low"
        }

    if parsed["answer"] == "INSUFFICIENT_EVIDENCE":
        return parsed

    valid_chunk_ids = {c["chunk_id"] for c in evidence_chunks}
    if not set(parsed["citations"]).issubset(valid_chunk_ids):
        return {
            "answer": "INSUFFICIENT_EVIDENCE",
            "citations": [],
            "confidence": "low"
        }

    if not parsed["answer"]:
        return {
            "answer": "INSUFFICIENT_EVIDENCE",
            "citations": [],
            "confidence": "low"
        }

    return parsed