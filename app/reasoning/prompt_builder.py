from typing import List, Dict


def build_answer_prompt(
    query: str,
    query_analysis: Dict,
    evidence_chunks: List[Dict]
) -> str:
    evidence_block = "\n\n".join(
        f"[{chunk['chunk_id']}]\n{chunk['text']}"
        for chunk in evidence_chunks
    )

    return f"""
You are an evidence-bound answer generator.

Rules you MUST follow:
- You may ONLY use information explicitly stated in the evidence chunks.
- You MAY paraphrase or summarize evidence if meaning is preserved.
- If the evidence clearly answers the question, you SHOULD answer.
- Every factual statement MUST be supported by at least one chunk.
- You MUST cite chunk_ids explicitly.
- If the evidence does NOT clearly support an answer, you MUST refuse.
- Refusal is correct behavior.
- Do NOT guess.
- Do NOT use external knowledge.

User Query:
{query}

Query Analysis:
{query_analysis}

Evidence Chunks:
{evidence_block}

Output MUST be valid JSON in exactly this format:

{{
  "answer": "string",
  "citations": ["chunk_id"],
  "confidence": "high | medium | low"
}}

If evidence is insufficient, output EXACTLY:

{{
  "answer": "INSUFFICIENT_EVIDENCE",
  "citations": [],
  "confidence": "low"
}}
""".strip()