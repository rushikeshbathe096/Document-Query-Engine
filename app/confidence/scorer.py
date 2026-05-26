from typing import Dict, List


def score_confidence(
    retrieval_metrics: Dict,
    answer_payload: Dict,
    verification_result: Dict
) -> float:
    """
    Compute a confidence score in range [0, 1] based on:
    - retrieval strength
    - evidence agreement
    - verification result
    """

    # Hard failure: verification did not pass
    if not verification_result.get("verified", False):
        return 0.0

    # Retrieval strength (expected 0..1, default conservative)
    retrieval_score = retrieval_metrics.get("normalized_score", 0.0)
    retrieval_score = max(0.0, min(1.0, retrieval_score))

    # Evidence agreement: how well-cited the answer is
    citations = answer_payload.get("citations", [])
    answer = answer_payload.get("answer", "")

    if not citations or not answer:
        evidence_agreement = 0.0
    else:
        answer_word_count = len(answer.split())
        # More citations per word of answer → higher confidence
        citation_density = len(citations) / max(answer_word_count, 1)
        evidence_agreement = min(1.0, citation_density * 5)

    confidence = 0.5 * retrieval_score + 0.5 * evidence_agreement

    return round(max(0.0, min(1.0, confidence)), 3)