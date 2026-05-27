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

    retrieval_score = max(0.0, min(1.0, retrieval_metrics.get("normalized_score", 0.0)))
    supporting_chunks = retrieval_metrics.get("supporting_chunks", len(answer_payload.get("citations", [])))
    supporting_chunks_score = min(1.0, supporting_chunks / 3.0)

    citations = answer_payload.get("citations", [])
    answer = answer_payload.get("answer", "")

    verification_score = 1.0 if verification_result.get("verified", False) else 0.0
    confidence = (
        0.55 * retrieval_score
        + 0.25 * supporting_chunks_score
        + 0.20 * verification_score
    )

    answer_word_count = len(answer.split())
    if answer_word_count <= 4:
        confidence *= 0.85

    return round(max(0.0, min(1.0, confidence)), 3)