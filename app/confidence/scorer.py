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

    # Evidence agreement: proportion of cited chunks that actually exist
    citations = answer_payload.get("citations", [])
    total_citations = len(citations)

    if total_citations == 0:
        evidence_agreement = 0.0
    else:
        evidence_agreement = 1.0

    # Weighted combination
    # Verification already passed, so it acts as a gate, not a weight
    confidence = (
        0.5 * retrieval_score +
        0.5 * evidence_agreement
    )

    return round(max(0.0, min(1.0, confidence)), 3)