import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

STOPWORDS = {
    "the", "is", "are", "was", "were", "a", "an", "to", "of",
    "and", "or", "in", "on", "for", "with", "that", "this",
    "it", "as", "by", "at", "from"
}


def _tokenize(text: str) -> set:
    return {
        t for t in text.lower().split()
        if t.isalpha() and t not in STOPWORDS
    }


def verify_answer(
    answer_payload: Dict,
    evidence_chunks: List[Dict]
) -> Dict:
    """
    Verifies that the generated answer is supported by the provided evidence.

    Returns:
    {
        "verified": bool,
        "reason": str
    }
    """

    answer = answer_payload.get("answer")
    citations = answer_payload.get("citations", [])

    if not answer or answer == "INSUFFICIENT_EVIDENCE":
        return {
            "verified": False,
            "reason": "No answer to verify."
        }

    if not citations:
        return {
            "verified": False,
            "reason": "Answer has no citations."
        }

    evidence_map = {c["chunk_id"]: c["text"] for c in evidence_chunks}

    for cid in citations:
        if cid not in evidence_map:
            return {
                "verified": False,
                "reason": f"Citation '{cid}' does not exist in provided evidence."
            }

    answer_lower = answer.lower().strip()
    evidence_texts = [evidence_map[cid].lower() for cid in citations]
    exact_match = any(answer_lower and answer_lower in evidence_text for evidence_text in evidence_texts)
    logger.debug("Verification exact match: %s", exact_match)

    if exact_match:
        logger.debug("Verification overlap ratio: %s", 1.0)
        logger.debug("Verification result: %s", True)
        return {
            "verified": True,
            "reason": "Exact answer found in evidence"
        }

    answer_tokens = _tokenize(answer)

    if not answer_tokens:
        logger.debug("Verification overlap ratio: %s", 0.0)
        logger.debug("Verification result: %s", False)
        return {
            "verified": False,
            "reason": "Answer contains no verifiable content."
        }

    best_overlap_ratio = 0.0
    for cid in citations:
        evidence_tokens = _tokenize(evidence_map[cid])

        overlap = answer_tokens & evidence_tokens
        overlap_ratio = len(overlap) / len(answer_tokens)
        best_overlap_ratio = max(best_overlap_ratio, overlap_ratio)

        if overlap_ratio >= 0.20:
            logger.debug("Verification overlap ratio: %s", overlap_ratio)
            logger.debug("Verification result: %s", True)
            return {
                "verified": True,
                "reason": "Answer is sufficiently supported by cited evidence."
            }

    logger.debug("Verification overlap ratio: %s", best_overlap_ratio)
    logger.debug("Verification result: %s", False)
    return {
        "verified": False,
        "reason": "Answer is not sufficiently supported by cited evidence."
    }