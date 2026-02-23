# Manual verification script for Phase 13
# NOT an automated test

from app.confidence.scorer import score_confidence


cases = [
    {
        "name": "Verified + strong retrieval",
        "retrieval": {"normalized_score": 0.9},
        "answer": {"citations": ["c1"]},
        "verification": {"verified": True},
    },
    {
        "name": "Verified + weak retrieval",
        "retrieval": {"normalized_score": 0.2},
        "answer": {"citations": ["c1"]},
        "verification": {"verified": True},
    },
    {
        "name": "Unverified answer",
        "retrieval": {"normalized_score": 0.9},
        "answer": {"citations": ["c1"]},
        "verification": {"verified": False},
    },
    {
        "name": "Verified but no citations",
        "retrieval": {"normalized_score": 0.8},
        "answer": {"citations": []},
        "verification": {"verified": True},
    },
]

for case in cases:
    score = score_confidence(
        retrieval_metrics=case["retrieval"],
        answer_payload=case["answer"],
        verification_result=case["verification"],
    )
    print(case["name"], "→", score)