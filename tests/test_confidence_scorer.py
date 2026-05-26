from app.confidence.scorer import score_confidence


def test_returns_zero_when_not_verified():
    result = score_confidence(
        retrieval_metrics={"normalized_score": 0.8},
        answer_payload={"answer": "test", "citations": ["c1"]},
        verification_result={"verified": False}
    )
    assert result == 0.0


def test_returns_positive_when_verified():
    result = score_confidence(
        retrieval_metrics={"normalized_score": 1.0},
        answer_payload={"answer": "test answer", "citations": ["c1", "c2"]},
        verification_result={"verified": True}
    )
    assert result > 0.0
    assert result <= 1.0


def test_lower_retrieval_score_lowers_confidence():
    high = score_confidence(
        retrieval_metrics={"normalized_score": 1.0},
        answer_payload={"answer": "test answer", "citations": ["c1"]},
        verification_result={"verified": True}
    )
    low = score_confidence(
        retrieval_metrics={"normalized_score": 0.2},
        answer_payload={"answer": "test answer", "citations": ["c1"]},
        verification_result={"verified": True}
    )
    assert high > low


def test_clamps_to_zero_one_range():
    result = score_confidence(
        retrieval_metrics={"normalized_score": 5.0},
        answer_payload={"answer": "test", "citations": ["c1"]},
        verification_result={"verified": True}
    )
    assert result <= 1.0
