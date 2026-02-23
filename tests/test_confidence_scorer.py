from app.confidence.scorer import score_confidence


def test_confidence_zero_if_not_verified():
    score = score_confidence(
        retrieval_metrics={"normalized_score": 1.0},
        answer_payload={"citations": ["c1"]},
        verification_result={"verified": False},
    )
    assert score == 0.0


def test_confidence_in_range():
    score = score_confidence(
        retrieval_metrics={"normalized_score": 0.7},
        answer_payload={"citations": ["c1"]},
        verification_result={"verified": True},
    )
    assert 0.0 <= score <= 1.0


def test_stronger_retrieval_means_higher_confidence():
    low = score_confidence(
        retrieval_metrics={"normalized_score": 0.2},
        answer_payload={"citations": ["c1"]},
        verification_result={"verified": True},
    )

    high = score_confidence(
        retrieval_metrics={"normalized_score": 0.9},
        answer_payload={"citations": ["c1"]},
        verification_result={"verified": True},
    )

    assert high > low


def test_no_citations_reduces_confidence():
    score = score_confidence(
        retrieval_metrics={"normalized_score": 0.9},
        answer_payload={"citations": []},
        verification_result={"verified": True},
    )
    assert score < 0.5