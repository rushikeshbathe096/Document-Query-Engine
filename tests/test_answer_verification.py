from app.reasoning.verification import verify_answer


def test_verification_passes_with_valid_evidence():
    answer = {
        "answer": "Experiment 1 is designed to study the basic operation of the RS Lab setup.",
        "citations": ["c1"],
    }

    evidence = [
        {
            "chunk_id": "c1",
            "text": "The objective of Experiment 1 is to study the basic operation of the RS Lab setup."
        }
    ]

    result = verify_answer(answer, evidence)

    assert result["verified"] is True


def test_verification_passes_on_exact_match_short_answer():
    answer = {
        "answer": "Rushikesh Bathe",
        "citations": ["c1"],
    }

    evidence = [
        {
            "chunk_id": "c1",
            "text": "Candidate Name: Rushikesh Bathe"
        }
    ]

    result = verify_answer(answer, evidence)

    assert result["verified"] is True
    assert result["reason"] == "Exact answer found in evidence"


def test_verification_fails_without_citations():
    answer = {
        "answer": "Some answer",
        "citations": [],
    }

    evidence = [
        {"chunk_id": "c1", "text": "Some text"}
    ]

    result = verify_answer(answer, evidence)

    assert result["verified"] is False


def test_verification_fails_on_invalid_citation():
    answer = {
        "answer": "Some answer",
        "citations": ["fake_id"],
    }

    evidence = [
        {"chunk_id": "c1", "text": "Some text"}
    ]

    result = verify_answer(answer, evidence)

    assert result["verified"] is False


def test_verification_fails_when_answer_not_supported():
    answer = {
        "answer": "Graph database optimization.",
        "citations": ["c1"],
    }

    evidence = [
        {
            "chunk_id": "c1",
            "text": "The objective of Experiment 1 is to study the basic operation of the RS Lab setup."
        }
    ]

    result = verify_answer(answer, evidence)

    assert result["verified"] is False