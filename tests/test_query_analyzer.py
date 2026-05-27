from app.reasoning.query_analyzer import QueryAnalyzer


def test_query_analyzer_extracts_json_from_prefixed_response(monkeypatch):
    monkeypatch.setattr(
        "app.reasoning.query_analyzer.generate_with_groq",
        lambda prompt: 'Sure!\n\n```json\n{"query_type": "lookup", "keywords": ["name"], "expected_sections": ["header"]}\n```',
    )

    analyzer = QueryAnalyzer()
    result = analyzer.analyze_query("What is the candidate name?")

    assert result["query_type"] == "lookup"
    assert result["keywords"] == ["name"]
    assert result["expected_sections"] == ["header"]


def test_query_analyzer_falls_back_on_malformed_response(monkeypatch):
    monkeypatch.setattr(
        "app.reasoning.query_analyzer.generate_with_groq",
        lambda prompt: "this is not valid json",
    )

    analyzer = QueryAnalyzer()
    result = analyzer.analyze_query("What is the candidate name?")

    assert result["query_type"] == "unknown"
    assert result["keywords"] == []
    assert result["expected_sections"] == []