from app.reasoning.query_analyzer import QueryAnalyzer

analyzer = QueryAnalyzer()

queries = [
    "What is HPBW?",
    "Compare HPBW and FNBW",
    "How do I calculate antenna gain?",
    "asdfghjkl"
]

for q in queries:
    print("QUERY:", q)
    print(analyzer.analyze_query(q))
    print("-" * 40)