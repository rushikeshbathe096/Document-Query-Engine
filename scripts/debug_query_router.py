# Manual verification script for Phase 10
# NOT an automated test

from app.reasoning.router import route_query

cases = [
    {"query_type": "definition"},
    {"query_type": "procedural"},
    {"query_type": "conceptual"},
    {"query_type": "comparison"},
    {"query_type": "lookup"},
    {"query_type": "unknown"},
]

for case in cases:
    print("\nQUERY TYPE:", case["query_type"])
    route = route_query(case)
    for k, v in route.items():
        print(f"  {k}: {v}")