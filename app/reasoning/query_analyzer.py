import json
import logging
from typing import Dict, Any
from app.llm.groq import generate_with_groq

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    def analyze_query(self, query: str) -> Dict[str, Any]:
        if not query or not query.strip():
            return {
                "query_type": "unknown",
                "keywords": [],
                "expected_sections": []
            }

        prompt = f"""
You are a query analysis module.

Analyze the following user query and return STRICT JSON ONLY.

Allowed query_type values:
["definition", "procedural", "conceptual", "comparison", "lookup", "factual", "summarization", "unknown"]

Return exactly this JSON format:
{{
  "query_type": "...",
  "keywords": ["...", "..."],
  "expected_sections": ["...", "..."]
}}

Rules:
- No explanations
- No markdown
- No extra text

User query:
"{query}"
"""

        try:
            text = generate_with_groq(prompt)

            start = text.find("{")
            end = text.rfind("}")

            if start == -1 or end == -1 or end <= start:
                raise ValueError("No JSON object found")

            json_text = text[start:end + 1]
            return json.loads(json_text)

        except Exception as e:
            logger.warning("QueryAnalyzer error: %s", e)
            return {
                "query_type": "unknown",
                "keywords": [],
                "expected_sections": []
            }
