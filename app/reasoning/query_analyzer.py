import json
import logging
import re
from typing import Dict, Any
from app.llm.groq import generate_with_groq

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    def _extract_first_json_object(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        decoder = json.JSONDecoder()
        for index, character in enumerate(cleaned):
            if character != "{":
                continue

            try:
                parsed, _ = decoder.raw_decode(cleaned[index:])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        raise ValueError("No valid JSON object found")

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
            parsed = self._extract_first_json_object(text)

            return {
                "query_type": parsed.get("query_type", "unknown"),
                "keywords": parsed.get("keywords", []),
                "expected_sections": parsed.get("expected_sections", []),
            }

        except Exception as e:
            logger.warning("QueryAnalyzer fallback to unknown: %s", e)
            return {
                "query_type": "unknown",
                "keywords": [],
                "expected_sections": []
            }
