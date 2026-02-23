import os
import json
from typing import Dict, Any
from google import genai
from dotenv import load_dotenv

load_dotenv()


class QueryAnalyzer:
    """
    Uses Gemini LLM to analyze the intent and structure of a user query.
    Outputs STRICT JSON describing query intent only.
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

        self.client = genai.Client(api_key=api_key)
        self.model = model

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
["definition", "procedural", "conceptual", "comparison", "lookup", "unknown"]

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
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )

            text = response.text.strip()

            # Extract JSON safely
            start = text.find("{")
            end = text.rfind("}")

            if start == -1 or end == -1 or end <= start:
                raise ValueError("No JSON object found")

            json_text = text[start:end + 1]
            parsed_json = json.loads(json_text)

            return parsed_json

        except Exception as e:
            print("QueryAnalyzer error:", e)
            return {
                "query_type": "unknown",
                "keywords": [],
                "expected_sections": []
            }


