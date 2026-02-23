from google import genai
from app.core.config import settings

_client = genai.Client(api_key=settings.GEMINI_API_KEY)

def generate_with_gemini(prompt: str) -> str:
    """
    Sends prompt to Gemini and returns plain text response.
    Keeps SDK isolated inside this module.
    """
    response = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text