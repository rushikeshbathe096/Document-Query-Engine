from groq import Groq
from app.core.config import settings

_client = Groq(api_key=settings.GROQ_API_KEY)

def generate_with_groq(prompt: str) -> str:
    response = _client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content
