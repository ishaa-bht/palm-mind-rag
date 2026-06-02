import json
import re
import google.generativeai as genai
from app.core.config import settings
from app.schemas.schemas import BookingInfo

genai.configure(api_key=settings.gemini_api_key)

model = genai.GenerativeModel("gemini-2.5-flash")


async def extract_booking(
    query: str,
    history: list[dict],
) -> BookingInfo | None:
    """
    Use Gemini to detect and extract interview booking intent from conversation.
    Returns BookingInfo if all 4 fields are present, otherwise None.

    Args:
        query: Current user message
        history: Full conversation history from Redis

    Returns:
        BookingInfo if booking intent detected with all fields, else None
    """
    # Format history for context
    history_text = "\n".join(
        [f"{msg['role'].upper()}: {msg['content']}" for msg in history]
    ) if history else "No prior history."

    prompt = f"""You are an assistant that extracts interview booking information.

Analyze the conversation below and the latest message.
If the user is trying to book an interview AND all four fields
(name, email, date, time) are clearly present in the conversation,
return ONLY a valid JSON object like this:

{{"name": "John Doe", "email": "john@example.com", "date": "2024-06-10", "time": "14:00"}}

If ANY field is missing or the user is NOT booking an interview, return ONLY:
null

Do NOT return any explanation. Return ONLY the JSON object or null.

Conversation History:
{history_text}

Latest Message: {query}"""

    response = model.generate_content(prompt)
    raw = response.text.strip()

    # Strip markdown code fences if Gemini wraps in ```json ... ```
    raw = re.sub(r"```json|```", "", raw).strip()

    if raw.lower() == "null" or not raw:
        return None

    try:
        data = json.loads(raw)
        return BookingInfo(
            name=data["name"],
            email=data["email"],
            date=data["date"],
            time=data["time"],
        )
    except (json.JSONDecodeError, KeyError):
        return None
