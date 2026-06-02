import json
import redis.asyncio as aioredis
from app.core.config import settings


# Single shared async Redis client
redis_client = aioredis.from_url(
    settings.redis_url,
    encoding="utf-8",
    decode_responses=True,
)

# Keep last 10 message pairs per session (20 messages total)
MAX_HISTORY = 20

# Session expires after 1 hour of inactivity
SESSION_TTL = 3600


async def get_history(session_id: str) -> list[dict]:
    """
    Retrieve chat history for a session from Redis.

    Args:
        session_id: Unique identifier for the conversation

    Returns:
        List of {role, content} dicts in chronological order
    """
    raw: list[str] = await redis_client.lrange(session_id, 0, -1)
    return [json.loads(msg) for msg in raw]


async def add_message(
    session_id: str,
    role: str,
    content: str,
) -> None:
    """
    Append a message to the session history in Redis.
    Trims history to MAX_HISTORY and refreshes TTL.

    Args:
        session_id: Unique identifier for the conversation
        role: "user" or "assistant"
        content: Message text
    """
    message = json.dumps({"role": role, "content": content})
    await redis_client.rpush(session_id, message)

    # Keep only last MAX_HISTORY messages
    await redis_client.ltrim(session_id, -MAX_HISTORY, -1)

    # Refresh session expiry on every message
    await redis_client.expire(session_id, SESSION_TTL)


async def clear_history(session_id: str) -> None:
    """
    Delete all history for a session (optional utility).

    Args:
        session_id: Unique identifier for the conversation
    """
    await redis_client.delete(session_id)