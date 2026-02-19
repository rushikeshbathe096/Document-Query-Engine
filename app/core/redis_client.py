import redis
from app.core.config import settings

redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

def ping_redis() -> bool:
    try:
        return redis_client.ping()
    except redis.ConnectionError:
        return False
