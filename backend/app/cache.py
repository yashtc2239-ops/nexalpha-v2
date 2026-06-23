"""
Cache-aside pattern with graceful degradation.

Why cache-aside (vs write-through): reads (analyze/backtest requests) far outnumber
writes here, and data only changes once a day (EOD prices) — so checking cache first,
falling through to compute-and-store on a miss, is the right tradeoff. Write-through
would mean populating cache on every write even if nobody reads it.

Why Redis: shared across multiple Flask worker processes/containers — an in-process
dict cache would be per-process and wouldn't help horizontal scaling.

Why a fallback: if Redis is down, the app should degrade to "slower" (in-memory,
single-process cache) rather than crash. This is what 'graceful degradation' means
in an interview answer — the dependency failing should never take the whole service down.
"""
import json
import time
from app.logger import logger
from app.config import config

_memory_cache = {}


class CacheClient:
    def __init__(self):
        self.backend = "memory"
        self.redis = None
        try:
            import redis
            self.redis = redis.from_url(config.REDIS_URL, socket_connect_timeout=1)
            self.redis.ping()
            self.backend = "redis"
            logger.info("Cache backend: redis")
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}); falling back to in-memory cache")
            self.backend = "memory"

    def get(self, key):
        if self.backend == "redis":
            try:
                raw = self.redis.get(key)
                return json.loads(raw) if raw else None
            except Exception as e:
                logger.warning(f"Redis GET failed, falling back to memory: {e}")
                self.backend = "memory"
        entry = _memory_cache.get(key)
        if not entry:
            return None
        value, expires_at = entry
        if expires_at < time.time():
            _memory_cache.pop(key, None)
            return None
        return value

    def set(self, key, value, ttl=None):
        ttl = ttl or config.CACHE_TTL_SECONDS
        if self.backend == "redis":
            try:
                self.redis.setex(key, ttl, json.dumps(value))
                return
            except Exception as e:
                logger.warning(f"Redis SET failed, falling back to memory: {e}")
                self.backend = "memory"
        _memory_cache[key] = (value, time.time() + ttl)


cache = CacheClient()
