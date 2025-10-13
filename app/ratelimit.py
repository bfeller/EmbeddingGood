from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Tuple


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


REQUESTS_PER_MINUTE_DEFAULT = 120
BURST_DEFAULT = 60


@dataclass
class TokenBucket:
    capacity: int
    refill_rate_per_sec: float
    tokens: float
    last_refill_ts: float

    def consume(self, amount: float = 1.0) -> Tuple[bool, float]:
        now = time.time()
        elapsed = max(0.0, now - self.last_refill_ts)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate_per_sec)
        self.last_refill_ts = now
        if self.tokens >= amount:
            self.tokens -= amount
            # remaining seconds until full reset (approx)
            to_refill = self.capacity - self.tokens
            reset_sec = to_refill / self.refill_rate_per_sec if self.refill_rate_per_sec > 0 else 60.0
            return True, reset_sec
        # time until 1 token available
        needed = amount - self.tokens
        wait_sec = needed / self.refill_rate_per_sec if self.refill_rate_per_sec > 0 else 60.0
        return False, wait_sec


class RateLimiter:
    def __init__(self) -> None:
        self.requests_per_minute = _env_int("RL_REQUESTS_PER_MINUTE", REQUESTS_PER_MINUTE_DEFAULT)
        self.burst = _env_int("RL_BURST", BURST_DEFAULT)
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def _get_bucket(self, key: str) -> TokenBucket:
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(
                    capacity=self.burst,
                    refill_rate_per_sec=self.requests_per_minute / 60.0,
                    tokens=float(self.burst),
                    last_refill_ts=time.time(),
                )
                self._buckets[key] = bucket
            return bucket

    def allow(self, key: str) -> Tuple[bool, float, int, int]:
        bucket = self._get_bucket(key)
        ok, reset_sec = bucket.consume(1.0)
        remaining = int(max(0, bucket.tokens))
        return ok, reset_sec, self.requests_per_minute, remaining


rate_limiter = RateLimiter()


