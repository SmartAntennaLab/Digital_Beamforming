"""Redis-backed global concurrency and token-bucket rate coordination."""

from __future__ import annotations

import hashlib
import math
import os
import time
import uuid
from dataclasses import dataclass
from typing import Protocol

from resource_policy import ResourcePolicy

_TOKEN_BUCKET_SCRIPT = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_per_ms = tonumber(ARGV[2])
local now_ms = tonumber(ARGV[3])
local ttl_ms = tonumber(ARGV[4])
local bucket = redis.call('HMGET', key, 'tokens', 'updated_ms')
local tokens = tonumber(bucket[1])
local updated_ms = tonumber(bucket[2])
if tokens == nil then
  tokens = capacity
  updated_ms = now_ms
end
local elapsed_ms = math.max(0, now_ms - updated_ms)
tokens = math.min(capacity, tokens + elapsed_ms * refill_per_ms)
local allowed = 0
local retry_ms = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
else
  retry_ms = math.ceil((1 - tokens) / refill_per_ms)
end
redis.call('HSET', key, 'tokens', tokens, 'updated_ms', now_ms)
redis.call('PEXPIRE', key, ttl_ms)
return {allowed, tostring(tokens), tostring(retry_ms)}
"""

_ACQUIRE_SLOT_SCRIPT = """
local key = KEYS[1]
local now_ms = tonumber(ARGV[1])
local expires_ms = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local token = ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, '-inf', now_ms)
local active = redis.call('ZCARD', key)
if active >= limit then
  return {0, active}
end
redis.call('ZADD', key, expires_ms, token)
redis.call('PEXPIRE', key, math.max(1000, expires_ms - now_ms + 1000))
return {1, active + 1}
"""

_ACTIVE_SLOTS_SCRIPT = """
local key = KEYS[1]
local now_ms = tonumber(ARGV[1])
redis.call('ZREMRANGEBYSCORE', key, '-inf', now_ms)
return redis.call('ZCARD', key)
"""


class CoordinationUnavailableError(RuntimeError):
    """Raised when a configured shared coordinator cannot be reached."""


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: float = 0.0


@dataclass(frozen=True)
class CoordinationSnapshot:
    backend: str
    available: bool
    global_active_calculations: int
    global_max_concurrent_calculations: int


class GlobalComputeCoordinator(Protocol):
    @property
    def backend_name(self) -> str: ...

    def consume_session_token(self, identity: str) -> RateLimitDecision: ...

    def acquire_slot(self, token: str) -> bool: ...

    def release_slot(self, token: str) -> None: ...

    def snapshot(self) -> CoordinationSnapshot: ...

    def ping(self) -> bool: ...


def _decode_number(value: object) -> float:
    if isinstance(value, bytes):
        value = value.decode("ascii")
    return float(value)


def _redis_key_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class RedisComputeCoordinator:
    """Coordinate admission atomically across Streamlit replicas."""

    def __init__(
        self,
        client,
        policy: ResourcePolicy,
        *,
        key_prefix: str = "dbf",
        wall_clock=time.time,
        monotonic_clock=time.monotonic,
        sleeper=time.sleep,
    ) -> None:
        if not key_prefix or any(character.isspace() for character in key_prefix):
            raise ValueError("Redis key prefix must be non-empty without whitespace.")
        self._client = client
        self._policy = policy
        self._prefix = key_prefix
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self._sleeper = sleeper

    @property
    def backend_name(self) -> str:
        return "redis"

    @property
    def _slots_key(self) -> str:
        return f"{self._prefix}:compute:slots"

    def consume_session_token(self, identity: str) -> RateLimitDecision:
        if not identity:
            raise ValueError("A non-empty rate-limit identity is required.")
        now_ms = int(self._wall_clock() * 1000.0)
        capacity = self._policy.session_burst
        refill_per_ms = self._policy.session_calculations_per_minute / 60_000.0
        ttl_ms = max(
            60_000,
            int(math.ceil(capacity / refill_per_ms * 2.0)),
        )
        key = f"{self._prefix}:rate:{_redis_key_digest(identity)}"
        try:
            response = self._client.eval(
                _TOKEN_BUCKET_SCRIPT,
                1,
                key,
                capacity,
                refill_per_ms,
                now_ms,
                ttl_ms,
            )
            allowed = bool(int(response[0]))
            retry_ms = _decode_number(response[2])
        except Exception as error:
            raise CoordinationUnavailableError(
                "Redis rate-limit coordination is unavailable."
            ) from error
        return RateLimitDecision(
            allowed=allowed,
            retry_after_seconds=max(0.0, retry_ms / 1000.0),
        )

    def acquire_slot(self, token: str) -> bool:
        if not token:
            raise ValueError("A non-empty global slot token is required.")
        deadline = (
            self._monotonic_clock()
            + self._policy.compute_queue_timeout_seconds
        )
        lease_ms = int(
            math.ceil(
                (
                    self._policy.compute_timeout_seconds
                    + self._policy.compute_queue_timeout_seconds
                    + 5.0
                )
                * 1000.0
            )
        )
        while True:
            now_ms = int(self._wall_clock() * 1000.0)
            try:
                response = self._client.eval(
                    _ACQUIRE_SLOT_SCRIPT,
                    1,
                    self._slots_key,
                    now_ms,
                    now_ms + lease_ms,
                    self._policy.global_max_concurrent_calculations,
                    token,
                )
            except Exception as error:
                raise CoordinationUnavailableError(
                    "Redis concurrency coordination is unavailable."
                ) from error
            if bool(int(response[0])):
                return True
            remaining = deadline - self._monotonic_clock()
            if remaining <= 0.0:
                return False
            self._sleeper(min(0.05, remaining))

    def release_slot(self, token: str) -> None:
        if not token:
            return
        try:
            self._client.zrem(self._slots_key, token)
        except Exception as error:
            raise CoordinationUnavailableError(
                "Redis concurrency slot release failed."
            ) from error

    def snapshot(self) -> CoordinationSnapshot:
        try:
            active = int(
                self._client.eval(
                    _ACTIVE_SLOTS_SCRIPT,
                    1,
                    self._slots_key,
                    int(self._wall_clock() * 1000.0),
                )
            )
        except Exception:
            return CoordinationSnapshot(
                backend=self.backend_name,
                available=False,
                global_active_calculations=0,
                global_max_concurrent_calculations=(
                    self._policy.global_max_concurrent_calculations
                ),
            )
        return CoordinationSnapshot(
            backend=self.backend_name,
            available=True,
            global_active_calculations=active,
            global_max_concurrent_calculations=(
                self._policy.global_max_concurrent_calculations
            ),
        )

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:
            return False

    @staticmethod
    def new_slot_token() -> str:
        return uuid.uuid4().hex


def create_redis_coordinator_from_environment(
    policy: ResourcePolicy,
) -> RedisComputeCoordinator | None:
    redis_url = os.getenv("DBF_REDIS_URL", "").strip()
    if not redis_url:
        return None
    try:
        import redis
    except ImportError as error:
        raise RuntimeError(
            "DBF_REDIS_URL requires the 'ops' dependency extra."
        ) from error
    client = redis.Redis.from_url(
        redis_url,
        socket_connect_timeout=0.5,
        socket_timeout=0.5,
        health_check_interval=30,
        decode_responses=False,
    )
    return RedisComputeCoordinator(
        client,
        policy,
        key_prefix=os.getenv("DBF_REDIS_PREFIX", "dbf").strip() or "dbf",
    )


__all__ = [
    "CoordinationSnapshot",
    "CoordinationUnavailableError",
    "GlobalComputeCoordinator",
    "RateLimitDecision",
    "RedisComputeCoordinator",
    "create_redis_coordinator_from_environment",
]
