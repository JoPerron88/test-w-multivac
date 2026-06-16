"""
rate_limiter.py — Token bucket rate limiter.

Thread-safe. Usable as a decorator and as a context manager.
Overflow policies: "raise" (immediate RateLimitExceeded) or "block" (wait for token).
"""

import threading
import time
import functools


class RateLimitExceeded(Exception):
    """Raised when no token is available and policy is 'raise'."""


class RateLimiter:
    """
    Token bucket rate limiter.

    Parameters
    ----------
    capacity : int
        Maximum number of tokens the bucket can hold.
    refill_rate : float
        Tokens added per second (0 means no refill).
    policy : str
        "raise" — raise RateLimitExceeded when empty.
        "block" — block the caller until a token is available.
    """

    def __init__(self, capacity: int, refill_rate: float, policy: str = "raise") -> None:
        if policy not in ("raise", "block"):
            raise ValueError(f"Unknown policy: {policy!r}. Use 'raise' or 'block'.")
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._policy = policy
        self._tokens: float = float(capacity)
        self._last_refill: float = time.monotonic()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        """Add tokens based on elapsed time. Must be called under the lock."""
        if self._refill_rate <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_refill
        added = elapsed * self._refill_rate
        self._tokens = min(self._capacity, self._tokens + added)
        self._last_refill = now

    def _acquire_raise(self) -> None:
        """Consume one token or raise RateLimitExceeded."""
        with self._lock:
            self._refill()
            if self._tokens < 1:
                raise RateLimitExceeded("Rate limit exceeded.")
            self._tokens -= 1

    def _acquire_block(self) -> None:
        """Consume one token, blocking until one is available."""
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                # Calculate how long until the next token is ready.
                wait = (1 - self._tokens) / self._refill_rate
            time.sleep(wait)

    def acquire(self) -> None:
        """Acquire a token according to the configured policy."""
        if self._policy == "raise":
            self._acquire_raise()
        else:
            self._acquire_block()

    # ------------------------------------------------------------------
    # Context manager interface
    # ------------------------------------------------------------------

    def __enter__(self) -> "RateLimiter":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    # ------------------------------------------------------------------
    # Decorator interface
    # ------------------------------------------------------------------

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.acquire()
            return func(*args, **kwargs)
        return wrapper
