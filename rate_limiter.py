import functools
import threading
import time


class RateLimitExceeded(Exception):
    pass


class RateLimiter:
    def __init__(self, capacity: int, refill_rate: float, policy: str = "raise") -> None:
        if policy not in ("raise", "block"):
            raise ValueError(f"policy must be 'raise' or 'block', got {policy!r}")
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._policy = policy
        self._tokens: float = float(capacity)
        self._last: float = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        if self._refill_rate <= 0:
            return
        now = time.monotonic()
        self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._refill_rate)
        self._last = now

    def acquire(self) -> None:
        if self._policy == "raise":
            with self._lock:
                self._refill()
                if self._tokens < 1:
                    raise RateLimitExceeded
                self._tokens -= 1
        else:
            while True:
                with self._lock:
                    self._refill()
                    if self._tokens >= 1:
                        self._tokens -= 1
                        return
                    wait = (1 - self._tokens) / self._refill_rate
                time.sleep(wait)

    def __enter__(self) -> "RateLimiter":
        self.acquire()
        return self

    def __exit__(self, *_) -> None:
        pass

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.acquire()
            return func(*args, **kwargs)
        return wrapper
