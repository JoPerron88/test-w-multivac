"""
Tests for rate_limiter.py — token bucket, thread-safe, decorator + context manager.

TDD order:
  1. Rejection beyond capacity
  2. Progressive token refill over time
  3. Thread safety under parallel access
  4. Both overflow policies (raise / block)
"""

import time
import threading
import pytest

from rate_limiter import RateLimiter, RateLimitExceeded


# ---------------------------------------------------------------------------
# 1. Rejection beyond capacity — policy="raise"
# ---------------------------------------------------------------------------

class TestRejectionBeyondCapacity:
    def test_decorator_allows_calls_up_to_capacity(self):
        limiter = RateLimiter(capacity=3, refill_rate=0, policy="raise")

        @limiter
        def work():
            return "ok"

        for _ in range(3):
            assert work() == "ok"

    def test_decorator_raises_when_capacity_exceeded(self):
        limiter = RateLimiter(capacity=2, refill_rate=0, policy="raise")

        @limiter
        def work():
            return "ok"

        work()
        work()
        with pytest.raises(RateLimitExceeded):
            work()

    def test_context_manager_raises_when_capacity_exceeded(self):
        limiter = RateLimiter(capacity=1, refill_rate=0, policy="raise")

        with limiter:
            pass  # consumes the only token

        with pytest.raises(RateLimitExceeded):
            with limiter:
                pass


# ---------------------------------------------------------------------------
# 2. Progressive token refill over time
# ---------------------------------------------------------------------------

class TestTokenRefill:
    def test_tokens_refill_over_time(self):
        # 1 token capacity, 10 tokens/s → replenishes in ~100 ms
        limiter = RateLimiter(capacity=1, refill_rate=10, policy="raise")

        with limiter:
            pass  # drain the bucket

        with pytest.raises(RateLimitExceeded):
            with limiter:
                pass  # still empty

        time.sleep(0.15)  # wait for refill

        with limiter:
            pass  # should succeed now

    def test_tokens_do_not_exceed_capacity(self):
        # capacity=2, wait long enough to "overfill" — must cap at 2
        limiter = RateLimiter(capacity=2, refill_rate=100, policy="raise")

        time.sleep(0.1)  # would add 10 tokens if uncapped

        # exactly 2 calls should succeed, 3rd must fail
        with limiter:
            pass
        with limiter:
            pass
        with pytest.raises(RateLimitExceeded):
            with limiter:
                pass


# ---------------------------------------------------------------------------
# 3. Thread safety under parallel access
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_access_does_not_exceed_capacity(self):
        capacity = 5
        limiter = RateLimiter(capacity=capacity, refill_rate=0, policy="raise")

        successes = []
        failures = []
        lock = threading.Lock()

        def attempt():
            try:
                with limiter:
                    with lock:
                        successes.append(1)
            except RateLimitExceeded:
                with lock:
                    failures.append(1)

        threads = [threading.Thread(target=attempt) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(successes) == capacity
        assert len(successes) + len(failures) == 20

    def test_decorator_thread_safety(self):
        capacity = 3
        limiter = RateLimiter(capacity=capacity, refill_rate=0, policy="raise")

        @limiter
        def work():
            return True

        results = []
        lock = threading.Lock()

        def attempt():
            try:
                work()
                with lock:
                    results.append("ok")
            except RateLimitExceeded:
                with lock:
                    results.append("fail")

        threads = [threading.Thread(target=attempt) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results.count("ok") == capacity
        assert len(results) == 10


# ---------------------------------------------------------------------------
# 4. Both overflow policies
# ---------------------------------------------------------------------------

class TestOverflowPolicies:
    def test_policy_raise_raises_immediately(self):
        limiter = RateLimiter(capacity=1, refill_rate=0, policy="raise")

        with limiter:
            pass

        start = time.monotonic()
        with pytest.raises(RateLimitExceeded):
            with limiter:
                pass
        elapsed = time.monotonic() - start

        # Should fail fast — well under 100 ms
        assert elapsed < 0.1

    def test_policy_block_waits_for_token(self):
        # 1 token, refill at 10/s → ~100 ms wait
        limiter = RateLimiter(capacity=1, refill_rate=10, policy="block")

        with limiter:
            pass  # drain

        start = time.monotonic()
        with limiter:  # must block until token available
            pass
        elapsed = time.monotonic() - start

        assert elapsed >= 0.08  # waited at least ~80 ms

    def test_policy_block_eventually_succeeds(self):
        limiter = RateLimiter(capacity=1, refill_rate=20, policy="block")

        @limiter
        def work():
            return "done"

        work()  # drain
        result = work()  # blocks ~50 ms then succeeds
        assert result == "done"
