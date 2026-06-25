import threading
import time
import pytest
from rate_limiter import RateLimiter, RateLimitExceeded


class TestRejectionBeyondCapacity:
    def test_decorator_allows_calls_up_to_capacity(self):
        limiter = RateLimiter(capacity=3, refill_rate=0)
        results = []

        @limiter
        def work():
            results.append(1)

        work()
        work()
        work()
        assert len(results) == 3

    def test_decorator_raises_when_capacity_exceeded(self):
        limiter = RateLimiter(capacity=2, refill_rate=0)

        @limiter
        def work():
            pass

        work()
        work()
        with pytest.raises(RateLimitExceeded):
            work()

    def test_context_manager_raises_when_capacity_exceeded(self):
        limiter = RateLimiter(capacity=1, refill_rate=0)
        with limiter:
            pass
        with pytest.raises(RateLimitExceeded):
            with limiter:
                pass


class TestTokenRefill:
    def test_tokens_refill_over_time(self):
        limiter = RateLimiter(capacity=1, refill_rate=10)
        with limiter:
            pass
        with pytest.raises(RateLimitExceeded):
            with limiter:
                pass
        time.sleep(0.15)
        with limiter:
            pass  # must not raise after refill

    def test_tokens_do_not_exceed_capacity(self):
        limiter = RateLimiter(capacity=2, refill_rate=100)
        time.sleep(0.1)  # would add 10 tokens if uncapped
        count = 0
        try:
            for _ in range(5):
                with limiter:
                    count += 1
        except RateLimitExceeded:
            pass
        assert count == 2


class TestThreadSafety:
    def test_concurrent_access_does_not_exceed_capacity(self):
        capacity = 5
        limiter = RateLimiter(capacity=capacity, refill_rate=0)
        successes = []
        mu = threading.Lock()

        def attempt():
            try:
                with limiter:
                    with mu:
                        successes.append(1)
            except RateLimitExceeded:
                pass

        threads = [threading.Thread(target=attempt) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(successes) == capacity

    def test_decorator_thread_safety(self):
        capacity = 3
        limiter = RateLimiter(capacity=capacity, refill_rate=0)
        successes = []
        mu = threading.Lock()

        @limiter
        def work():
            with mu:
                successes.append(1)

        def attempt():
            try:
                work()
            except RateLimitExceeded:
                pass

        threads = [threading.Thread(target=attempt) for _ in range(15)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(successes) == capacity


class TestOverflowPolicies:
    def test_policy_raise_raises_immediately(self):
        limiter = RateLimiter(capacity=0, refill_rate=0, policy="raise")
        start = time.monotonic()
        with pytest.raises(RateLimitExceeded):
            with limiter:
                pass
        assert time.monotonic() - start < 0.1

    def test_policy_block_waits_for_token(self):
        limiter = RateLimiter(capacity=1, refill_rate=10, policy="block")
        with limiter:
            pass  # drain
        start = time.monotonic()
        with limiter:  # blocks ~100 ms for refill
            pass
        assert time.monotonic() - start >= 0.08

    def test_policy_block_eventually_succeeds(self):
        limiter = RateLimiter(capacity=1, refill_rate=20, policy="block")

        @limiter
        def work():
            return "done"

        work()  # drain
        assert work() == "done"  # blocks ~50 ms then succeeds
