"""retry_sync: bounded synchronous retry that heals transient failures and surfaces the rest at once."""

import pytest

from miles.utils.retry_utils import retry_sync


class _Clock:
    """A fake monotonic clock advanced only by the retries' own sleeps, so tests never wait."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _always(_error: BaseException) -> bool:
    return True


def test_returns_without_retrying_when_the_call_succeeds():
    clock = _Clock()
    calls = []

    def fn(attempt):
        calls.append(attempt)
        return "ok"

    assert retry_sync(fn, is_retryable=_always, sleep_fn=clock.sleep, monotonic=clock.monotonic) == "ok"
    assert calls == [0] and clock.slept == []


def test_retries_a_transient_failure_then_succeeds():
    clock = _Clock()
    attempts = {"n": 0}

    def fn(attempt):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("throttled")
        return attempts["n"]

    result = retry_sync(fn, is_retryable=_always, jitter_ratio=0.0, sleep_fn=clock.sleep, monotonic=clock.monotonic)
    assert result == 3
    assert clock.slept == [1.0, 2.0]  # exponential backoff, two retries


def test_a_non_retryable_error_propagates_on_the_first_raise():
    clock = _Clock()
    calls = {"n": 0}

    def fn(_attempt):
        calls["n"] += 1
        raise ValueError("permanent")

    with pytest.raises(ValueError, match="permanent"):
        retry_sync(
            fn, is_retryable=lambda e: not isinstance(e, ValueError), sleep_fn=clock.sleep, monotonic=clock.monotonic
        )
    assert calls["n"] == 1 and clock.slept == []


def test_honors_retry_after_secs_over_the_backoff():
    clock = _Clock()
    attempts = {"n": 0}

    class Throttled(RuntimeError):
        retry_after_secs = 7

    def fn(_attempt):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise Throttled()
        return "done"

    retry_sync(fn, is_retryable=_always, jitter_ratio=0.0, sleep_fn=clock.sleep, monotonic=clock.monotonic)
    assert clock.slept == [7.0]  # the server's Retry-After wins over the 1.0s initial backoff


def test_gives_up_and_reraises_at_the_deadline():
    clock = _Clock()
    calls = {"n": 0}

    def fn(_attempt):
        calls["n"] += 1
        raise RuntimeError("still throttled")

    with pytest.raises(RuntimeError, match="still throttled"):
        retry_sync(
            fn,
            is_retryable=_always,
            total_seconds=2.5,
            jitter_ratio=0.0,
            sleep_fn=clock.sleep,
            monotonic=clock.monotonic,
        )
    # 1.0s then 2.0s would overrun the 2.5s budget, so it stops rather than sleeping past the deadline.
    assert clock.slept == [1.0] and calls["n"] == 2
