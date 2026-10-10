import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from miles.utils.tracking_utils.structured_log import log_structured

logger = logging.getLogger(__name__)

_DEFAULT_INITIAL_DELAY = 1.0
_DEFAULT_MAX_DELAY = 60.0
_DEFAULT_BACKOFF_FACTOR = 2.0
_DEFAULT_JITTER_RATIO = 0.1
_DEFAULT_TOTAL_SECONDS = 600.0

_T = TypeVar("_T")


class NonRetryableError(Exception):
    pass


def retry_sync(
    fn: Callable[[int], _T],
    *,
    is_retryable: Callable[[BaseException], bool],
    total_seconds: float = _DEFAULT_TOTAL_SECONDS,
    initial_delay: float = _DEFAULT_INITIAL_DELAY,
    max_delay: float = _DEFAULT_MAX_DELAY,
    backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
    jitter_ratio: float = _DEFAULT_JITTER_RATIO,
    sleep_fn: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
    log_fields: dict[str, Any] | None = None,
) -> _T:
    """Synchronous retry for transient failures, bounded by a wall-clock deadline.

    ``is_retryable`` decides which exceptions heal on retry; everything else (a bad request, a missing
    object, a programming bug) propagates on the first raise. If the error carries ``retry_after_secs``
    (the Databricks SDK sets it from a 429/503 ``Retry-After`` header), that wait is honored for that
    attempt instead of the backoff. ``fn`` receives the zero-based attempt number.
    """
    sleep = sleep_fn or time.sleep
    now = monotonic or time.monotonic
    expires_at = now() + total_seconds
    delay = initial_delay
    attempt = 0
    while True:
        try:
            return fn(attempt)
        except Exception as error:
            if not is_retryable(error):
                raise
            attempt += 1
            retry_after = getattr(error, "retry_after_secs", None)
            base = float(retry_after) if isinstance(retry_after, (int, float)) and retry_after > 0 else delay
            sleep_seconds = base + random.uniform(0.0, base * jitter_ratio)
            if expires_at - now() <= sleep_seconds:
                logger.warning(f"retry_sync: giving up after {total_seconds:.1f}s", exc_info=True)
                raise
            log_structured(
                logger.info,
                **{"op": "retry_sync", **(log_fields or {})},
                phase="attempt_failed",
                attempt=attempt,
                sleep_s=round(sleep_seconds, 3),
                remaining_s=round(expires_at - now(), 3),
                error=repr(error),
            )
            sleep(sleep_seconds)
            delay = min(delay * backoff_factor, max_delay)


async def retry(
    fn: Callable[[int], Awaitable[_T]],
    *,
    initial_delay: float = _DEFAULT_INITIAL_DELAY,
    max_delay: float = _DEFAULT_MAX_DELAY,
    backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    max_attempts: int | None = None,
) -> _T:
    """Retry until ``fn`` does not throw, with exponential backoff."""
    assert max_attempts is None or max_attempts >= 1

    attempt = 0
    delay = initial_delay
    while True:
        try:
            return await fn(attempt)
        except NonRetryableError:
            raise
        except Exception:
            attempt += 1
            if max_attempts is not None and attempt >= max_attempts:
                logger.warning(f"retry: attempt {attempt} failed, giving up (max_attempts={max_attempts})")
                raise
            logger.warning(f"retry: attempt {attempt} failed, retrying in {delay:.1f}s", exc_info=True)
            await sleep_fn(delay)
            delay = min(delay * backoff_factor, max_delay)


async def retry_until_deadline(
    fn: Callable[[float], Awaitable[_T]],
    *,
    total_seconds: float,
    retry_on: type[Exception] | tuple[type[Exception], ...],
    initial_delay: float = _DEFAULT_INITIAL_DELAY,
    max_delay: float = _DEFAULT_MAX_DELAY,
    backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
    jitter_ratio: float = _DEFAULT_JITTER_RATIO,
    log_fields: dict[str, Any] | None = None,
) -> _T:
    expires_at = time.monotonic() + total_seconds
    delay = initial_delay
    attempt = 0

    while True:
        try:
            return await fn(max(0.0, expires_at - time.monotonic()))
        except retry_on as e:
            attempt += 1
            sleep_seconds = delay + random.uniform(0.0, delay * jitter_ratio)
            if expires_at - time.monotonic() <= sleep_seconds:
                logger.warning(f"retry_until_deadline: giving up after {total_seconds:.1f}s", exc_info=True)
                raise
            log_structured(
                logger.info,
                **{"op": "retry_until_deadline", **(log_fields or {})},
                phase="attempt_failed",
                attempt=attempt,
                sleep_s=round(sleep_seconds, 3),
                remaining_s=round(expires_at - time.monotonic(), 3),
                error=repr(e),
            )
            await asyncio.sleep(sleep_seconds)
            delay = min(delay * backoff_factor, max_delay)
