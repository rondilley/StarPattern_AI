"""Exponential backoff with jitter for API calls."""

from __future__ import annotations

import asyncio
import functools
import random
import time
from typing import Any, Callable, TypeVar

from star_pattern.utils.logging import get_logger

logger = get_logger("retry")

F = TypeVar("F", bound=Callable[..., Any])

# HTTP statuses that will never succeed on retry: the request itself is
# wrong, or the caller is not permitted. Retrying these burns the full
# backoff schedule (roughly 14 seconds at the default settings) for a
# guaranteed failure, and hides the real cause behind a timeout-shaped
# delay. 408 and 429 are absent on purpose; both are worth retrying.
_PERMANENT_HTTP_STATUSES = frozenset({400, 401, 403, 404, 405, 409, 410, 422})

# Class names of the same failures, for clients that do not expose a
# status code. Matched by name so that optional vendor packages do not
# have to be importable here.
_PERMANENT_EXCEPTION_NAMES = frozenset(
    {
        "AuthenticationError",
        "BadRequestError",
        "InvalidArgument",
        "NotFoundError",
        "NotFound",
        "PermissionDenied",
        "PermissionDeniedError",
        "UnprocessableEntityError",
    }
)


def is_permanent_failure(exc: BaseException) -> bool:
    """True when retrying the call cannot change the outcome."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _PERMANENT_HTTP_STATUSES:
        return True
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int) and status in _PERMANENT_HTTP_STATUSES:
        return True
    return type(exc).__name__ in _PERMANENT_EXCEPTION_NAMES


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator for retry with exponential backoff and jitter.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay between retries.
        exponential_base: Multiplier for exponential growth.
        jitter: Add random jitter to delay.
        retryable_exceptions: Exception types that trigger retry.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if is_permanent_failure(e):
                        logger.error(f"{func.__name__} failed permanently: {e}. " f"Not retrying.")
                        raise
                    if attempt == max_retries:
                        break
                    delay = min(base_delay * (exponential_base**attempt), max_delay)
                    if jitter:
                        delay *= 0.5 + random.random()
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} "
                        f"failed: {e}. Retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
            raise last_exception  # type: ignore[misc]

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if is_permanent_failure(e):
                        logger.error(f"{func.__name__} failed permanently: {e}. " f"Not retrying.")
                        raise
                    if attempt == max_retries:
                        break
                    delay = min(base_delay * (exponential_base**attempt), max_delay)
                    if jitter:
                        delay *= 0.5 + random.random()
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} "
                        f"failed: {e}. Retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
            raise last_exception  # type: ignore[misc]

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator
