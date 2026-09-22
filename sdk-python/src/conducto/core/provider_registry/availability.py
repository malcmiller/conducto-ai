"""Bounded, off-lock evaluation of provider availability."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait

DEFAULT_AVAILABILITY_TIMEOUT_SECONDS = 2.0

_AVAILABILITY_EXECUTOR = ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="conducto-provider-availability"
)


def evaluate_available(available: bool | Callable[[], bool], timeout: float) -> bool:
    """Treat failed or timed-out health predicates as unavailable.

    Predicates run in a bounded worker pool, never on the registry's calling
    thread or under its synchronization lock.
    """
    if not callable(available):
        return bool(available)
    future = _AVAILABILITY_EXECUTOR.submit(available)
    done, _ = wait((future,), timeout=timeout)
    if not done:
        future.cancel()
        return False
    if future.exception() is not None:
        # Health predicates are application code; any failure means unavailable.
        return False
    return bool(future.result())
