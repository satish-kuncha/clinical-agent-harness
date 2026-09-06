import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from clinical_agent_harness.harness.errors import RetryableError

T = TypeVar("T")


async def run_with_retry(
    operation: Callable[[], Awaitable[T]],
    max_attempts: int = 3,
    delay_seconds: float = 0.1,
    timeout_seconds: float = 10.0,
) -> T:
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return await asyncio.wait_for(
                operation(),
                timeout=timeout_seconds,
            )

        except asyncio.TimeoutError as exc:
            last_error = exc

        except RetryableError as exc:
            last_error = exc

        except Exception:
            raise

        if attempt < max_attempts - 1:
            await asyncio.sleep(delay_seconds)

    if last_error is not None:
        raise last_error

    raise RuntimeError("Operation failed without an exception.")