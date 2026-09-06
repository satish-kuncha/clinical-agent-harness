import pytest

from clinical_agent_harness.harness.retry import run_with_retry
from clinical_agent_harness.harness.errors import RetryableError

@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_failures():
    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1

        if attempts < 3:
            raise RuntimeError("Temporary failure")

        return "success"

    result = await run_with_retry(operation)

    assert result == "success"
    assert attempts == 3


@pytest.mark.asyncio
async def test_retry_stops_after_three_attempts():
    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("Persistent failure")

    with pytest.raises(RuntimeError, match="Persistent failure"):
        await run_with_retry(operation)

    assert attempts == 3

import asyncio


@pytest.mark.asyncio
async def test_retry_times_out_slow_operation():
    attempts = 0

    async def slow_operation():
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(1)

    with pytest.raises(asyncio.TimeoutError):
        await run_with_retry(
            slow_operation,
            max_attempts=3,
            timeout_seconds=0.01,
        )

    assert attempts == 3

@pytest.mark.asyncio
async def test_non_retryable_error_fails_immediately():
    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1
        raise ValueError("permanent failure")

    with pytest.raises(ValueError):
        await run_with_retry(
            operation,
            max_attempts=3,
        )

    assert attempts == 1


@pytest.mark.asyncio
async def test_retryable_error_is_retried():
    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1

        if attempts < 3:
            raise RetryableError("temporary failure")

        return "success"

    result = await run_with_retry(
        operation,
        max_attempts=3,
    )

    assert result == "success"
    assert attempts == 3