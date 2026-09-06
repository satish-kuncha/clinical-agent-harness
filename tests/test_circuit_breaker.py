import asyncio

import pytest

from clinical_agent_harness.harness.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
)
from clinical_agent_harness.harness.errors import RetryableError


@pytest.mark.asyncio
async def test_circuit_breaker_starts_closed():
    breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout_seconds=1.0,
    )

    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_failure_threshold():
    breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout_seconds=1.0,
    )

    async def failing_operation():
        raise RetryableError("Provider unavailable")

    for _ in range(3):
        with pytest.raises(RetryableError):
            await breaker.call(failing_operation)

    assert breaker.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_open_circuit_fails_fast_without_calling_operation():
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout_seconds=10.0,
    )

    calls = 0

    async def failing_operation():
        nonlocal calls
        calls += 1
        raise RetryableError("Provider unavailable")

    # First failure opens the circuit.
    with pytest.raises(RetryableError):
        await breaker.call(failing_operation)

    assert breaker.state == CircuitState.OPEN
    assert calls == 1

    # Second call must not reach the provider.
    with pytest.raises(Exception):
        await breaker.call(failing_operation)

    assert calls == 1


@pytest.mark.asyncio
async def test_successful_half_open_probe_closes_circuit():
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout_seconds=0.05,
    )

    async def failing_operation():
        raise RetryableError("Provider unavailable")

    async def successful_operation():
        return "success"

    # Open the circuit.
    with pytest.raises(RetryableError):
        await breaker.call(failing_operation)

    assert breaker.state == CircuitState.OPEN

    # Wait for recovery period.
    await asyncio.sleep(0.1)

    # The next request should be allowed as a HALF_OPEN probe.
    result = await breaker.call(successful_operation)

    assert result == "success"
    assert breaker.state == CircuitState.CLOSED