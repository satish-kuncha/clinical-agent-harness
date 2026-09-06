import pytest

from clinical_agent_harness.harness.inner import run_inner_harness


@pytest.mark.asyncio
async def test_prescription_agent_failure_is_not_silently_ignored(
    monkeypatch,
):
    async def failing_agent(*args, **kwargs):
        raise RuntimeError("Simulated model unavailable")

    monkeypatch.setattr(
        "clinical_agent_harness.harness.inner.prescription_agent.run",
        failing_agent,
    )

    with pytest.raises(RuntimeError, match="Simulated model unavailable"):
        await run_inner_harness(
            """
            Patient PAT-12345 should receive metformin 500 mg twice daily
            for treatment of type 2 diabetes.
            """
        )


@pytest.mark.asyncio
async def test_guardrail_failure_prevents_prescription_generation(
    monkeypatch,
):
    async def failing_guardrail(*args, **kwargs):
        raise RuntimeError("Simulated guardrail unavailable")

    monkeypatch.setattr(
        "clinical_agent_harness.harness.inner.check_semantic_safety",
        failing_guardrail,
    )

    with pytest.raises(RuntimeError, match="Simulated guardrail unavailable"):
        await run_inner_harness(
            """
            Patient PAT-12345 should receive metformin 500 mg twice daily
            for treatment of type 2 diabetes.
            """
        )