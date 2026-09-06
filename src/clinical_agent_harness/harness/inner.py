from clinical_agent_harness.agent.prescription_agent import prescription_agent
from clinical_agent_harness.guardrails.check import check_semantic_safety
from clinical_agent_harness.harness.models import InnerHarnessResult


async def run_inner_harness(text: str) -> InnerHarnessResult:
    guardrail = await check_semantic_safety(text)

    if not guardrail.allowed:
        return InnerHarnessResult(
            allowed=False,
            guardrail=guardrail,
            prescription=None,
        )

    result = await prescription_agent.run(text)

    return InnerHarnessResult(
        allowed=True,
        guardrail=guardrail,
        prescription=result.output,
    )