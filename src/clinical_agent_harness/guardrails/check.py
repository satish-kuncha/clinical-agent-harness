from clinical_agent_harness.guardrails.models import GuardrailDecision
from clinical_agent_harness.guardrails.semantic import semantic_guardrail


async def check_semantic_safety(text: str) -> GuardrailDecision:
    result = await semantic_guardrail.run(text)

    return result.output