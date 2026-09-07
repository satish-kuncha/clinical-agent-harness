import pytest

from clinical_agent_harness.domain.prescription import ClinicalPrescription
from clinical_agent_harness.guardrails.models import GuardrailDecision


@pytest.fixture
def mock_llm(monkeypatch):
    from clinical_agent_harness.harness import graph as graph_module

    async def fake_guardrail(*args, **kwargs):
        return GuardrailDecision(
            allowed=True,
            reason="Input passed semantic safety evaluation.",
        )

    async def fake_prescription(*args, **kwargs):
        return type(
            "FakeResult",
            (),
            {
                "output": ClinicalPrescription(
                    patient_id="PAT-67890",
                    medication="metformin",
                    dose_mg=500,
                    frequency_per_day=1,
                    rationale="Treatment of type 2 diabetes.",
                )
            },
        )()

    monkeypatch.setattr(
        graph_module,
        "check_semantic_safety",
        fake_guardrail,
    )

    monkeypatch.setattr(
        graph_module.prescription_agent,
        "run",
        fake_prescription,
    )