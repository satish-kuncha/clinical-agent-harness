from pydantic import BaseModel

from clinical_agent_harness.domain.prescription import ClinicalPrescription
from clinical_agent_harness.guardrails.models import GuardrailDecision


class InnerHarnessResult(BaseModel):
    allowed: bool
    guardrail: GuardrailDecision
    prescription: ClinicalPrescription | None = None