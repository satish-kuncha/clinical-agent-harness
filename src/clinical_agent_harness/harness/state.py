from enum import Enum
from typing import TypedDict

from clinical_agent_harness.domain.patient import PatientClinicalContext
from clinical_agent_harness.domain.policy import ClinicalPolicyResult
from clinical_agent_harness.domain.prescription import ClinicalPrescription

class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"

class WorkflowStatus(str, Enum):
    RUNNING = "running"
    BLOCKED = "blocked"
    FAILED = "failed"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"


class PrescriptionWorkflowState(TypedDict, total=False):
    input_text: str
    status: WorkflowStatus
    prescription: ClinicalPrescription | None
    patient: PatientClinicalContext | None
    policy_result: ClinicalPolicyResult | None
    reason: str | None
    approval: ApprovalDecision | None

