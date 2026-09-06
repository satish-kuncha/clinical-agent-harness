from clinical_agent_harness.domain.patient import PatientClinicalContext
from clinical_agent_harness.domain.policy import (
    ClinicalPolicyResult,
    PolicyDecision,
)
from clinical_agent_harness.domain.prescription import ClinicalPrescription


def evaluate_renal_policy(
    patient: PatientClinicalContext,
    prescription: ClinicalPrescription,
) -> ClinicalPolicyResult:

    if patient.patient_id != prescription.patient_id:
        return ClinicalPolicyResult(
            decision=PolicyDecision.BLOCK,
            reason="Patient identity does not match the prescription.",
        )

    if (
        prescription.medication.lower() == "metformin"
        and patient.egfr < 30
        and prescription.dose_mg * prescription.frequency_per_day > 500
    ):
        return ClinicalPolicyResult(
            decision=PolicyDecision.BLOCK,
            reason=(
                "Toy renal policy: metformin daily dose exceeds "
                "the allowed threshold for eGFR below 30."
            ),
        )

    return ClinicalPolicyResult(
        decision=PolicyDecision.ALLOW,
        reason="Prescription passes the deterministic renal policy.",
    )