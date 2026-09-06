from clinical_agent_harness.domain.patient_repository import PatientRepository
from clinical_agent_harness.domain.policy import PolicyDecision
from clinical_agent_harness.domain.prescription import ClinicalPrescription
from clinical_agent_harness.policy.renal import evaluate_renal_policy


def test_policy_uses_trusted_patient_context_not_user_claim():
    repository = PatientRepository()

    # Imagine the untrusted clinical text claimed:
    #
    # "Patient PAT-12345 has eGFR 90"
    #
    # We deliberately do NOT use that value.
    #
    # The repository says eGFR = 25.

    patient = repository.get_patient("PAT-12345")

    assert patient is not None
    assert patient.egfr == 25

    prescription = ClinicalPrescription(
        patient_id="PAT-12345",
        medication="metformin",
        dose_mg=500,
        frequency_per_day=2,
        rationale="Treatment of type 2 diabetes",
    )

    result = evaluate_renal_policy(
        patient=patient,
        prescription=prescription,
    )

    assert result.decision == PolicyDecision.BLOCK