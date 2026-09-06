from clinical_agent_harness.domain.patient import PatientClinicalContext
from clinical_agent_harness.domain.policy import PolicyDecision
from clinical_agent_harness.domain.prescription import ClinicalPrescription
from clinical_agent_harness.policy.renal import evaluate_renal_policy


def test_metformin_allowed_when_egfr_is_above_threshold():
    patient = PatientClinicalContext(
        patient_id="PAT-12345",
        egfr=45,
    )

    prescription = ClinicalPrescription(
        patient_id="PAT-12345",
        medication="metformin",
        dose_mg=500,
        frequency_per_day=2,
        rationale="Treatment of type 2 diabetes",
    )

    result = evaluate_renal_policy(patient, prescription)

    assert result.decision == PolicyDecision.ALLOW


def test_metformin_blocked_when_egfr_is_below_threshold():
    patient = PatientClinicalContext(
        patient_id="PAT-12345",
        egfr=25,
    )

    prescription = ClinicalPrescription(
        patient_id="PAT-12345",
        medication="metformin",
        dose_mg=500,
        frequency_per_day=2,
        rationale="Treatment of type 2 diabetes",
    )

    result = evaluate_renal_policy(patient, prescription)

    assert result.decision == PolicyDecision.BLOCK


def test_patient_identity_mismatch_is_blocked():
    patient = PatientClinicalContext(
        patient_id="PAT-12345",
        egfr=45,
    )

    prescription = ClinicalPrescription(
        patient_id="PAT-99999",
        medication="metformin",
        dose_mg=500,
        frequency_per_day=2,
        rationale="Treatment of type 2 diabetes",
    )

    result = evaluate_renal_policy(patient, prescription)

    assert result.decision == PolicyDecision.BLOCK