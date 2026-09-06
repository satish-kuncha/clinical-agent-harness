from clinical_agent_harness.domain.prescription import ClinicalPrescription


def test_structurally_valid_prescription_has_no_clinical_policy_check():
    prescription = ClinicalPrescription(
        patient_id="PAT-12345",
        medication="metformin",
        dose_mg=500,
        frequency_per_day=2,
        rationale="Treatment of type 2 diabetes",
    )

    daily_dose = (
        prescription.dose_mg
        * prescription.frequency_per_day
    )

    # This prescription is structurally valid.
    assert prescription.patient_id == "PAT-12345"
    assert daily_dose == 1000

    # IMPORTANT:
    # There is currently no deterministic clinical policy
    # that can evaluate this prescription against renal function.