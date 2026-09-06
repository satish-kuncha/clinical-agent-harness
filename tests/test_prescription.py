import pytest
from pydantic import ValidationError

from clinical_agent_harness.domain.prescription import ClinicalPrescription


def test_valid_prescription():
    prescription = ClinicalPrescription(
        patient_id="PAT-12345",
        medication="metformin",
        dose_mg=500,
        frequency_per_day=2,
        rationale="Treatment for type 2 diabetes."
    )

    assert prescription.dose_mg == 500


def test_invalid_patient_id():
    with pytest.raises(ValidationError):
        ClinicalPrescription(
            patient_id="PAT-ABC",
            medication="metformin",
            dose_mg=500,
            frequency_per_day=2,
            rationale="Treatment for type 2 diabetes."
        )


def test_invalid_dose():
    with pytest.raises(ValidationError):
        ClinicalPrescription(
            patient_id="PAT-12345",
            medication="metformin",
            dose_mg=-500,
            frequency_per_day=2,
            rationale="Treatment for type 2 diabetes."
        )


def test_invalid_frequency():
    with pytest.raises(ValidationError):
        ClinicalPrescription(
            patient_id="PAT-12345",
            medication="metformin",
            dose_mg=500,
            frequency_per_day=10,
            rationale="Treatment for type 2 diabetes."
        )