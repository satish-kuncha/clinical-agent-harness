from clinical_agent_harness.domain.patient import PatientClinicalContext


class PatientRepository:
    def __init__(self) -> None:
        self._patients = {
            "PAT-12345": PatientClinicalContext(
                patient_id="PAT-12345",
                egfr=25,
            ),
            "PAT-67890": PatientClinicalContext(
                patient_id="PAT-67890",
                egfr=60,
            ),
        }

    def get_patient(self, patient_id: str) -> PatientClinicalContext | None:
        return self._patients.get(patient_id)