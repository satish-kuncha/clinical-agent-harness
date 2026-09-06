from clinical_agent_harness.domain.patient_repository import PatientRepository


def test_repository_returns_patient_context():
    repository = PatientRepository()

    patient = repository.get_patient("PAT-12345")

    assert patient is not None
    assert patient.patient_id == "PAT-12345"
    assert patient.egfr == 25


def test_repository_returns_none_for_unknown_patient():
    repository = PatientRepository()

    patient = repository.get_patient("PAT-99999")

    assert patient is None