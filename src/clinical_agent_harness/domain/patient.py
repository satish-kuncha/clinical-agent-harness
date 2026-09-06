from pydantic import BaseModel, Field


class PatientClinicalContext(BaseModel):
    patient_id: str = Field(pattern=r"^PAT-[0-9]{5}$")
    egfr: float = Field(gt=0, le=200)