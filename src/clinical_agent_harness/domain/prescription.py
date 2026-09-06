from pydantic import BaseModel, Field


class ClinicalPrescription(BaseModel):
    patient_id: str = Field(
        pattern=r"^PAT-[0-9]{5}$"
    )

    medication: str = Field(
        min_length=2,
        max_length=100
    )

    dose_mg: float = Field(
        gt=0,
        lt=2000
    )

    frequency_per_day: int = Field(
        ge=1,
        le=4
    )

    rationale: str = Field(
        min_length=15,
        max_length=500
    )