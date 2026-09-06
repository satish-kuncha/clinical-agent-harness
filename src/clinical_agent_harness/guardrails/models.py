from pydantic import BaseModel, Field


class GuardrailDecision(BaseModel):
    allowed: bool
    reason: str = Field(min_length=5, max_length=500)