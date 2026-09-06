from enum import Enum

from pydantic import BaseModel


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"


class ClinicalPolicyResult(BaseModel):
    decision: PolicyDecision
    reason: str