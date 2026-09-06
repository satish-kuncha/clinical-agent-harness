from pydantic_ai import Agent
from dotenv import load_dotenv
from clinical_agent_harness.domain.prescription import ClinicalPrescription

load_dotenv()

prescription_agent = Agent(
    "groq:openai/gpt-oss-20b",
    output_type=ClinicalPrescription,
    instructions="""
    Extract a clinical prescription from the provided clinical text.

    Return only information supported by the input.

    Do not invent:
    - patient IDs
    - medications
    - doses
    - frequencies
    - clinical rationales

    If the information cannot be determined reliably,
    do not fabricate a value.
    """,
)