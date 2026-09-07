import asyncio

from clinical_agent_harness.application.bootstrap import (
    initialize_application,
)
from clinical_agent_harness.application.service import (
    PrescriptionVerificationService,
)
from clinical_agent_harness.observability.langfuse import langfuse


async def main() -> None:
    initialize_application()

    service = PrescriptionVerificationService()

    result = await service.verify(
        """
        Patient PAT-12345 should receive metformin 500 mg
        twice daily for treatment of type 2 diabetes.
        """,
        thread_id="observability-demo-001",
    )

    print("\nWorkflow result:")
    print(result)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        langfuse.shutdown()