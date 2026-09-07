from typing import Any

from clinical_agent_harness.harness.graph import build_prescription_graph
from clinical_agent_harness.observability.langfuse import (
    create_langgraph_handler,
)


class PrescriptionVerificationService:
    """Application service for prescription verification."""

    def __init__(self, checkpointer: Any = None) -> None:
        self.graph = build_prescription_graph(
            checkpointer=checkpointer,
        )

        self.langfuse_handler = create_langgraph_handler()

    async def verify(
        self,
        input_text: str,
        *,
        thread_id: str,
    ) -> dict:
        """Run the prescription verification workflow."""

        config = {
            "configurable": {
                "thread_id": thread_id,
            },
            "callbacks": [self.langfuse_handler],
            "run_name": "prescription-verification",
        }

        return await self.graph.ainvoke(
            {
                "input_text": input_text,
            },
            config=config,
        )