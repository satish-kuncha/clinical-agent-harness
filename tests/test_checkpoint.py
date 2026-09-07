import pytest

from clinical_agent_harness.harness.checkpointer import create_checkpointer
from clinical_agent_harness.harness.graph import build_prescription_graph
from clinical_agent_harness.harness.state import (
    ApprovalDecision,
    WorkflowStatus,
)
from langgraph.types import Command


@pytest.mark.asyncio
async def test_workflow_resumes_after_interrupt(tmp_path):
    database_path = str(tmp_path / "clinical_agent_harness.db")

    # ---------------------------------------------------------
    # PROCESS 1: Start the workflow
    # ---------------------------------------------------------
    checkpointer_1 = await create_checkpointer(database_path)
    graph_1 = build_prescription_graph(checkpointer=checkpointer_1)

    config = {
        "configurable": {
            "thread_id": "checkpoint-test-001",
        }
    }

    interrupted_state = await graph_1.ainvoke(
        {
            "input_text": (
                "Patient PAT-67890 should receive metformin "
                "500 mg once daily for treatment of type 2 diabetes."
            ),
        },
        config=config,
    )

    assert interrupted_state["status"] == WorkflowStatus.AWAITING_APPROVAL
    assert "__interrupt__" in interrupted_state

    # ---------------------------------------------------------
    # PROCESS 2: Re-create the graph from the same database
    # ---------------------------------------------------------
    await checkpointer_1.conn.close()

    checkpointer_2 = await create_checkpointer(database_path)
    graph_2 = build_prescription_graph(checkpointer=checkpointer_2)

    # ---------------------------------------------------------
    # Resume the SAME thread with human approval
    # ---------------------------------------------------------
    final_state = await graph_2.ainvoke(
        Command(resume=ApprovalDecision.APPROVED),
        config=config,
    )

    assert final_state["status"] == WorkflowStatus.COMPLETED
    assert final_state["approval"] == ApprovalDecision.APPROVED

    await checkpointer_2.conn.close()