from langgraph.graph import END, START, StateGraph

from clinical_agent_harness.agent.prescription_agent import prescription_agent
from clinical_agent_harness.guardrails.check import check_semantic_safety
from clinical_agent_harness.domain.patient_repository import PatientRepository
from clinical_agent_harness.policy.renal import evaluate_renal_policy
from clinical_agent_harness.domain.policy import PolicyDecision
from clinical_agent_harness.harness.retry import run_with_retry
from clinical_agent_harness.harness.state import (
    ApprovalDecision,
    PrescriptionWorkflowState,
    WorkflowStatus,
)
from langgraph.types import interrupt

#nodes
async def guardrail_node(
    state: PrescriptionWorkflowState,
) -> dict:
    async def operation():
        return await check_semantic_safety(
            state["input_text"]
        )

    try:
        decision = await run_with_retry(
            operation,
            max_attempts=3,
            timeout_seconds=10.0,
        )
    except Exception as exc:
        return {
            "status": WorkflowStatus.FAILED,
            "reason": "Safety guardrail unavailable after 3 attempts.",
        }

    if not decision.allowed:
        return {
            "status": WorkflowStatus.BLOCKED,
            "reason": decision.reason,
        }

    return {
        "status": WorkflowStatus.RUNNING,
        "reason": None,
    }


async def prescription_node(
    state: PrescriptionWorkflowState,
) -> dict:
    async def operation():
        return await prescription_agent.run(
            state["input_text"]
        )

    try:
        result = await run_with_retry(
            operation,
            max_attempts=3,
            timeout_seconds=10.0,
        )
    except Exception as exc:
        print(f"\nPRESCRIPTION NODE ERROR: {type(exc).__name__}: {exc}")
        return {
                "status": WorkflowStatus.FAILED,
                "reason": "Prescription agent unavailable after 3 attempts.",
            }

    return {
        "status": WorkflowStatus.RUNNING,
        "reason": None,
        "prescription": result.output,
    }


async def patient_lookup_node(
    state: PrescriptionWorkflowState,
) -> dict:
    repository = PatientRepository()

    prescription = state["prescription"]

    if prescription is None:
        return {
            "status": WorkflowStatus.FAILED,
            "reason": "Prescription is missing.",
        }

    patient = repository.get_patient(prescription.patient_id)

    if patient is None:
        return {
            "status": WorkflowStatus.BLOCKED,
            "reason": "Patient was not found.",
        }

    return {
        "status": WorkflowStatus.RUNNING,
        "reason": None,
        "patient": patient,
    }


async def policy_node(
    state: PrescriptionWorkflowState,
) -> dict:
    prescription = state["prescription"]
    patient = state["patient"]

    if prescription is None or patient is None:
        return {
            "status": WorkflowStatus.FAILED,
            "reason": "Required policy inputs are missing.",
        }

    result = evaluate_renal_policy(
        patient=patient,
        prescription=prescription,
    )

    if result.decision == PolicyDecision.BLOCK:
        return {
            "status": WorkflowStatus.BLOCKED,
            "reason": result.reason,
            "policy_result": result,
        }

    return {
        "status": WorkflowStatus.AWAITING_APPROVAL,
        "reason": result.reason,
        "policy_result": result,
    }

async def approval_node(
    state: PrescriptionWorkflowState,
) -> dict:
    approval = state.get("approval")

    if approval is None:
        approval = interrupt(
            {
                "type": "human_approval_required",
                "message": "Prescription requires human approval.",
                "patient_id": (
                    state["prescription"].patient_id
                    if state.get("prescription")
                    else None
                ),
                "prescription": state.get("prescription"),
            }
        )

    if approval == ApprovalDecision.APPROVED:
        return {
            "status": WorkflowStatus.COMPLETED,
            "reason": "Prescription approved.",
            "approval": ApprovalDecision.APPROVED,
        }

    if approval == ApprovalDecision.REJECTED:
        return {
            "status": WorkflowStatus.BLOCKED,
            "reason": "Prescription rejected during human approval.",
            "approval": ApprovalDecision.REJECTED,
        }

    return {
        "status": WorkflowStatus.FAILED,
        "reason": "Invalid approval decision.",
    }

#edge routings
def route_after_guardrail(
    state: PrescriptionWorkflowState,
) -> str:
    status = state.get("status")

    if status == WorkflowStatus.BLOCKED:
        return "blocked"

    if status == WorkflowStatus.FAILED:
        return "failed"

    return "allowed"

def route_after_policy(
    state: PrescriptionWorkflowState,
) -> str:
    status = state.get("status")

    if status == WorkflowStatus.BLOCKED:
        return "blocked"

    if status == WorkflowStatus.AWAITING_APPROVAL:
        return "approval"

    return "failed"

def route_after_prescription(
    state: PrescriptionWorkflowState,
) -> str:
    if state.get("status") == WorkflowStatus.FAILED:
        return "failed"

    return "success"

def route_after_approval(
    state: PrescriptionWorkflowState,
) -> str:
    status = state.get("status")

    if status == WorkflowStatus.COMPLETED:
        return "completed"

    if status == WorkflowStatus.BLOCKED:
        return "blocked"

    return "failed"




#putting the graph together
def build_prescription_graph(checkpointer=None,):
    graph = StateGraph(PrescriptionWorkflowState)

    graph.add_node("guardrail", guardrail_node)
    graph.add_node("prescription", prescription_node)
    graph.add_node("patient_lookup", patient_lookup_node)
    graph.add_node("policy", policy_node)
    graph.add_node("approval", approval_node)

    graph.add_edge(START, "guardrail")

    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {
            "allowed": "prescription",
            "blocked": END,
            "failed": END,
        },
    )

    graph.add_conditional_edges(
        "prescription",
        route_after_prescription,
        {
            "failed": END,
            "success": "patient_lookup",
        },
    )

    graph.add_edge("patient_lookup", "policy")

    graph.add_conditional_edges(
        "policy",
        route_after_policy,
        {
            "blocked": END,
            "approval": "approval",
            "failed": END,
        },
    )

    graph.add_conditional_edges(
        "approval",
        route_after_approval,
        {
            "completed": END,
            "blocked": END,
            "failed": END,
        },
    )


    return graph.compile(checkpointer=checkpointer,)