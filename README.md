
### `README.md`

```markdown
# Clinical Agent Harness

A production-oriented learning project for building a **Clinical Prescription & Contraindication Verification Agent** using an explicit Agent Harness architecture.

The central idea is:

> **Agent = Model + Harness**

The LLM is treated as an **untrusted cognitive engine**.

The harness provides:

- Type contracts
- Semantic safety checks
- Trusted data boundaries
- Deterministic policies
- Workflow/state-machine control
- Retries
- Timeouts
- Circuit breakers
- Durable state
- Human-in-the-loop approval
- Behavioral evaluations
- Observability
- Production hardening

---

# Project Goal

This project is intentionally designed as a hands-on learning environment for understanding how production-grade AI agents should be engineered.

The objective is not to build a clinically deployable prescribing system.

Instead, the project demonstrates the engineering patterns required to prevent an LLM from becoming the uncontrolled decision-maker.

The current clinical policy is **synthetic and educational only**.

It must not be interpreted as real clinical decision support.

---

# Core Philosophy

Traditional application:

```text
Input
  ↓
Business Logic
  ↓
Output

LLM application:

Input
  ↓
LLM
  ↓
Output

Production-grade agent:

Input
  ↓
Validation
  ↓
Safety Guardrails
  ↓
LLM
  ↓
Structured Output
  ↓
Trusted Data
  ↓
Deterministic Policy
  ↓
Human Approval
  ↓
Controlled Outcome

The LLM is therefore one component inside the system, not the system itself.

Architecture
                         Clinical Input
                              │
                              ▼
                     ┌──────────────────┐
                     │ Semantic         │
                     │ Guardrail        │
                     └────────┬─────────┘
                              │
                         ALLOWED?
                        /         \
                      NO           YES
                      │             │
                      ▼             ▼
                   BLOCKED    ┌───────────────┐
                              │ Prescription  │
                              │     Agent     │
                              └───────┬───────┘
                                      │
                                      ▼
                              Structured Output
                                      │
                                      ▼
                              Patient Lookup
                                      │
                                      ▼
                            Deterministic Policy
                                      │
                               ALLOW / BLOCK
                                  /       \
                               BLOCK      ALLOW
                                │           │
                                ▼           ▼
                             BLOCKED    Human Approval
                                            │
                                      APPROVE / REJECT
                                         /          \
                                       REJECT       APPROVE
                                         │             │
                                         ▼             ▼
                                      BLOCKED       COMPLETED
Technology Stack
Component	Technology
Language	Python 3.13
Package Manager	uv
LLM Agent	Pydantic AI
LLM Provider	Groq
Model	openai/gpt-oss-20b
Workflow Engine	LangGraph
Persistence	SQLite
Async SQLite	aiosqlite
Checkpointing	langgraph-checkpoint-sqlite
Validation	Pydantic
Testing	pytest
Async Testing	pytest-asyncio
Observability	Langfuse — next phase
Project Structure
clinical-agent-harness/
│
├── README.md
├── roadmap.md
├── pyproject.toml
├── uv.lock
│
├── src/
│   └── clinical_agent_harness/
│       │
│       ├── domain/
│       │   ├── prescription.py
│       │   ├── patient.py
│       │   ├── patient_repository.py
│       │   └── policy.py
│       │
│       ├── agent/
│       │   ├── prescription_agent.py
│       │   └── run_agent.py
│       │
│       ├── guardrails/
│       │   ├── models.py
│       │   ├── semantic.py
│       │   └── check.py
│       │
│       ├── policy/
│       │   └── renal.py
│       │
│       └── harness/
│           ├── inner.py
│           ├── models.py
│           ├── state.py
│           ├── graph.py
│           ├── retry.py
│           ├── errors.py
│           ├── checkpointer.py
│           └── circuit_breaker.py
│
└── tests/
    ├── test_prescription.py
    ├── test_semantic_guardrail.py
    ├── test_inner_harness.py
    ├── test_inner_harness_failures.py
    ├── test_missing_clinical_policy.py
    ├── test_renal_policy.py
    ├── test_patient_repository.py
    ├── test_trusted_patient_context.py
    ├── test_retry.py
    ├── test_graph.py
    ├── test_checkpoint.py
    ├── test_circuit_breaker.py
    │
    └── evals/
        ├── conftest.py
        └── test_workflow_evals.py
Domain Model

The core prescription model is strongly typed.

class ClinicalPrescription(BaseModel):
    patient_id: str
    medication: str
    dose_mg: float
    frequency_per_day: int
    rationale: str

The model enforces constraints such as:

Valid patient ID
Positive dosage
Bounded dosage
Valid frequency
Required rationale

This prevents arbitrary LLM output from entering the workflow.

LLM Boundary

The prescription agent is implemented with Pydantic AI.

Conceptually:

Clinical Text
     ↓
Pydantic AI
     ↓
LLM
     ↓
Structured Output
     ↓
ClinicalPrescription

The LLM is instructed not to fabricate:

Patient IDs
Medications
Dosages
Frequencies
Rationales

However, the system does not rely solely on those instructions.

The harness validates the result independently.

Semantic Guardrail

The system contains a semantic safety guardrail before the prescription agent executes.

It is designed to identify unsafe inputs such as:

Prompt injection
Attempts to bypass safety controls
Unsafe prescribing manipulation
Instructions attempting to override deterministic policy

Flow:

Input
  ↓
Semantic Safety Check
  ↓
Allowed?
 /    \
NO     YES
│       │
▼       ▼
BLOCK   Prescription Agent

A blocked input never reaches the prescription stage.

Trusted Clinical Context

Patient information is retrieved from a trusted repository.

Prescription
     │
     ▼
Patient ID
     │
     ▼
Trusted Repository
     │
     ▼
PatientClinicalContext

The workflow does not trust user-provided clinical values when authoritative system data is available.

For example:

User input:
eGFR = 90

Trusted repository:
eGFR = 25

Policy receives:
eGFR = 25

This establishes a clear trust boundary between:

UNTRUSTED INPUT

and:

TRUSTED SYSTEM DATA
Deterministic Policy

The project contains a deterministic renal policy.

Patient Context
      +
Prescription
      ↓
Deterministic Policy
      ↓
ALLOW / BLOCK

The policy does not ask the LLM to determine whether the prescription is safe.

This is intentional.

The LLM extracts and structures information.

The deterministic system applies the policy.

The current renal policy is a synthetic educational rule and is not medical guidance.

Workflow State Machine

The workflow uses LangGraph.

Current states:

RUNNING
BLOCKED
FAILED
AWAITING_APPROVAL
COMPLETED

High-level workflow:

Guardrail
    ↓
Prescription
    ↓
Patient Lookup
    ↓
Policy
    ↓
Human Approval
    ↓
Completed

Every potentially failing node has explicit terminal-state routing.

For example:

Patient Lookup
    │
    ├── Success → Policy
    │
    ├── Unknown Patient → BLOCKED → END
    │
    └── Failure → FAILED → END

This prevents workflow deadlocks and invalid transitions.

Retry Architecture

The harness implements bounded retries.

Current configuration:

Maximum attempts: 3
Timeout per attempt: 10 seconds

Only transient failures are retried.

Retryable:
    TimeoutError
    RetryableError

Non-retryable:
    Validation errors
    NonRetryableError
    CircuitOpenError

Architecture:

Circuit Breaker
       ↓
     Retry
       ↓
   Pydantic AI
       ↓
      LLM

Pydantic AI handles structured-output validation and its own model-output retries.

The outer harness is responsible for transient infrastructure/provider failures.

This prevents retry multiplication.

Timeout Protection

Every provider attempt is bounded.

Conceptually:

Provider Call
     │
     ├── succeeds → continue
     │
     └── timeout → retry

A permanently hanging provider cannot hold the workflow indefinitely.

Circuit Breaker

The harness contains a provider-level circuit breaker.

States:

CLOSED
   │
   │ repeated failures
   ▼
OPEN
   │
   │ recovery timeout
   ▼
HALF_OPEN
   │
   ├── success → CLOSED
   │
   └── failure → OPEN

Purpose:

Prevent continuously calling an unhealthy provider.

Retry protects an individual operation.

Circuit breaking protects the dependency.

Human-in-the-Loop

The workflow pauses before completion.

Policy
  ↓
Human Approval
  ↓
AWAITING_APPROVAL

The workflow can resume with:

APPROVED

or:

REJECTED

Result:

APPROVED → COMPLETED

REJECTED → BLOCKED

The human approval gate is therefore part of the workflow state machine rather than a UI-only concept.

Durable Checkpointing

LangGraph checkpointing is implemented using SQLite.

Components:

SQLite
aiosqlite
AsyncSqliteSaver
langgraph-checkpoint-sqlite

The workflow can:

Start
  ↓
Reach Approval
  ↓
Persist State
  ↓
Process Stops
  ↓
Process Restarts
  ↓
Graph Recreated
  ↓
Same thread_id
  ↓
Resume

This demonstrates durable workflow state rather than in-memory-only execution.

Behavioral Evaluations

The project currently contains:

16 behavioral evaluation scenarios

They cover:

Valid prescription
Prompt injection
Renal policy violation
Unknown patient
Patient identity mismatch
Human approval
Human rejection
Missing prescription
Non-retryable LLM failure
Transient LLM failure
Circuit breaker fail-fast behavior
Circuit breaker recovery
Transient guardrail failure
Guardrail failure isolation
Timeout retry behavior
Blocked semantic input isolation

The goal is to test behavior at the workflow level rather than merely testing individual functions.

Test Suite

Current status:

60 / 60 tests passing

Coverage includes:

✓ Domain validation
✓ Prescription model
✓ Semantic guardrail
✓ Inner harness
✓ Guardrail failures
✓ Patient repository
✓ Trusted patient context
✓ Deterministic renal policy
✓ Retry behavior
✓ Timeout behavior
✓ Circuit breaker
✓ LangGraph workflow
✓ Failure routing
✓ Human approval
✓ Human rejection
✓ SQLite checkpointing
✓ Durable resume
✓ Behavioral evaluations
Observability — Next Phase

The next major implementation step is to make the harness observable and measurable.

The planned architecture is:

Application
     ↓
Agent Harness
     ↓
Langfuse
     ↓
Traces
Spans
Generations
Metrics
Evaluations

Langfuse is intended to provide production-oriented LLM observability.

Target trace:

workflow_run
│
├── guardrail
│    └── LLM generation
│
├── prescription
│    └── LLM generation
│
├── patient_lookup
│
├── policy
│
└── approval

Important metrics will include:

workflow_success_rate
workflow_failure_rate
provider_failure_rate
retry_rate
timeout_rate
circuit_open_rate
guardrail_block_rate
policy_block_rate
human_rejection_rate
workflow_latency
LLM_latency
token_usage
structured_output_failures
Evaluation Quality — Planned

Behavioral evaluations currently answer:

Did the workflow behave correctly?

The next layer is:

How good was the AI behavior?

Potential evaluation dimensions:

Correctness
Safety
Faithfulness
Extraction quality
Tool-use correctness
Policy adherence
Regression detection

Target architecture:

Evaluation Dataset
       ↓
Agent
       ↓
Observed Result
       ↓
Evaluator
       ↓
Score
       ↓
Langfuse
Production Hardening — Planned

Future production hardening will cover:

Security
Prompt injection resistance
Input validation
PII protection
Trusted/untrusted data separation
Tool authorization
Reliability
Retry limits
Timeouts
Circuit breakers
Graceful failure
Durable state
Observability
Structured logging
Correlation IDs
Tracing
Metrics
Alerts
Operations
Health checks
Deployment strategy
Rollback
Monitoring
Incident investigation
Advanced Harness Topics

After observability, evaluation quality, and production hardening, the project will move into deeper agent reliability topics.

1. Idempotency

Prevent duplicated external actions during retries.

Request
  ↓
Idempotency Key
  ↓
External Operation
2. Concurrency

Understand race conditions and concurrent workflow execution.

Topics include:

Locks
Optimistic concurrency
State versioning
SQLite limitations
Concurrent workflow execution
3. Tool Isolation / Sandboxing

Tools should not have unrestricted capabilities.

Agent
  ↓
Tool Authorization
  ↓
Sandbox
  ↓
Allowed Operation

Topics include:

Tool allowlists
Argument validation
Capability-based permissions
Resource limits
Network isolation
File-system restrictions
4. Context-Window Management

Long-running agents can accumulate excessive context.

Raw Context
    ↓
Relevant Context
    ↓
Bounded Context
    ↓
LLM

Topics include:

Context pruning
Summarization
State compaction
Token budgets
Retrieval
Context prioritization
5. Semantic Injection Through Trusted Data

Trusted data can contain untrusted text.

For example:

Patient Record

"Ignore all previous instructions..."

If inserted directly into the model context, the model may interpret the text as an instruction.

Therefore:

Trusted Data
     ↓
Content Classification
     ↓
Data / Instruction Separation
     ↓
Controlled Context
     ↓
LLM

Key principle:

A trusted source does not mean every piece of text inside that source should be treated as a trusted instruction.

Current Architecture Maturity

The project has evolved through the following stages:

Stage 1
LLM call

↓

Stage 2
Structured LLM output

↓

Stage 3
Semantic guardrails

↓

Stage 4
Deterministic policy

↓

Stage 5
Trusted data boundary

↓

Stage 6
State-machine orchestration

↓

Stage 7
Retries + timeouts

↓

Stage 8
Circuit breaker

↓

Stage 9
Human approval

↓

Stage 10
Durable checkpointing

↓

Stage 11
Behavioral evaluations

↓

Stage 12
60 automated tests

↓

NEXT
Observability + Evaluation Quality
Engineering Principles Learned
1. The LLM is not the authority

The LLM proposes.

The harness validates.

The deterministic system decides.

2. Safety should be enforced structurally

Do not depend entirely on:

"Please don't do X."

Prefer:

LLM
 ↓
Validation
 ↓
Guardrail
 ↓
Trusted Data
 ↓
Deterministic Policy
 ↓
Human Gate
3. Trusted data must remain authoritative

Do not allow:

User input
    ↓
Override trusted clinical data

Instead:

User input
    ↓
Identifier
    ↓
Trusted lookup
    ↓
Authoritative context
4. Retries need boundaries

Retrying everything can make failures worse.

Correct approach:

Transient infrastructure failure
        ↓
Retry

Permanent failure
        ↓
Fail immediately
5. Circuit breakers protect dependencies

Retry asks:

Can this operation succeed if we try again?

Circuit breaker asks:

Should we keep calling this dependency at all?

6. Workflow state must be explicit

Instead of implicit control flow:

function calls

use explicit states:

RUNNING
BLOCKED
FAILED
AWAITING_APPROVAL
COMPLETED
7. Every failure path needs a destination

A workflow should never leave a failed state without knowing where to go.

Failure
  ↓
Explicit terminal state
  ↓
END
8. Persistence is part of correctness

If a human approval is lost during a process restart, the workflow is not truly durable.

Therefore:

Workflow State
     ↓
Checkpoint
     ↓
Persistent Storage
9. Tests should validate behavior

A unit test can prove:

function X returns Y

A behavioral evaluation proves:

Unsafe input
   ↓
System blocks it

For agent systems, both are important.

Current Project Status
Inner Harness                  COMPLETE
Structured Output              COMPLETE
Semantic Guardrails            COMPLETE
Outer Harness                  COMPLETE
State Machine                  COMPLETE
Trusted Data Boundary          COMPLETE
Deterministic Policy           COMPLETE
Retry                          COMPLETE
Timeout                        COMPLETE
Circuit Breaker                COMPLETE
Human-in-the-Loop              COMPLETE
SQLite Checkpointing           COMPLETE
Behavioral Evaluations        COMPLETE
Test Suite                     60 / 60 PASSING

Observability                  NEXT
Evaluation Quality             NEXT
Production Hardening           NEXT

Idempotency                    FUTURE
Concurrency                    FUTURE
Tool Isolation                FUTURE
Context Management             FUTURE
Semantic Data Injection        FUTURE
Recommended Next Steps

The implementation sequence is:

1. Observability
       ↓
2. Evaluation Quality
       ↓
3. Production Hardening
       ↓
4. Idempotency
       ↓
5. Concurrency
       ↓
6. Tool Isolation / Sandboxing
       ↓
7. Context-Window Management
       ↓
8. Semantic Injection Through Trusted Data

The immediate next goal is therefore:

Make the harness observable and measurable.

Important Disclaimer

This repository is an engineering learning project.

The clinical policy implemented here is synthetic and intentionally simplified.

Nothing in this repository should be used as:

Medical advice
Clinical decision support
A prescribing recommendation
A replacement for clinician judgment
A production healthcare safety policy

The purpose is to learn AI Agent Harness Engineering patterns.

Guiding Principle

The most important architectural principle in this project is:

Do not make the LLM safer by simply asking it to behave better. Make the system safer by designing the harness so that unsafe behavior cannot easily become an unsafe outcome.

The LLM reasons.

The harness controls.

The deterministic system decides.

The human approves.

Observability tells us what happened.

Evaluations tell us whether it worked.

That is the foundation of production-grade Agent Engineering.