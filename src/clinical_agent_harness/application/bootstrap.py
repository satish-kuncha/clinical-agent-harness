from clinical_agent_harness.observability.langfuse import (
    initialize_instrumentation,
)


def initialize_application() -> None:
    initialize_instrumentation()