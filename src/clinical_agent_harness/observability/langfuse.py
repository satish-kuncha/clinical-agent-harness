from typing import Optional

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langfuse.types import (
    MaskOtelSpansParams,
    MaskOtelSpansResult,
    OtelSpanPatch,
)
from pydantic_ai import Agent


load_dotenv()


def mask_otel_spans(
    *,
    params: MaskOtelSpansParams,
) -> Optional[MaskOtelSpansResult]:
    """Remove sensitive LLM content before telemetry is exported."""

    patches = {}

    for identifier, span in params.spans.items():
        attributes_to_delete = []

        for attribute_name in span.attributes:
            normalized_name = attribute_name.lower()

            if any(
                term in normalized_name
                for term in (
                    "prompt",
                    "completion",
                    "input",
                    "output",
                )
            ):
                attributes_to_delete.append(attribute_name)

        if attributes_to_delete:
            patches[identifier] = OtelSpanPatch(
                delete_attributes=tuple(attributes_to_delete),
                set_attributes={
                    "clinical_ai.masking_applied": True,
                },
            )

    if not patches:
        return None

    return MaskOtelSpansResult(
        span_patches=patches,
    )


langfuse = Langfuse(
    mask_otel_spans=mask_otel_spans,
)


def initialize_instrumentation() -> None:
    Agent.instrument_all()


def create_langgraph_handler() -> CallbackHandler:
    """Create a Langfuse callback handler for LangGraph."""
    return CallbackHandler()


def verify_connection() -> bool:
    return langfuse.auth_check()