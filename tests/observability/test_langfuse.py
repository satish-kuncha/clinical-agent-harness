from clinical_agent_harness.observability.langfuse import langfuse


def test_langfuse_client_is_initialized():
    assert langfuse is not None