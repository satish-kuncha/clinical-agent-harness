from langfuse import get_client
from dotenv import load_dotenv
from pydantic_ai import Agent
from langfuse.langchain import CallbackHandler


load_dotenv()

langfuse = get_client()


def initialize_instrumentation() -> None:
    Agent.instrument_all()


def create_langgraph_handler() -> CallbackHandler:
    """Create a Langfuse callback handler for LangGraph."""
    return CallbackHandler()

def verify_connection() -> bool:
    return langfuse.auth_check()