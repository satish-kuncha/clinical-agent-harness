import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


async def create_checkpointer(
    database_path: str = "clinical_agent_harness.db",
) -> AsyncSqliteSaver:
    connection = await aiosqlite.connect(database_path)

    checkpointer = AsyncSqliteSaver(connection)

    await checkpointer.setup()

    return checkpointer