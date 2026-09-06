import asyncio

from clinical_agent_harness.agent.prescription_agent import prescription_agent


async def main() -> None:
    clinical_text = """
    Patient PAT-12345 should receive metformin -5000 mg
    99 times per day for treatment of diabetes.
    """

    result = await prescription_agent.run(clinical_text)

    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())