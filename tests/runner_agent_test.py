import os
from dotenv import load_dotenv
from agents import Agent
import asyncio
from pathlib import Path

from harness.runner import AgentRunner
from harness.dataset import EvalDataset, EvalCase


load_dotenv(override=True)

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "support_triage.yaml"

# Define two sub tools the agent can call
from agents import function_tool

@function_tool
def search_kb(query: str) -> str:
    """
    Search the support knowledge base for relevant articles.
    """
    return f"Found KB article: refund policy allows returns within 30 days."

@function_tool
def escalate_ticket(reason: str) -> str:
    """
    Escalate the support ticket to a human agent.
    """
    return f"Ticket escalated: {reason}"

# Build the agent
agent = Agent(
    name="SupportTriageAgent",
    model="gpt-5.4-mini",
    instructions="You are a support triage agent.  Use the tools available to help customers.",
    tools=[search_kb, escalate_ticket],
)

# Run TC-001 through the runner
case = EvalCase(
    id="TC-001",
    description="Delayed order should reference refund policy",
    input="My order has been delayed three weeks"
)

async def main():
    runner = AgentRunner(agent)
    result = await runner.run(case)
    print(result)

async def run_dataset():
    dataset = EvalDataset(
        DATASET_PATH
    )
    runner = AgentRunner(agent)

    for case in dataset:
        result = await runner.run(case)
        status = "✅" if not result.error else "❌"
        print(f"{status} {result.case_id} | tools={result.actual_tools} | tokens={result.prompt_tokens + result.completion_tokens} | latency={result.latency_ms:.0f}ms")

# asyncio.run(main())

asyncio.run(run_dataset())