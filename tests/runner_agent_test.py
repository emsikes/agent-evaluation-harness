import os
from dotenv import load_dotenv
from agents import Agent
import asyncio
from pathlib import Path

from harness.runner import AgentRunner
from harness.dataset import EvalDataset, EvalCase
from harness.scorer import Scorer
from harness.reporter import Reporter

load_dotenv(override=True)

# DATASET_PATH = Path(__file__).parent.parent / "datasets" / "support_triage.yaml"
# DATASET_PATH = Path(__file__).parent.parent / "datasets" / "adversarial.yaml"
DATASET_PATH = Path(__file__).parent.parent / "datasets" / "known_good.yaml"

# Define two sub tools the agent can call
from agents import function_tool


@function_tool
def search_kb(query: str) -> str:
    """Search the support knowledge base for relevant articles."""
    return f"Found KB article matching '{query}': our support team can assist with order status, \
        refunds, shipping delays, and account issues. Refund policy allows returns within 30 days."

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
    instructions="""You are a support triage agent.
You MUST use the search_kb tool to look up information before responding to any customer question about orders, refunds, shipping, or account issues.
Never answer from memory — always search the knowledge base first.
Only escalate using escalate_ticket when the customer's issue cannot be resolved through the knowledge base.
For greetings with no specific question, you may respond directly without tools.""",
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
    scorer = Scorer()
    reporter = Reporter()

    cases = []
    runs = []
    scores = []

    for case in dataset:
        run = await runner.run(case)
        score = await scorer.score(case, run)

        cases.append(case)
        runs.append(run)
        scores.append(score)

        status = "❌" if not score.passed else "✅"
        print(f"{status} {run.case_id} | passed={score.passed} \
              | tools={run.actual_tools} | tokens={run.prompt_tokens + run.completion_tokens} \
              | latency={run.latency_ms:.0f}ms | error={run.error}")
        
        if score.violations:
            for v in score.violations:
                print(f"    -> {v}")
        if score.judge_reasoning:
            print(f"    -> judge: {score.judge_reasoning}")

    await reporter.report(cases, runs, scores)

# asyncio.run(main())

asyncio.run(run_dataset())