# harness/runner.py

from __future__ import annotations

import time
from dataclasses import dataclass, field
import asyncio

from agents import Agent, Runner

from harness.dataset import EvalCase



@dataclass
class RunResult:
    """
    The recorded output of running a single EvalCase through the agent.
    This is what the scorer receives, not the raw agent output, but a
    structured trace of everything that happened during the run.
    """
    case_id: str
    actual_tools: list[str] = field(default_factory=list)
    actual_output: str = ""
    turn_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None


class AgentRunner:
    """
    Wraps the OpenAI Agents SDK and executes an agent against a single EvalCase.
    Captures the full trace - tools called, output, token usage, and latency.
    The agent is injected as a dependency so the runner works with both
    real and mock agents without changing any runner code.
    """

    def __init__(self, agent: Agent):
        self.agent = agent

    async def run(self, case: EvalCase) -> RunResult:
        """
        Executes the agent against a single EvalCase and returns RunResult.
        Wraps the entire run in a try/except so a crashed case produces a
        failed RunResult rather than killing the whole eval suite.
        """
        start = time.monotonic()
        result = RunResult(case_id=case.id)

        try:
            response = await Runner.run(self.agent, case.input)
            result.actual_output = response.final_output
            if response.raw_responses:
                usage = response.raw_responses[0].usage
                result.prompt_tokens = usage.input_tokens
                result.completion_tokens = usage.output_tokens

            result.actual_tools = [
                step.tool_name
                for step in response.new_items
                if hasattr(step, "tool_name")
            ]
            result.turn_count = len(response.new_items)

        except Exception as e:
            result.error = str(e)

        finally:
            result.latency_ms = (time.monotonic() - start) * 1000

        return result