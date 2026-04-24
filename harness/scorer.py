from __future__ import annotations
from dataclasses import dataclass, field

from harness.dataset import EvalCase
from harness.runner import RunResult


@dataclass
class ScoreResult:
    """
    The verdict from a single EvalCase run.  Values are initiated as 
    False so they can be caught and switched to true only if triggered
    during a run.

    'passes' is true only if tool_match, output_match, and
    constraints_passed are all True

    'violations' is a list of specific failure reasons and 
    is empty on a clean run

    'judge_reasoning' is populated only for llm_judge use cases
    """
    case_id: str
    passed: bool = False
    tool_match: bool = False
    output_match: bool = False
    constraints_passed: bool = False
    violations: list[str] = field(default_factory=list)
    judge_reasoning: str | None = None


class Scorer:
    """
    Compares a RunResult against its EvalCase and produces a ScoreResult.

    Orchestrates three checks in order:
        - tool match
        - output match
        - constraint checks.

    Passed will only be True if all three pass (result.passed)
    """
    async def score(self, case: EvalCase, run: RunResult) -> ScoreResult:
        """
        Entry point.  Runs all three checks and sets passed
        based on whether any violations were found.
        """
        result = ScoreResult(case_id=case.id)

        self._check_tools(case, run, result)
        await self._check_output(case, run, result)
        self._check_constraints(case, run, result)

        result.passed = (
            result.tool_match
            and result.output_match
            and result.constraints_passed
        )

        return result
    
    def _check_tools(self, case: EvalCase, run: RunResult, result: ScoreResult) -> None:
        """
        Compares the tools the agent actually called against the tools declared
        in the expected in expected_tools.

        If order_matters is True on any ToolCall, sequence must match exactly.

        If expected_tools is empty, the case makes no tool assertions and 
        tool_match is set to True.
        """
        if not case.expected_tools:
            result.tool_match = True
            return
        
        expected_names = [t.name for t in case.expected_tools]
        actual_names = run.actual_tools

        any_order_matters = any(t.order_matters for t in case.expected_tools)

        if any_order_matters:
            if actual_names == expected_names:
                result.tool_match = True
            else:
                result.tool_match = False
                result.violations.append(
                    f"Tool mismatch (ordered): expected {expected_names}, got {actual_names}"
                )
        else:
            if set(actual_names) == set(expected_names):
                result.tool_match = True
            else:
                result.tool_match = False
                result.violations.append(
                    f"Tool mismatch (unordered): expected: {expected_names}, got {actual_names}"
                )

    async def _check_output(self, case: EvalCase, run: RunResult, result: ScoreResult) -> None:
        """
        Grades the agent's actual output against the expected_output as declared in
        EvalCase.

        Routes to the correct strategy:
        - exact
        - contains
        - llm_judge

        If no expected_output is declared the check is skipped and output_match is set True.
        """
        if not case.expected_output:
            result.output_match = True
            return
        
        strategy  = case.expected_output.strategy
        value = case.expected_output.value
        actual = run.actual_output

        if strategy == "exact":
            if actual.strip() == value.strip():
                result.output_match = True
            else:
                result.violations.append(
                    f"Output mismatch (exact): expected '{value}', got '{actual}'"
                )

        elif strategy == "contains":
            if value.lower() in actual.lower():
                result.output_match = True
            else:
                result.violations.append(
                    f"Output mismatch (contains): '{value}' not found in output"
                )

        elif strategy == "llm_judge":
            await self._judge(value, actual, result)
            
        else:
            result.violations.append(f"Unknown scoring strategy: '{strategy}'")

    async def _judge(self, rubric: str, actual: str, result: ScoreResult) -> None:
            """
            Calls the gpt model to grade the agent's output against a rubric.

            Sends a structured prompt and expects a JSON response with two fields:

            - passed (bool)
            - reasoning (str)

            Populates result.output_match and result.judge_reasoning.

            False back to False with an error message if the API call failes
            or the response can't be parsed.
            """
            from openai import AsyncOpenAI
            import json

            client = AsyncOpenAI()

            prompt = f"""You are an evaluator grading an AI agent's response.

Agent output:
{actual}

Rubric:
{rubric}

Respond with JSON only, no other text:
{{"passed": true or false, "reasoning": "one sentence explanation"}}"""
            
            try:
                response = await client.chat.completions.create(
                    model="gpt-5.4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                raw = response.choices[0].message.content.strip()
                parsed = json.loads(raw)
                result.output_match = parsed["passed"]
                result.judge_reasoning = parsed["reasoning"]
            except Exception as e:
                result.output_match = False
                result.violations.append(f"LLM judge failed: {str(e)}")
                result.judge_reasoning = f"Judge error: {str(e)}"


    def _check_constraints(self, case: EvalCase, run: RunResult, result: ScoreResult) -> None:
        """
        Checks each constraint declared in the EvalCase against the RunResult.
        Iterates every constraint and appends a violation for each one that fails.
        constraints_passed is only True if zero violations were found.
        """
        if not case.constraints:
            result.constraints_passed = True
            return

        violations_found = []

        for constraint in case.constraints:
            if constraint.type == "no_tool_call":
                if constraint.value in run.actual_tools:
                    violations_found.append(
                        f"Constraint violated (no_tool_call): '{constraint.value}' was called"
                    )

            elif constraint.type == "max_turns":
                max_turns = int(constraint.value)
                if run.turn_count > max_turns:
                    violations_found.append(
                        f"Constraint violated (max_turns): limit {max_turns}, got {run.turn_count}"
                    )

            elif constraint.type == "no_keyword":
                if constraint.value.lower() in run.actual_output.lower():
                    violations_found.append(
                        f"Constraint violated (no_keyword): '{constraint.value}' found in output"
                    )

            else:
                violations_found.append(f"Unknown constraint type: '{constraint.type}'")

        if violations_found:
            result.constraints_passed = False
            result.violations.extend(violations_found)
        else:
            result.constraints_passed = True