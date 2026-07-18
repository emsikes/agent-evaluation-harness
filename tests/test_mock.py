# tests/test_mock.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from harness.dataset import EvalCase, ToolCall, ExpectedOutput, Constraint
from harness.runner import RunResult
from harness.scorer import Scorer


async def score(case: EvalCase, run: RunResult):
    """Helper — runs scorer and returns ScoreResult."""
    scorer = Scorer()
    return await scorer.score(case, run)


# ── Crash detection ────────────────────────────────────────────────────────────

async def test_crashed_run_fails():
    """A run with error set must never score as passed."""
    case = EvalCase(
        id="MOCK-001",
        description="Crashed run should fail",
        input="anything"
    )
    run = RunResult(case_id="MOCK-001", error="API timeout")
    result = await score(case, run)
    assert not result.passed
    assert any("Runner error" in v for v in result.violations)


# ── Tool checking ──────────────────────────────────────────────────────────────

async def test_correct_tool_passes():
    """Agent calling the expected tool should pass tool_match."""
    case = EvalCase(
        id="MOCK-002",
        description="Correct tool call",
        input="check order status",
        expected_tools=[ToolCall(name="search_kb", order_matters=False)]
    )
    run = RunResult(case_id="MOCK-002", actual_tools=["search_kb"], actual_output="Found result.")
    result = await score(case, run)
    assert result.tool_match


async def test_wrong_tool_fails():
    """Agent calling wrong tool should fail tool_match with a violation."""
    case = EvalCase(
        id="MOCK-003",
        description="Wrong tool call",
        input="check order status",
        expected_tools=[ToolCall(name="search_kb", order_matters=False)]
    )
    run = RunResult(case_id="MOCK-003", actual_tools=["escalate_ticket"], actual_output="Escalated.")
    result = await score(case, run)
    assert not result.tool_match
    assert any("Tool mismatch" in v for v in result.violations)


async def test_ordered_tools_wrong_order_fails():
    """Tools called in wrong order should fail when order_matters=True."""
    case = EvalCase(
        id="MOCK-004",
        description="Wrong tool order",
        input="lookup then escalate",
        expected_tools=[
            ToolCall(name="search_kb", order_matters=True),
            ToolCall(name="escalate_ticket", order_matters=True),
        ]
    )
    run = RunResult(
        case_id="MOCK-004",
        actual_tools=["escalate_ticket", "search_kb"],
        actual_output="Done."
    )
    result = await score(case, run)
    assert not result.tool_match


# ── Output checking ────────────────────────────────────────────────────────────

async def test_contains_pass():
    """Output containing expected substring should pass."""
    case = EvalCase(
        id="MOCK-005",
        description="Contains check pass",
        input="refund policy",
        expected_output=ExpectedOutput(strategy="contains", value="30 days")
    )
    run = RunResult(case_id="MOCK-005", actual_output="Refunds are valid within 30 days.")
    result = await score(case, run)
    assert result.output_match


async def test_contains_fail():
    """Output missing expected substring should fail."""
    case = EvalCase(
        id="MOCK-006",
        description="Contains check fail",
        input="refund policy",
        expected_output=ExpectedOutput(strategy="contains", value="30 days")
    )
    run = RunResult(case_id="MOCK-006", actual_output="Please contact support.")
    result = await score(case, run)
    assert not result.output_match
    assert any("contains" in v for v in result.violations)


async def test_exact_pass():
    """Output matching exactly should pass."""
    case = EvalCase(
        id="MOCK-007",
        description="Exact match pass",
        input="status code",
        expected_output=ExpectedOutput(strategy="exact", value="ESCALATE")
    )
    run = RunResult(case_id="MOCK-007", actual_output="ESCALATE")
    result = await score(case, run)
    assert result.output_match


async def test_exact_fail():
    """Output not matching exactly should fail."""
    case = EvalCase(
        id="MOCK-008",
        description="Exact match fail",
        input="status code",
        expected_output=ExpectedOutput(strategy="exact", value="ESCALATE")
    )
    run = RunResult(case_id="MOCK-008", actual_output="I will escalate this.")
    result = await score(case, run)
    assert not result.output_match


# ── Constraint checking ────────────────────────────────────────────────────────

async def test_no_tool_call_constraint_passes():
    """Agent not calling forbidden tool should pass constraint."""
    case = EvalCase(
        id="MOCK-009",
        description="no_tool_call pass",
        input="delete request",
        constraints=[Constraint(type="no_tool_call", value="delete_record")]
    )
    run = RunResult(case_id="MOCK-009", actual_tools=["search_kb"], actual_output="Found info.")
    result = await score(case, run)
    assert result.constraints_passed


async def test_no_tool_call_constraint_fails():
    """Agent calling forbidden tool should fail constraint with violation."""
    case = EvalCase(
        id="MOCK-010",
        description="no_tool_call fail",
        input="delete request",
        constraints=[Constraint(type="no_tool_call", value="delete_record")]
    )
    run = RunResult(case_id="MOCK-010", actual_tools=["delete_record"], actual_output="Deleted.")
    result = await score(case, run)
    assert not result.constraints_passed
    assert any("no_tool_call" in v for v in result.violations)


async def test_max_turns_constraint_passes():
    """Agent completing within turn limit should pass."""
    case = EvalCase(
        id="MOCK-011",
        description="max_turns pass",
        input="simple question",
        constraints=[Constraint(type="max_turns", value="2")]
    )
    run = RunResult(case_id="MOCK-011", actual_output="Done.", turn_count=1)
    result = await score(case, run)
    assert result.constraints_passed


async def test_max_turns_constraint_fails():
    """Agent exceeding turn limit should fail with violation."""
    case = EvalCase(
        id="MOCK-012",
        description="max_turns fail",
        input="simple question",
        constraints=[Constraint(type="max_turns", value="2")]
    )
    run = RunResult(case_id="MOCK-012", actual_output="Done.", turn_count=5)
    result = await score(case, run)
    assert not result.constraints_passed
    assert any("max_turns" in v for v in result.violations)


async def test_no_keyword_constraint_passes():
    """Output without forbidden keyword should pass."""
    case = EvalCase(
        id="MOCK-013",
        description="no_keyword pass",
        input="file request",
        constraints=[Constraint(type="no_keyword", value="filepath")]
    )
    run = RunResult(case_id="MOCK-013", actual_output="I cannot share that information.")
    result = await score(case, run)
    assert result.constraints_passed


async def test_no_keyword_constraint_fails():
    """Output containing forbidden keyword should fail."""
    case = EvalCase(
        id="MOCK-014",
        description="no_keyword fail",
        input="file request",
        constraints=[Constraint(type="no_keyword", value="filepath")]
    )
    run = RunResult(case_id="MOCK-014", actual_output="The filepath is /etc/secrets.")
    result = await score(case, run)
    assert not result.constraints_passed
    assert any("no_keyword" in v for v in result.violations)

async def test_allowed_tools_passes():
    """Agent calling only permitted tools should pass."""
    case = EvalCase(
        id="MOCK-016",
        description="allowed_tools pass",
        input="order status",
        constraints=[Constraint(type="allowed_tools", value='["search_kb"]')]
    )
    run = RunResult(case_id="MOCK-016", actual_tools=["search_kb"], actual_output="Found result.")
    result = await score(case, run)
    assert result.constraints_passed


async def test_allowed_tools_fails():
    """Agent calling tool outside allowlist should fail with violation."""
    case = EvalCase(
        id="MOCK-017",
        description="allowed_tools fail",
        input="order status",
        constraints=[Constraint(type="allowed_tools", value='["search_kb"]')]
    )
    run = RunResult(case_id="MOCK-017", actual_tools=["search_kb", "delete_record"], actual_output="Done.")
    result = await score(case, run)
    assert not result.constraints_passed
    assert any("allowed_tools" in v for v in result.violations)


# ── Overall passed logic ───────────────────────────────────────────────────────

async def test_passed_requires_all_three():
    """passed=True only when tool_match, output_match, and constraints all pass."""
    case = EvalCase(
        id="MOCK-015",
        description="All three must pass",
        input="full check",
        expected_tools=[ToolCall(name="search_kb", order_matters=False)],
        expected_output=ExpectedOutput(strategy="contains", value="policy"),
        constraints=[Constraint(type="no_tool_call", value="delete_record")]
    )
    run = RunResult(
        case_id="MOCK-015",
        actual_tools=["search_kb"],
        actual_output="Our refund policy covers 30 days.",
        turn_count=1
    )
    result = await score(case, run)
    assert result.passed
    assert result.tool_match
    assert result.output_match
    assert result.constraints_passed