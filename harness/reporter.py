from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from harness.dataset import EvalCase
from harness.runner import RunResult
from harness.scorer import ScoreResult


DB_PATH = Path(__file__).parent.parent.parent / "data" / "runs.db"

@dataclass
class RunRecord:
    """
    A single flattened record combining EvalCase metadata, RunResult,
    and ScoreResult into one object ready to write to SQLLite.  One
    RunRecord is created per case per batch run, run_id ties all records
    from the same batch together, run_at is the UTC timestamp of the batch run.
    """
    run_id: str
    ran_at: str
    case_id: str
    tags: list[str]
    passed: bool
    tool_match: bool
    output_match: bool
    constraints_passed: bool
    violations: list[str]
    judge_reasoning: str | None
    actual_output: str
    actual_tools: list[str]
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    error: str | None
    turn_count: int


class Reporter:
    """
    Persists batch run results to SQLite and prints a structured report.
    Each call to report() generates a new run_id that ties all records
    from that batch together for regression comparison.
    """
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """
        Creates the runs table if it doesn't exist.
        Called on every instantiation, safe to call repeatedly
        because we use 'IF NOT EXISTS'.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id              TEXT,
                    ran_at              TEXT,
                    case_id             TEXT,
                    tags                TEXT,
                    passed              INTEGER,
                    tool_match          INTEGER,
                    output_match        INTEGER,
                    constraints_ok      INTEGER,
                    violations          TEXT,
                    judge_reasoning     TEXT,
                    actual_output       TEXT,
                    actual_tools        TEXT,
                    prompt_tokens       INTEGER,
                    completion_tokens   INTEGER,
                    latency_ms          REAL,
                    error               TEXT,
                    turn_count          INTEGER
                )
        """)
            
    def _make_record(
        self,
        run_id: str,
        ran_at: str,
        case: EvalCase,
        run: RunResult,
        score: ScoreResult
    ) -> RunRecord:
        """
        Combines EvalCase and RunResult, and ScoreResult into a single
        flat RunRecord ready to insert into SQLite.
        Called once per case per batch run.
        """
        return RunRecord(
            run_id=run_id,
            ran_at=ran_at,
            case_id=case.id,
            tags=case.tags,
            passed=score.passed,
            tool_match=score.tool_match,
            output_match=score.output_match,
            constraints_passed=score.constraints_passed,
            violations=score.violations,
            judge_reasoning=score.judge_reasoning,
            actual_output=run.actual_output,
            actual_tools=run.actual_tools,
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
            latency_ms=run.latency_ms,
            error=run.error,
            turn_count=run.turn_count,
        )

    def _insert_records(self, records: list[RunRecord]) -> None:
        """
        Writes a list of RunRecords to SQLite in a single transaction.
        Lists are JSON-serialized before storage since SQLite has no
        native list type.  All records from one batch are committed
        together.  If any insert fails, the whole batch rolls back.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO runs VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    (
                        r.run_id,
                        r.ran_at,
                        r.case_id,
                        json.dumps(r.tags),
                        int(r.passed),
                        int(r.tool_match),
                        int(r.output_match),
                        int(r.constraints_passed),
                        json.dumps(r.violations),
                        r.judge_reasoning,
                        r.actual_output,
                        json.dumps(r.actual_tools),
                        r.prompt_tokens,
                        r.completion_tokens,
                        r.latency_ms,
                        r.error,
                        r.turn_count,
                    )
                    for r in records
                ],
            )

    async def report(
        self,
        cases: list[EvalCase],
        runs: list[RunResult],
        scores: list[ScoreResult],
    ) -> None:
        """
        Main entry point.  Takes parallel lists of cases, runs, and scores
        from a completed batch run.  Generates a run_id and timestamp,
        builds RunRecords, writes to SQLite, then prints the report.
        The three lists must be in the same order, index N of each
        list corresponds to the same test case.
        """
        run_id = str(uuid.uuid4())[:8]
        ran_at = datetime.now(timezone.utc).isoformat()

        records = [
            self._make_record(run_id, ran_at, case, run, score)
            for case, run, score in zip(cases, runs, scores)
        ]

        self._insert_records(records)
        self._print_report(run_id, ran_at, records)

    def _print_report(
        self, run_id: str, ran_at: str, records: list[RunRecord]
    ) -> None:
        """
        Prints a structured summary to stdout.  Covers four sections:
        run header, cost, and latency summary, regressions, and safety
        failures.  Regressions are cases that passed last run but
        failed this run - requires a previous run in the database compare.
        """
        passed = [ r for r in records if r.passed]
        failed = [r for r in records if not r.passed]
        total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in records)
        avg_tokens = total_tokens // len(records) if records else 0
        avg_latency = sum(r.latency_ms for r in records) / len(records) if records else 0
        max_latency = max(r.latency_ms for r in records) if records else 0

        print(f"\n{'-' * 60}")
        print(f"Run ID  : {run_id}")
        print(f"Ran at  : {ran_at}")
        print(f"Cases   : {len(records)} total | {len(passed)} passed | {len(failed)} failed")
        print(f"Tokens  : total={total_tokens:,} avg={avg_tokens}/case")
        print(f"Latency : avg={avg_latency:.0f}ms max={max_latency:.0f}ms")

        regressions = self._find_regressions(run_id, records)
        if regressions:
            print(f"\nREGRESSIONS ({len(regressions)} case(s) passed last run, failing now:")
            for r in regressions:
                print(f"❌    {r.case_id}")
                if r.judge_reasoning:
                    print(f"    judge: {r.judge_reasoning}")
        else:
            print(f"\nNo regressions detected.")

        safety_failures = [
            r for r in failed
            if any(t in ["safety", "adversarial"] for t in r.tags)
        ]
        if safety_failures:
            print(f"\nSAFETY FAILURES {len(safety_failures)}):")
            for r in safety_failures:
                print(f"    ❌ {r.case_id} | tags={r.tags}")
                print(f"    output: {r.actual_output[:120]}...")
                if r.judge_reasoning:
                    print(f"    judge: {r.judge_reasoning}")
                if r.violations:
                    for v in r.violations:
                        print(f"    -> {v}")
        print(f"{'-' * 60}\n")

    def _find_regressions(
        self, current_run_id: str, records: list[RunRecord]
    ) -> list[RunRecord]:
        """
        Queries SQLite for the most recent previous run and compares
        pass/fail status for each case.  A regression is a case that
        passed in the previous run but is failing in the current run.
        Returns and empty list if no previous run exists.
        """
        with sqlite3.connect(self.db_path) as conn:
            prev = conn.execute("""
                SELECT case_id, passed FROM runs
                WHERE run_id = (
                    SELECT run_id FROM runs
                    WHERE run_id != ?
                    ORDER BY ran_at DESC
                    LIMIT 1
                )
        """, (current_run_id,)).fetchall()
            
        if not prev:
            return []
        
        prev_passed = {row[0] for row in prev if row[1]}
        return [
            r for r in records
            if not r.passed and r.case_id in prev_passed
        ]