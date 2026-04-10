# Agent Evaluation Harness

![Python](https://img.shields.io/badge/python-3.12-blue?style=flat-square&logo=python&logoColor=white)
![Pydantic](https://img.shields.io/badge/pydantic-v2-red?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/status-active--development-orange?style=flat-square)
![Layer](https://img.shields.io/badge/layer-1%20of%204%20complete-brightgreen?style=flat-square)

A structured evaluation harness for tool-calling AI agents. Not a chatbot demo — a test rig for answering five questions on every change:

1. Did the agent complete the task?
2. Did it choose the right tools?
3. Did it stay inside allowed boundaries?
4. Did cost and latency stay acceptable?
5. Did it resist unsafe or adversarial inputs?

---

## Architecture

The harness is organized into four layers, each independently testable:

```
Dataset → Runner → Scorer → Reporter
```

| Layer | Status | Purpose |
|---|---|---|
| 1. Dataset | ✅ Complete | Typed, validated test cases loaded from YAML |
| 2. Runner | 🔲 In Progress | Executes agent against each case, captures trace |
| 3. Scorer | 🔲 Planned | Grades behavior against expected outcomes |
| 4. Reporter | 🔲 Planned | Surfaces regressions, cost, latency, safety failures |

---

## Project Structure

```
agent-eval-harness/
├── code/
│   ├── harness/
│   │   ├── __init__.py
│   │   ├── dataset.py       # Layer 1: schema and loader
│   │   ├── runner.py        # Layer 2: agent execution
│   │   ├── scorer.py        # Layer 3: behavioral grading
│   │   └── reporter.py      # Layer 4: reporting and regression
│   └── datasets/
│       ├── support_triage.yaml
│       └── adversarial.yaml
├── tests.ipynb
├── pyproject.toml
└── venv/
```

---

## Quickstart

**Requirements:** Python 3.12, pip

```bash
git clone https://github.com/emsikes/agent-evaluation-harness.git
cd agent-evaluation-harness
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e .
```

---

## Layer 1: Dataset

Test cases are defined in YAML and validated against a Pydantic schema. Invalid cases raise before they reach the runner.

### Schema

```yaml
- id: TC-001
  description: Delayed order should reference refund policy
  input: My order has been delayed three weeks
  expected_tools:
    - name: search_kb
      order_matters: true
    - name: escalate_ticket
      order_matters: true
  expected_output:
    strategy: contains        # contains | exact | llm_judge
    value: "refund policy"
  constraints:
    - type: no_tool_call
      value: delete_record
    - type: max_turns
      value: "3"
  tags:
    - happy_path
```

### Scoring Strategies

| Strategy | How it works |
|---|---|
| `contains` | Agent output must contain the specified substring |
| `exact` | Agent output must match the value exactly |
| `llm_judge` | An LLM grades the output against a rubric in `value` |

### Constraint Types

| Type | Enforces |
|---|---|
| `no_tool_call` | Named tool must never be called |
| `max_turns` | Agent must complete within N turns |
| `no_keyword` | Named word must not appear in output |

### Loading a Dataset

```python
from harness.dataset import EvalDataset

dataset = EvalDataset("code/datasets/support_triage.yaml")

print(f"Loaded {len(dataset)} cases")
for case in dataset:
    print(f"  {case.id} — {case.tags}")
```

```
Loaded 2 cases
  TC-001 — ['happy_path']
  TC-004 — ['safety', 'adversarial']
```

---

## Planned: Layer 2 — Runner

The runner will execute the agent against each `EvalCase` using the OpenAI Agents SDK and return a structured trace per case:

```python
@dataclass
class RunResult:
    case_id: str
    actual_tools: list[str]
    actual_output: str
    turn_count: int
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    raw_trace: dict
```

Tool calls will be mockable for deterministic testing without live API calls.

---

## Planned: Layer 3 — Scorer

The scorer will compare `RunResult` against `EvalCase` expectations and produce a finding:

```python
@dataclass
class ScoreResult:
    case_id: str
    passed: bool
    tool_match: bool
    output_match: bool
    constraints_passed: bool
    violations: list[str]
    judge_reasoning: str | None
```

LLM-as-judge scoring will use GPT-5 or Claude to grade open-ended outputs against rubrics.

---

## Planned: Layer 4 — Reporter

The reporter will store run history in SQLite and surface:

- Pass/fail counts by tag (run only `["safety"]` cases in CI)
- Regression diffs vs previous runs
- Cost and latency trends over time
- Safety failure logs with full trace

---

## Roadmap

- [x] Layer 1: Dataset schema and YAML loader
- [ ] Layer 2: Runner with OpenAI Agents SDK
- [ ] Layer 3: Scorer (contains, exact, llm_judge)
- [ ] Layer 4: Reporter with SQLite run history
- [ ] pytest integration for CI-triggered regression runs
- [ ] `adversarial.yaml` reference dataset (prompt injection, jailbreak, tool misuse)

---

## Contributing

This project is in active development. Contributions are not open yet — watch the repo for updates.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Author

**Matt Sikes** — Principal Architect, AI/ML and Security  
[LinkedIn](https://linkedin.com/in/matt-sikes) · [The Inference Loop](https://theinferenceloop.substack.com)
