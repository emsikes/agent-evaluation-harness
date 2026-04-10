# harness/dataset.py

import yaml
from pathlib import Path
from pydantic import BaseModel, Field
"""
model_dump()→ plain Python dict
model_dump_json()→ JSON string
model_validate(dict)→ creates instance from a dict (used by the YAML loader)
model_fields→ introspect what fields the model has
"""


class ToolCall(BaseModel):
    name: str
    order_matters: bool = True


class ExpectedOutput(BaseModel):
    """
    Declares what the agent's response should satisfy.
    strategy: how to evaluate — 'contains', 'exact', or 'llm_judge'
    value: the string to match, or the rubric text for the LLM judge
    """
    strategy: str
    value: str


class Constraint(BaseModel):
    """
    A machine-checkable boundary the agent must not violate.
    type: what kind of constraint — 'no_tool_call', 'max_turns', 'no_keyword'
    value: the tool name, turn count, or keyword string to enforce
    """
    type: str
    value: str


class EvalCase(BaseModel):
    """
    Test case model.
    - id will be the constant key for tracking regressions across runs
    - description is the human intent
    - input is what is sent to the agent
    """
    id: str
    description: str
    input: str
    expected_tools: list[ToolCall] = Field(default_factory=list)
    expected_output: ExpectedOutput | None = None
    constraints: list[Constraint] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class EvalDataset:
    """
    Loads a YAML file and validates each entry as an EvalCase.
    Raises on the first malformed case so bad data never reaches the runner.
    """
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.cases: list[EvalCase] = self._load()

    def _load(self) -> list[EvalCase]:
        """
        Reads the YAML file, iterates each entry, and calls EvalCase.model_validate()
        on it.  Pydantic then validates structure and types, rasing ValidationError if
        anything is wrong.
        """
        raw = yaml.safe_load(self.path.read_text())
        return [EvalCase.model_validate(entry) for entry in raw]
    
    def __len__(self):
        return len(self.cases)
    
    def __iter__(self):
        return iter(self.cases)


