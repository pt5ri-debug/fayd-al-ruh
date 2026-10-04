from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# JSON Schema handed to the model so it is forced into a single structured shape.
MODEL_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["concepts", "evidence", "hypotheses", "questions", "plans"],
    "properties": {
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["label"],
                "properties": {
                    "label": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["description", "source"],
                "properties": {
                    "description": {"type": "string"},
                    "source": {"type": "string"},
                    "supports": {"type": "boolean"},
                    "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
        },
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["statement"],
                "properties": {
                    "statement": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
        },
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text"],
                "properties": {
                    "text": {"type": "string"},
                    "topic": {"type": "string"},
                    "priority": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
        },
        "plans": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["goal"],
                "properties": {
                    "goal": {"type": "string"},
                    "priority": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["description"],
                            "properties": {
                                "description": {"type": "string"},
                                "depends_on": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                    },
                },
            },
        },
    },
}

# Keys the model must never emit: it may only propose, never act or decide truth.
FORBIDDEN_KEYS: frozenset[str] = frozenset({
    "beliefs",
    "belief",
    "belief_updates",
    "tool",
    "tools",
    "tool_call",
    "tool_calls",
    "function_call",
    "function_calls",
    "actions",
    "execute",
    "execution",
    "commands",
    "shell",
    "command",
    "set_confidence",
    "set_status",
    "confidence_update",
})


class StructuredOutputError(ValueError):
    """Raised when a model response cannot be parsed into the schema."""


@dataclass(frozen=True)
class ConceptProposal:
    label: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceProposal:
    description: str
    source: str
    supports: bool = True
    strength: float = 0.5


@dataclass(frozen=True)
class HypothesisProposal:
    statement: str
    confidence: float = 0.5


@dataclass(frozen=True)
class QuestionProposal:
    text: str
    topic: str = ""
    priority: float = 0.5


@dataclass(frozen=True)
class PlanStepProposal:
    description: str
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanProposal:
    goal: str
    steps: tuple[PlanStepProposal, ...] = ()
    priority: float = 0.5


@dataclass
class StructuredProposal:
    """A validated, side-effect-free set of things the model proposed.

    Proposals are inert data. Registering them, mutating beliefs, or running
    anything is the job of the validation layer, never the model.
    """

    concepts: list[ConceptProposal] = field(default_factory=list)
    evidence: list[EvidenceProposal] = field(default_factory=list)
    hypotheses: list[HypothesisProposal] = field(default_factory=list)
    questions: list[QuestionProposal] = field(default_factory=list)
    plans: list[PlanProposal] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.concepts or self.evidence or self.hypotheses or self.questions or self.plans)


def _extract_json(text: str) -> Any:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as error:
            raise StructuredOutputError(f"Model output is not valid JSON: {error}") from error
    raise StructuredOutputError("Model output does not contain a JSON object.")


def _reject_forbidden(value: Any, path: str = "") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in FORBIDDEN_KEYS:
                raise StructuredOutputError(
                    f"Model output contains forbidden key '{key}' at {path or 'root'}."
                )
            _reject_forbidden(nested, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_forbidden(nested, f"{path}[{index}]")


def _require_list(data: dict, key: str) -> list:
    value = data.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise StructuredOutputError(f"Field '{key}' must be a list.")
    return value


def _require_object(item: Any, key: str) -> dict:
    if not isinstance(item, dict):
        raise StructuredOutputError(f"Each '{key}' entry must be an object.")
    return item


def _clamp(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StructuredOutputError("Confidence values must be numbers.")
    return min(1.0, max(0.0, float(value)))


def parse_structured_output(text: str) -> StructuredProposal:
    """Parse a model response into an inert :class:`StructuredProposal`.

    The parser is strict: it rejects non-JSON output, wrong field types, and
    any forbidden key (beliefs, tools, actions). Missing sections default to
    empty, and confidence values are clamped into ``[0, 1]``.
    """
    data = _extract_json(text)
    if not isinstance(data, dict):
        raise StructuredOutputError("Model output must be a JSON object.")
    _reject_forbidden(data)

    concepts: list[ConceptProposal] = []
    for item in _require_list(data, "concepts"):
        entry = _require_object(item, "concepts")
        label = entry.get("label")
        if not isinstance(label, str) or not label.strip():
            raise StructuredOutputError("Concept label must be a non-empty string.")
        aliases = entry.get("aliases", [])
        if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
            raise StructuredOutputError("Concept aliases must be a list of strings.")
        concepts.append(ConceptProposal(label.strip(), tuple(alias.strip() for alias in aliases if alias.strip())))

    evidence: list[EvidenceProposal] = []
    for item in _require_list(data, "evidence"):
        entry = _require_object(item, "evidence")
        description, source = entry.get("description"), entry.get("source")
        if not isinstance(description, str) or not description.strip():
            raise StructuredOutputError("Evidence description must be a non-empty string.")
        if not isinstance(source, str) or not source.strip():
            raise StructuredOutputError("Evidence source must be a non-empty string.")
        supports = entry.get("supports", True)
        if not isinstance(supports, bool):
            raise StructuredOutputError("Evidence 'supports' must be a boolean.")
        evidence.append(EvidenceProposal(description.strip(), source.strip(), supports, _clamp(entry.get("strength"), 0.5)))

    hypotheses: list[HypothesisProposal] = []
    for item in _require_list(data, "hypotheses"):
        entry = _require_object(item, "hypotheses")
        statement = entry.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            raise StructuredOutputError("Hypothesis statement must be a non-empty string.")
        hypotheses.append(HypothesisProposal(statement.strip(), _clamp(entry.get("confidence"), 0.5)))

    questions: list[QuestionProposal] = []
    for item in _require_list(data, "questions"):
        entry = _require_object(item, "questions")
        text_value = entry.get("text")
        if not isinstance(text_value, str) or not text_value.strip():
            raise StructuredOutputError("Question text must be a non-empty string.")
        topic = entry.get("topic", "")
        if not isinstance(topic, str):
            raise StructuredOutputError("Question topic must be a string.")
        questions.append(QuestionProposal(text_value.strip(), topic.strip(), _clamp(entry.get("priority"), 0.5)))

    plans: list[PlanProposal] = []
    for item in _require_list(data, "plans"):
        entry = _require_object(item, "plans")
        goal = entry.get("goal")
        if not isinstance(goal, str) or not goal.strip():
            raise StructuredOutputError("Plan goal must be a non-empty string.")
        steps: list[PlanStepProposal] = []
        raw_steps = entry.get("steps", [])
        if not isinstance(raw_steps, list):
            raise StructuredOutputError("Plan steps must be a list.")
        for step in raw_steps:
            step_entry = _require_object(step, "plans.steps")
            description = step_entry.get("description")
            if not isinstance(description, str) or not description.strip():
                raise StructuredOutputError("Plan step description must be a non-empty string.")
            depends_on = step_entry.get("depends_on", [])
            if not isinstance(depends_on, list) or any(not isinstance(dep, str) for dep in depends_on):
                raise StructuredOutputError("Plan step dependencies must be a list of strings.")
            steps.append(PlanStepProposal(description.strip(), tuple(dep.strip() for dep in depends_on if dep.strip())))
        plans.append(PlanProposal(goal.strip(), tuple(steps), _clamp(entry.get("priority"), 0.5)))

    return StructuredProposal(concepts, evidence, hypotheses, questions, plans)
