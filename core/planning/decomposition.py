"""The task decomposition engine (Phase 8, section 5).

Section 5 asks for an engine that turns a complex goal into executable tasks,
each with its parent, preconditions, inputs, expected outputs, dependencies,
selected tool, resource estimate, permissions, success criteria, and failure
handling -- and that avoids excessive decomposition into trivial tasks.

This engine is **rule-driven, not model-driven**: it recognises the shape of a
goal from its own words (a read, a compute, a write, a search, a comparison, a
summary) and emits the tasks that shape needs, in dependency order, each bound to
a tool the catalogue permits. A goal it does not recognise yields a single
task that asks for the honest next step, never a fabricated plan. Decomposition
is **bounded** (:data:`MAX_TASKS`) and **shallow** (:data:`MAX_DEPTH`) so it
cannot explode, and it never emits a task with no tool.

The engine also detects the two blocking conditions section 5 names -- a task
that needs user approval, and a task blocked by missing information -- and
reports them rather than hiding them.

Dependency-free and Tk-free: importing it loads no model and touches no network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.planning.goals import Goal
from core.planning.tasks import Task
from core.planning.tools import RiskCategory, ToolCatalogue, select_tool

#: The most tasks one goal may decompose into -- bounded so a plan stays
#: inspectable and a runaway decomposition cannot occur (section 5's rule).
MAX_TASKS = 12
#: The deepest a subgoal may nest.
MAX_DEPTH = 2


class DecompositionError(ValueError):
    """Raised when a decomposition request is malformed (e.g. a depth over bound)."""


# -- goal shapes ------------------------------------------------------------

_READ = re.compile(r"\b(read|open|load|inspect|view|show|list|find|search|scan)\b", re.I)
#: A goal that asks for a directory listing rather than a named file's contents.
_LIST = re.compile(r"\b(list|listing|ls|dir|directory|folder|files|entries)\b", re.I)
_WRITE = re.compile(r"\b(write|create|save|generate|produce|build|make|export|report)\b", re.I)
_COMPUTE = re.compile(r"\b(compute|calculate|sum|count|average|mean|total|convert|evaluate)\b", re.I)
_SUMMARISE = re.compile(r"\b(summar|summary|analyse|analyze|review|explain|describe)\b", re.I)
_COMPARE = re.compile(r"\b(compare|diff|difference|versus|vs\.?|against)\b", re.I)

#: A short, declarative success line the engine can attach to a written artifact.
_ARTIFACT_HINT = re.compile(r"\b(?:to|into|as|named|called)\s+([\w./-]+\.[A-Za-z0-9]+)", re.I)


@dataclass
class DecompositionEngine:
    """Turn a goal into tasks, deterministically and within bounds."""

    catalogue: ToolCatalogue
    max_tasks: int = MAX_TASKS
    max_depth: int = MAX_DEPTH

    # ------------------------------------------------------------- public API

    def decompose(
        self,
        goal: Goal,
        *,
        depth: int = 0,
        known_facts: frozenset[str] = frozenset(),
    ) -> tuple[Task, ...]:
        """Decompose ``goal`` into tasks (never more than :data:`MAX_TASKS`).

        ``depth`` is how far a subgoal is nested; a depth over the bound is
        refused rather than silently truncated, so a caller cannot ask for an
        unbounded decomposition.
        """
        if depth > self.max_depth:
            raise DecompositionError(
                f"decomposition depth {depth} exceeds the bound of {self.max_depth}."
            )
        description = goal.description.strip()
        if not description:
            return ()

        tasks = self._tasks_for(goal, depth=depth)
        if len(tasks) > self.max_tasks:
            tasks = tasks[: self.max_tasks]
        return tuple(tasks)

    def notes_for(self, goal: Goal) -> tuple[str, ...]:
        """What the engine could **not** turn into a task, said honestly.

        A goal that asks to summarise or compare needs a language model, which is
        not a sandbox tool; the engine records that as a note rather than emitting
        a task that could not really do it.
        """
        text = goal.description
        notes: list[str] = []
        if _SUMMARISE.search(text):
            notes.append(
                "a summarise step would need a language model; the plan keeps only "
                "the steps the sandbox can really run."
            )
        if _COMPARE.search(text):
            notes.append(
                "a compare step would need a model or a dedicated diff tool; the plan "
                "keeps only the steps the sandbox can really run."
            )
        return tuple(notes)

    # ------------------------------------------------------------- internals

    def _tasks_for(self, goal: Goal, *, depth: int) -> list[Task]:
        """The task list a goal's shape implies."""
        text = goal.description
        artifact = self._artifact_name(text)
        readish = bool(_READ.search(text))
        writes = bool(_WRITE.search(text))
        computes = bool(_COMPUTE.search(text))

        tasks: list[Task] = []
        counter = _Counter()

        # A read/scan step, when the goal names one (or when nothing else fits).
        source = self._source_hint(text)
        read_task: Task | None = None
        if readish or (not writes and not computes and source):
            # A listing goal ("list the files") is a directory listing, not a
            # file read -- the two need different tools and different inputs, and
            # a read task with no file to read could never satisfy the goal.
            listing = bool(_LIST.search(text)) and not source
            if listing:
                read_task = self._make_task(
                    goal,
                    counter,
                    description="List the directory entries",
                    tool_hint="list",
                    inputs=(),
                    outputs=("directory entries",),
                    dependencies=(),
                    depth=depth,
                    arguments={"relative": "."},
                )
            else:
                read_task = self._make_task(
                    goal,
                    counter,
                    description=f"Read {source}" if source else "Read the required input",
                    tool_hint="read",
                    inputs=(source,) if source else (),
                    outputs=(f"content of {source}" if source else "input content",),
                    dependencies=(),
                    depth=depth,
                    arguments={"relative": source} if source else {},
                )
            tasks.append(read_task)

        # A compute step: only when the goal actually names an arithmetic task,
        # because the calculator tool takes an expression the plan must supply.
        if computes:
            expression = self._expression_hint(text)
            compute_task = self._make_task(
                goal,
                counter,
                description="Compute the required value",
                tool_hint="compute",
                inputs=(),
                outputs=("computed value",),
                dependencies=(),
                depth=depth,
                arguments={"expression": expression} if expression else {},
            )
            if compute_task.tool:
                tasks.append(compute_task)

        # A write step, when the goal names an artifact (or asks to produce one).
        if writes:
            write_deps = tuple(task.task_id for task in tasks)
            content_ref = f"{{{read_task.task_id}}}" if read_task else "result"
            tasks.append(
                self._make_task(
                    goal,
                    counter,
                    description=f"Write {artifact}" if artifact else "Write the result",
                    tool_hint="write",
                    inputs=("result",),
                    outputs=(artifact,) if artifact else ("written file",),
                    dependencies=write_deps,
                    depth=depth,
                    arguments={"relative": artifact or "result.txt", "content": content_ref},
                )
            )

        # A goal whose shape we did not recognise: one honest task, no fabrication.
        if not tasks:
            tasks.append(
                self._make_task(
                    goal,
                    counter,
                    description=goal.description.strip(),
                    tool_hint="",
                    inputs=(),
                    outputs=(goal.desired_outcome or goal.description,),
                    dependencies=(),
                    depth=depth,
                )
            )
        return tasks

    def _make_task(
        self,
        goal: Goal,
        counter: _Counter,
        *,
        description: str,
        tool_hint: str,
        inputs: tuple[str, ...],
        outputs: tuple[str, ...],
        dependencies: tuple[str, ...],
        depth: int,
        arguments: dict[str, str] | None = None,
    ) -> Task:
        """Build one task, selecting a permitted tool for its requirement."""
        task_id = f"{goal.goal_id}-t{counter.next()}"
        draft = Task(
            task_id=task_id,
            description=description,
            goal_id=goal.goal_id,
            parent_task="" if depth == 0 else goal.parent_goal,
            inputs=inputs,
            expected_outputs=outputs,
            dependencies=dependencies,
            required_permissions=goal.required_permissions,
            success_criteria=(f"{description} produced output",),
            failure_handling="retry once if reversible, else report and stop",
        )
        selection = select_tool(
            draft, self.catalogue, max_risk=RiskCategory.MEDIUM, requirement=tool_hint
        )
        tool = selection.chosen
        descriptor = self.catalogue.get(tool)
        reversible = descriptor.reversible if descriptor else True
        approval = bool(descriptor and descriptor.requires_approval)
        return Task(
            task_id=task_id,
            description=description,
            goal_id=goal.goal_id,
            parent_task=draft.parent_task,
            inputs=inputs,
            expected_outputs=outputs,
            dependencies=dependencies,
            tool=tool,
            arguments=dict(arguments or {}),
            required_permissions=goal.required_permissions,
            estimated_resources=(tool,) if tool else (),
            success_criteria=(f"{description} produced output",),
            failure_handling="retry once if reversible, else report and stop",
            requires_approval=approval,
            reversible=reversible,
            note="" if tool else "no permitted tool matched; the task is blocked",
        )

    @staticmethod
    def _artifact_name(text: str) -> str:
        """The output filename a goal names, if any.

        Prefers a filename after a write preposition ("to summary.md"); otherwise
        takes the last filename-like token, which is what "write summary.md"
        implies. The caller drops it when it is the same as the read source.
        """
        match = _ARTIFACT_HINT.search(text)
        if match:
            return match.group(1)
        tokens = [
            token
            for token in re.findall(r"[\w./-]+\.[A-Za-z0-9]{1,8}", text)
            if "." in token and not token.startswith(".")
        ]
        return tokens[-1] if tokens else ""

    @staticmethod
    def _source_hint(text: str) -> str:
        """A filename the goal names, if any (used as the read task's input)."""
        for token in re.findall(r"[\w./-]+\.[A-Za-z0-9]{1,8}", text):
            if "." in token and not token.startswith("."):
                return token
        return ""

    @staticmethod
    def _expression_hint(text: str) -> str:
        """A safe arithmetic expression the goal names, if any.

        Only digits, the four operators, parentheses, and a decimal point are
        accepted, so a goal can never smuggle a non-arithmetic string into the
        calculator.
        """
        match = re.search(r"([-+]?\d[\d\s.()+\-*/]*\d|\d)", text)
        if not match:
            return ""
        candidate = match.group(1).strip()
        if re.fullmatch(r"[0-9+\-*/().\s]+", candidate) and any(
            operator in candidate for operator in "+-*/"
        ):
            return candidate
        return ""

    # -------------------------------------------------- blocking conditions

    def tasks_requiring_approval(self, tasks: tuple[Task, ...]) -> tuple[Task, ...]:
        """Tasks whose tool requires user approval (section 5)."""
        return tuple(
            task
            for task in tasks
            if (descriptor := self.catalogue.get(task.tool)) is not None
            and descriptor.requires_approval
        )

    def tasks_blocked_by_information(self, tasks: tuple[Task, ...]) -> tuple[Task, ...]:
        """Tasks with no permitted tool -- blocked by missing information/capability."""
        return tuple(task for task in tasks if not task.tool)

    def classify(self, tasks: tuple[Task, ...]) -> dict[str, tuple[str, ...]]:
        """The section-5 classification of a task set, by id.

        Returns the sequential, independent, parallelizable, blocked, and
        approval-requiring tasks -- computed from the graph, so the caller reads
        the same classification the plan's validator will.
        """
        from core.planning.dependencies import DependencyGraph

        graph = DependencyGraph.from_tasks(tasks)
        return {
            "sequential": tuple(t.task_id for t in graph.sequential()),
            "independent": tuple(t.task_id for t in graph.independent()),
            "parallelizable": tuple(t.task_id for t in graph.parallelizable()),
            "blocked": tuple(t.task_id for t in self.tasks_blocked_by_information(tasks)),
            "requires_approval": tuple(
                t.task_id for t in self.tasks_requiring_approval(tasks)
            ),
        }


@dataclass
class _Counter:
    """A tiny monotonic counter for deterministic task ids."""

    value: int = 0

    def next(self) -> int:
        self.value += 1
        return self.value


def default_decomposition_engine(catalogue: ToolCatalogue) -> DecompositionEngine:
    """The engine the coordinator uses by default."""
    return DecompositionEngine(catalogue)


__all__ = [
    "MAX_DEPTH",
    "MAX_TASKS",
    "DecompositionEngine",
    "DecompositionError",
    "default_decomposition_engine",
]
