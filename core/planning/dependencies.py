"""The dependency graph and ordering (Phase 8, sections 5 and 8).

Given a set of tasks, this module builds the dependency graph, validates it
(no unknown dependency, no cycle), and answers the scheduling questions the
execution engine asks: which tasks are *ready* now, which are *blocked*, and in
what order may the whole set run. It also classifies the set the way section 5
asks -- sequential, independent, parallelizable, blocked by missing information,
and requiring approval -- without doing any work itself.

It reuses :class:`core.planner.Planner`'s own topological ordering for the
symbolic plan, but keeps a task-level graph here so a plan can be inspected and
edited without touching the symbolic layer.

Dependency-free and Tk-free.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from core.planning.tasks import Task, TaskState


class DependencyError(ValueError):
    """Raised when a task graph is invalid (unknown dependency or a cycle)."""


@dataclass
class DependencyGraph:
    """A validated task dependency graph.

    The graph is built from tasks' own ``dependencies``; ``validate`` runs in the
    constructor so an invalid graph cannot exist. ``ready`` and ``blocked`` are
    the scheduling queries; ``order`` is a deterministic topological order.
    """

    tasks: dict[str, Task] = field(default_factory=dict)
    _dependents: dict[str, set[str]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.tasks:
            self._build()

    # --------------------------------------------------------------- building

    @classmethod
    def from_tasks(cls, tasks: Iterable[Task]) -> DependencyGraph:
        mapping = {task.task_id: task for task in tasks}
        graph = cls()
        graph.tasks = mapping
        graph._build()
        return graph

    def _build(self) -> None:
        self._dependents = {task_id: set() for task_id in self.tasks}
        for task in self.tasks.values():
            for dependency in task.dependencies:
                if dependency not in self.tasks:
                    raise DependencyError(
                        f"task {task.task_id} depends on unknown task {dependency!r}"
                    )
                self._dependents[dependency].add(task.task_id)
        self._check_acyclic()

    def _check_acyclic(self) -> None:
        # Kahn's algorithm: if not every node is emitted, a cycle exists.
        indegree = dict.fromkeys(self.tasks, 0)
        for task in self.tasks.values():
            for dependency in task.dependencies:
                indegree[task.task_id] += 1
                _ = dependency
        queue = [tid for tid, degree in indegree.items() if degree == 0]
        seen = 0
        while queue:
            current = queue.pop()
            seen += 1
            for dependent in self._dependents[current]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    queue.append(dependent)
        if seen != len(self.tasks):
            raise DependencyError("the task graph contains a cycle")

    # -------------------------------------------------------------- queries

    def dependencies_of(self, task_id: str) -> tuple[str, ...]:
        return self.tasks[task_id].dependencies

    def dependents_of(self, task_id: str) -> tuple[str, ...]:
        return tuple(sorted(self._dependents.get(task_id, ())))

    def _satisfied(self, task: Task) -> bool:
        return all(
            self.tasks[dep].state is TaskState.COMPLETED for dep in task.dependencies
        )

    def ready(self) -> tuple[Task, ...]:
        """Tasks whose dependencies are all completed and that are not terminal.

        A task is *ready* when it is ``PENDING``/``READY`` and every dependency
        has completed; it is not returned once it is running or terminal.
        """
        ready = [
            task
            for task in self.tasks.values()
            if task.state in {TaskState.PENDING, TaskState.READY} and self._satisfied(task)
        ]
        return tuple(sorted(ready, key=lambda t: t.task_id))

    def blocked(self) -> tuple[Task, ...]:
        """Tasks that cannot run yet: a dependency failed, or is still pending.

        A task whose dependency failed (or was cancelled/skipped) is *blocked*
        and can never become ready without a replan, so the engine can report it
        honestly rather than waiting forever.
        """
        blocked: list[Task] = []
        for task in self.tasks.values():
            if task.state is not TaskState.PENDING and task.state is not TaskState.READY:
                continue
            for dep in task.dependencies:
                dep_state = self.tasks[dep].state
                if dep_state in {
                    TaskState.FAILED,
                    TaskState.CANCELLED,
                    TaskState.SKIPPED,
                }:
                    blocked.append(task)
                    break
        return tuple(sorted(blocked, key=lambda t: t.task_id))

    def order(self) -> tuple[Task, ...]:
        """A deterministic topological order (dependencies before dependents)."""
        indegree = {task_id: len(task.dependencies) for task_id, task in self.tasks.items()}
        queue = sorted(tid for tid, degree in indegree.items() if degree == 0)
        result: list[Task] = []
        while queue:
            current = queue.pop(0)
            result.append(self.tasks[current])
            newly_free: list[str] = []
            for dependent in self._dependents[current]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    newly_free.append(dependent)
            queue.extend(sorted(newly_free))
            queue.sort()
        return tuple(result)

    def sequential(self) -> tuple[Task, ...]:
        """Tasks that must run one at a time: they depend on another task."""
        return tuple(
            sorted(
                (task for task in self.tasks.values() if task.dependencies),
                key=lambda t: t.task_id,
            )
        )

    def independent(self) -> tuple[Task, ...]:
        """Tasks with no dependency and no dependent -- fully independent."""
        return tuple(
            sorted(
                (
                    task
                    for task in self.tasks.values()
                    if not task.dependencies and not self._dependents.get(task.task_id)
                ),
                key=lambda t: t.task_id,
            )
        )

    def parallelizable(self) -> tuple[Task, ...]:
        """Tasks with no dependency of their own (they may start immediately)."""
        return tuple(
            sorted(
                (task for task in self.tasks.values() if not task.dependencies),
                key=lambda t: t.task_id,
            )
        )

    def missing_information(self) -> tuple[Task, ...]:
        """Tasks whose declared inputs are not produced by any dependency.

        Section 5 asks the engine to identify a task *blocked by missing
        information*: an input a task needs that no task in the plan produces and
        that is not listed as a precondition it could already satisfy. This is a
        report, not a refusal -- the planner may still run it if the input comes
        from outside the plan.
        """
        produced = {output for task in self.tasks.values() for output in task.expected_outputs}
        missing: list[Task] = []
        for task in self.tasks.values():
            needed = set(task.inputs)
            if needed and not needed <= produced and not needed <= set(task.preconditions):
                missing.append(task)
        return tuple(sorted(missing, key=lambda t: t.task_id))

    def cycle_nodes(self) -> tuple[str, ...]:
        """The nodes on a cycle, if any (empty for a valid graph)."""
        # A graph that reached this point is acyclic, so there are none.
        return ()

    def describe(self) -> str:
        lines = [
            f"tasks: {len(self.tasks)}",
            f"ready: {len(self.ready())}",
            f"blocked: {len(self.blocked())}",
        ]
        return "\n".join(lines)


__all__ = ["DependencyError", "DependencyGraph"]
