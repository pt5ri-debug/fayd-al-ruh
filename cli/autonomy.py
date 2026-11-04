"""CLI for Phase 8 -- autonomous planning and task execution.

    python -m cli autonomy plan "Read report.txt and write summary.md"
    python -m cli autonomy run  "Read report.txt and write summary.md" [--approve]
    python -m cli autonomy tools                 # the tool catalogue
    python -m cli autonomy goals                 # the recorded goals
    python -m cli autonomy history [--json]      # the execution audit trail
    python -m cli autonomy gate                  # the held-out gate's score
    python -m cli autonomy show <goal_id>        # one goal and its plan

Every command drives the *same* :class:`core.execution.coordinator.ExecutionCoordinator`
the GUI and the evaluation use, so the CLI, the window, and the gate share one
planner, one sandbox, one permission gate, and one audit log. A plan is validated
before it runs; an action outside the sandbox allow-list is refused; a high-risk
action needs ``--approve``; and a tool's success is never reported as the goal's.

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` the runtime is
unavailable, ``4`` a gate refused the operation, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3
EXIT_GATE = 4
EXIT_FAILURE = 5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autonomy", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    plan = sub.add_parser("plan", help="Plan a goal into a validated task plan.")
    plan.add_argument("goal", help="The goal, in plain language.")
    plan.add_argument("--root", default=".", help="The sandbox root (default: the current directory).")
    plan.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    run = sub.add_parser("run", help="Plan and execute a goal, then verify it.")
    run.add_argument("goal", help="The goal, in plain language.")
    run.add_argument("--root", default=".", help="The sandbox root (default: the current directory).")
    run.add_argument("--approve", action="store_true", help="Approve high-risk actions.")
    run.add_argument("--owner", default="", help="The Phase-5 personality the run is attributed to.")
    run.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    tools = sub.add_parser("tools", help="List the tool catalogue.")
    tools.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    goals = sub.add_parser("goals", help="List the recorded goals.")
    goals.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    show = sub.add_parser("show", help="Show one goal and its most recent plan.")
    show.add_argument("goal_id", help="The goal id.")
    show.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    history = sub.add_parser("history", help="The execution audit trail, newest first.")
    history.add_argument("--limit", type=int, default=20, help="How many entries (default: 20).")
    history.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    gate = sub.add_parser("gate", help="The held-out planning/execution gate score.")
    gate.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    return parser


def _data_root() -> str | None:
    """The writable data folder, or ``None`` when the layout cannot be resolved."""
    try:
        from core.paths import data_dir

        return str(data_dir())
    except Exception:
        return None


def _coordinator(root: str, *, owner: str = "", approve: bool = False):
    """Build the execution coordinator over the real sandbox at ``root``."""
    from core.execution.coordinator import ExecutionCoordinator
    from core.execution.engine import ExecutionEngine
    from core.execution.permissions import PermissionManager
    from core.planning.coordinator import PlanningCoordinator
    from core.tools import build_sandbox

    sandbox = build_sandbox(Path(root), read_only=False)
    permissions = PermissionManager(approvals=sandbox.approvals)
    if approve:
        # An explicit ``--approve`` grants a live approval for every tool the
        # sandbox permits, exactly as a user pressing Approve in the window does.
        for tool in sorted(sandbox.policy.allowed_tools):
            sandbox.approvals.request(tool, requested_by="cli").approve("user", ttl_seconds=3600)
    planner = PlanningCoordinator(data_root=_data_root())
    engine = ExecutionEngine(
        sandbox=sandbox,
        catalogue=planner.catalogue,
        permissions=permissions,
        actor=owner or "system",
    )
    return ExecutionCoordinator(
        planner=planner, engine=engine, root=Path(root), owner=owner, data_root=_data_root()
    )


def _goal_for(coordinator, description: str):
    return coordinator.planner.goals.create(description, desired_outcome=description)


def run_plan(args: argparse.Namespace) -> int:
    coordinator = _coordinator(args.root)
    goal = _goal_for(coordinator, args.goal)
    plan = coordinator.planner.plan(goal)
    if args.json:
        print(json.dumps(plan.to_primitive(), ensure_ascii=False, indent=2))
    else:
        print(plan.describe())
        print()
        print(plan.validation.describe() if plan.validation else "(no validation)")
    return EXIT_OK


def run_run(args: argparse.Namespace) -> int:
    coordinator = _coordinator(args.root, owner=args.owner, approve=args.approve)
    goal = _goal_for(coordinator, args.goal)
    result = coordinator.run_goal(goal)
    if args.json:
        print(json.dumps(
            {
                "goal": goal.goal_id,
                "plan": result.plan.plan_id,
                "status": result.execution.status.value,
                "outcome": result.verification.outcome.value,
                "success": result.success,
                "goal_achieved": result.verification.goal_achieved,
                "tasks": [
                    {"task": e.task_id, "tool": e.tool, "state": e.state.value, "ok": e.ok}
                    for e in result.execution.executions
                ],
                "recovery": result.recovery.describe(),
            },
            ensure_ascii=False, indent=2,
        ))
    else:
        print(result.describe())
    return EXIT_OK if result.success else EXIT_GATE


def run_tools(args: argparse.Namespace) -> int:
    from core.planning.coordinator import default_planner

    catalogue = default_planner().catalogue
    if args.json:
        print(json.dumps(
            [
                {
                    "tool": d.tool_id, "risk": d.risk.value, "reversible": d.reversible,
                    "mutates": d.mutates, "tags": list(d.tags), "description": d.description,
                }
                for d in catalogue.tools()
            ],
            ensure_ascii=False, indent=2,
        ))
    else:
        print(catalogue.describe())
    return EXIT_OK


def run_goals(args: argparse.Namespace) -> int:
    from core.planning.coordinator import default_planner

    planner = default_planner(data_root=_data_root())
    goals = planner.goals.all_goals()
    if args.json:
        print(json.dumps([g.to_primitive() for g in goals], ensure_ascii=False, indent=2))
    else:
        print(planner.goals.describe())
    return EXIT_OK


def run_show(args: argparse.Namespace) -> int:
    from core.planning.coordinator import default_planner

    planner = default_planner(data_root=_data_root())
    goal = planner.goals.get(args.goal_id)
    if goal is None:
        print(f"no goal {args.goal_id!r}")
        return EXIT_UNAVAILABLE
    if args.json:
        print(json.dumps(goal.to_primitive(), ensure_ascii=False, indent=2))
    else:
        print(goal.describe())
    return EXIT_OK


def run_history(args: argparse.Namespace) -> int:
    from core.execution.history import ExecutionHistory

    history = ExecutionHistory(data_root=_data_root())
    history.load()
    entries = history.recent(args.limit)
    if args.json:
        print(json.dumps([e.to_primitive() for e in entries], ensure_ascii=False, indent=2))
    elif entries:
        for entry in entries:
            print(entry.describe())
    else:
        print("no execution history")
    return EXIT_OK


def run_gate(args: argparse.Namespace) -> int:
    from core.planning_gate import task_execution_accuracy

    score = task_execution_accuracy()
    if args.json:
        print(json.dumps({"task_execution_accuracy": score}, indent=2))
    else:
        print(f"task_execution_accuracy: {score:.4f}")
    return EXIT_OK if score >= 1.0 else EXIT_GATE


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return EXIT_USAGE
    handlers = {
        "plan": run_plan,
        "run": run_run,
        "tools": run_tools,
        "goals": run_goals,
        "show": run_show,
        "history": run_history,
        "gate": run_gate,
    }
    try:
        return handlers[args.command](args)
    except ValueError as error:
        print(f"invalid: {error}")
        return EXIT_USAGE
    except FileNotFoundError as error:
        print(f"not available: {error}")
        return EXIT_UNAVAILABLE
    except Exception as error:  # pragma: no cover - defensive
        print(f"unexpected failure: {type(error).__name__}: {error}")
        return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
