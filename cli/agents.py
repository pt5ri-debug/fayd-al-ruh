"""The Agents CLI: the headless twin of the multi-agent layer.

It drives the *same* :class:`core.agents.system.AgentSystem` the rest of the
project uses, so the CLI, a GUI, and the tests share one set of personalities, one
capability gate, one runtime, and one isolation policy -- no logic is
reimplemented here.

Commands:

* ``list`` -- the seven personalities as agents, with their capabilities;
* ``show <character>`` -- one personality's full profile (identity, style,
  capabilities, memory and learning scope);
* ``capabilities <character>`` -- what it may and may not do, and why;
* ``route <question> [--conversation C] [--override X] [--user-choice Y]`` --
  which personality should answer, and the traceable reason;
* ``run <character> <question> [--timeout S]`` -- run a question through a
  personality's agent runtime (through an injected echo executor by default, so
  the CLI works with no model; ``--mouth`` uses the real mouth when installed);
* ``collaborate <from> <question> <target...>`` -- ask several personalities the
  same question and keep each answer separate;
* ``conflicts <from> <question> <target...> [--policy preserve|evidence]`` --
  compare the answers and report whether any consensus exists;
* ``monitoring [--json]`` -- each personality's measured activity;
* ``isolation`` -- the memory and learning isolation policy as it stands.

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` the personality is
unknown, ``4`` a gate refused the operation, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from core.agents.capabilities import AgentCapability, CapabilityDenied
from core.agents.conflict import AggregationPolicy, ConflictResolver, Position
from core.agents.routing import TaskType
from core.agents.runtime import AgentResponse
from core.agents.system import AgentSystem
from core.learning_isolation import PRIVATE_OWNERS, SHAREABLE_OWNERS, LearningAccessError


def _system(*, mouth: str = "") -> AgentSystem:
    """The shared agent system (an echo executor unless a mouth is named)."""
    if mouth:
        return AgentSystem(mouth=mouth)
    return AgentSystem(executor=_echo)


def _echo(request: object) -> AgentResponse:
    """A stand-in executor so the CLI works with no model installed."""
    name = getattr(getattr(request, "profile", None), "name", "the personality")
    question = getattr(request, "question", "")
    return AgentResponse(
        ok=True,
        text=f"[echo] {name} would answer {question!r} (no model loaded in the CLI).",
        capabilities_used=(AgentCapability.ANSWER,),
    )


def _fail(message: str, code: int = 2) -> int:
    print(message)
    return code


# ------------------------------------------------------------------ commands

def _cmd_list(args: argparse.Namespace) -> int:
    system = _system()
    profiles = system.profiles()
    if args.json:
        print(json.dumps([profile.to_primitive() for profile in profiles], indent=2, ensure_ascii=False))
        return 0
    for profile in profiles:
        print(profile.describe())
        print()
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    system = _system()
    try:
        profile = system.profile(args.character)
    except LearningAccessError as error:
        return _fail(str(error), 3)
    if args.json:
        print(json.dumps(profile.to_primitive(), indent=2, ensure_ascii=False))
        return 0
    print(profile.describe())
    return 0


def _cmd_capabilities(args: argparse.Namespace) -> int:
    system = _system()
    capabilities = system.capabilities(args.character)
    if args.json:
        print(json.dumps(capabilities.to_primitive(), indent=2, ensure_ascii=False))
        return 0
    print(f"{args.character}: {capabilities.describe()}")
    return 0


def _cmd_route(args: argparse.Namespace) -> int:
    system = _system()
    decision = system.route(
        args.question,
        conversation=args.conversation or "",
        task_type=TaskType(args.task_type) if args.task_type else None,
        override=args.override or "",
        user_choice=args.user_choice or "",
    )
    if args.json:
        print(json.dumps(decision.to_primitive(), indent=2, ensure_ascii=False))
        return 0
    print(decision.describe())
    return 0 if decision.suitable or args.allow_none else 4


def _cmd_run(args: argparse.Namespace) -> int:
    system = _system(mouth=args.mouth)
    outcome = system.run(args.character, args.question, timeout=args.timeout)
    if args.json:
        print(
            json.dumps(
                {
                    "task": outcome.task.task_id,
                    "owner": outcome.task.owner,
                    "status": outcome.status.value,
                    "ok": outcome.ok,
                    "duration_seconds": round(outcome.duration_seconds, 4),
                    "text": outcome.response.text,
                    "error": outcome.response.error,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0 if outcome.ok else 4
    print(outcome.describe())
    return 0 if outcome.ok else 4


def _cmd_collaborate(args: argparse.Namespace) -> int:
    system = _system()
    report = system.collaboration.collaborate(args.sender, args.question, tuple(args.targets))
    if args.json:
        print(
            json.dumps(
                {
                    "shared_id": report.shared_id,
                    "ok": report.ok,
                    "results": [
                        {
                            "from": result.from_owner,
                            "to": result.to_owner,
                            "ok": result.ok,
                            "text": result.text,
                            "error": result.error,
                        }
                        for result in report.results
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0 if report.ok else 4
    print(report.describe())
    return 0 if report.ok else 4


def _cmd_conflicts(args: argparse.Namespace) -> int:
    system = _system()
    report = system.collaboration.collaborate(args.sender, args.question, tuple(args.targets))
    positions = tuple(
        Position(owner=result.to_owner, text=result.text)
        for result in report.results
        if result.ok
    )
    policy = AggregationPolicy(args.policy)
    conflict = ConflictResolver().resolve(args.question, positions, policy=policy)
    if args.json:
        print(json.dumps(conflict.to_primitive(), indent=2, ensure_ascii=False))
        return 0
    print(conflict.describe())
    return 0


def _cmd_monitoring(args: argparse.Namespace) -> int:
    system = _system()
    report = system.monitoring()
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "owner": activity.owner,
                        "total": activity.total,
                        "succeeded": activity.succeeded,
                        "failed": activity.failed,
                        "refused": activity.refused,
                        "success_rate": round(activity.success_rate, 4),
                    }
                    for activity in report.activities
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    print(report.describe())
    return 0


def _cmd_isolation(args: argparse.Namespace) -> int:
    from core.agents.learning_scope import aggregated_dataset_is_clean, aggregated_dataset_owners

    if args.json:
        print(
            json.dumps(
                {
                    "private": sorted(PRIVATE_OWNERS),
                    "shareable": sorted(SHAREABLE_OWNERS),
                    "aggregated_dataset": list(aggregated_dataset_owners()),
                    "dataset_is_clean": aggregated_dataset_is_clean(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    print(f"private characters : {', '.join(sorted(PRIVATE_OWNERS))}")
    print(f"shareable characters: {', '.join(sorted(SHAREABLE_OWNERS))}")
    print(f"aggregated training dataset owners: {', '.join(aggregated_dataset_owners())}")
    print(f"dataset is clean of private owners: {aggregated_dataset_is_clean()}")
    return 0


# ------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cli agents",
        description="The seven personalities as independent agents in one system.",
    )
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="list every personality as an agent")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show", help="show one personality's profile")
    p_show.add_argument("character")
    p_show.add_argument("--json", action="store_true")
    p_show.set_defaults(func=_cmd_show)

    p_caps = sub.add_parser("capabilities", help="what a personality may do")
    p_caps.add_argument("character")
    p_caps.add_argument("--json", action="store_true")
    p_caps.set_defaults(func=_cmd_capabilities)

    p_route = sub.add_parser("route", help="choose a personality for a question")
    p_route.add_argument("question")
    p_route.add_argument("--conversation", default="")
    p_route.add_argument("--task-type", default="")
    p_route.add_argument("--override", default="")
    p_route.add_argument("--user-choice", default="")
    p_route.add_argument("--allow-none", action="store_true", help="exit 0 even when none is suitable")
    p_route.add_argument("--json", action="store_true")
    p_route.set_defaults(func=_cmd_route)

    p_run = sub.add_parser("run", help="run a question through a personality's runtime")
    p_run.add_argument("character")
    p_run.add_argument("question")
    p_run.add_argument("--timeout", type=float, default=None)
    p_run.add_argument("--mouth", default="", help="answer through the real mouth (needs a model)")
    p_run.add_argument("--json", action="store_true")
    p_run.set_defaults(func=_cmd_run)

    p_collab = sub.add_parser("collaborate", help="ask several personalities and keep each answer")
    p_collab.add_argument("sender")
    p_collab.add_argument("question")
    p_collab.add_argument("targets", nargs="+")
    p_collab.add_argument("--json", action="store_true")
    p_collab.set_defaults(func=_cmd_collaborate)

    p_conf = sub.add_parser("conflicts", help="compare several answers for consensus")
    p_conf.add_argument("sender")
    p_conf.add_argument("question")
    p_conf.add_argument("targets", nargs="+")
    p_conf.add_argument("--policy", choices=[p.value for p in AggregationPolicy], default="preserve")
    p_conf.add_argument("--json", action="store_true")
    p_conf.set_defaults(func=_cmd_conflicts)

    p_mon = sub.add_parser("monitoring", help="each personality's measured activity")
    p_mon.add_argument("--json", action="store_true")
    p_mon.set_defaults(func=_cmd_monitoring)

    p_iso = sub.add_parser("isolation", help="the memory and learning isolation policy")
    p_iso.add_argument("--json", action="store_true")
    p_iso.set_defaults(func=_cmd_isolation)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(args, "func", None) is None:
        return _cmd_list(argparse.Namespace(json=False))
    try:
        return int(args.func(args))
    except LearningAccessError as error:
        return _fail(str(error), 3)
    except CapabilityDenied as error:
        return _fail(str(error), 4)
    except (ValueError, OSError, KeyError) as error:
        return _fail(f"{type(error).__name__}: {error}", 5)


__all__ = ["build_parser", "main"]
