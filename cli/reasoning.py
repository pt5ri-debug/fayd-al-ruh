"""CLI for Phase 4 -- the reasoning and inference engine.

    python -m cli reasoning decompose "<question>"   # the question's reasoning units
    python -m cli reasoning reason "<question>"      # the whole reasoning, routed per part
    python -m cli reasoning plan "<how-to question>"
    python -m cli reasoning decide "<what-to-do question>"
    python -m cli reasoning believe "<does-it-hold question>"
    python -m cli reasoning compare "<is-it-like question>"
    python -m cli reasoning facts "<question>"        # the graph's stored relations
    python -m cli reasoning infer "<question>"        # what the graph entails
    python -m cli reasoning connect "<question>"      # walked multi-hop chains
    python -m cli reasoning causal "<question>"       # directed causal paths
    python -m cli reasoning conflicts [<question>]    # contradictions the graph holds
    python -m cli reasoning verify <a> <relation> <b>
    python -m cli reasoning choose <tool> [<tool> ...]  # bounded tool choice
    python -m cli reasoning metrics                    # the measured reasoning metrics

**Reasoning is measured, not asserted.** Every command drives the *same*
:class:`core.reasoning_api.ReasoningAPI` the controller and the GUI use, so the
CLI, the window, and the answer path reason over one graph. Nothing here decides
*what* an engine returns -- it only routes a question to the engine its kind
warrants, and a question no engine can answer yields an honest absence.

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` the reasoning engine is
unavailable, ``4`` a gate refused the operation, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from core.reasoning_api import ReasoningAPI, ReasoningResult

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3
EXIT_GATE = 4
EXIT_FAILURE = 5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reasoning", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    decompose = sub.add_parser("decompose", help="Break a question into reasoning units.")
    decompose.add_argument("question", help="The question to decompose.")
    decompose.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    reason = sub.add_parser("reason", help="Reason about a question, routing each part.")
    reason.add_argument("question", help="The question to reason about.")
    reason.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    for name, help_text in (
        ("plan", "The ordered procedure the corpus states."),
        ("decide", "The ranked recommendation the graph states."),
        ("believe", "The evidence-weighed verdict the graph supports."),
        ("compare", "The structural analogy the graph supports."),
        ("facts", "The graph's stored relations on the question's concepts."),
        ("infer", "The relations the graph entails."),
        ("connect", "The walked multi-hop chains between the question's concepts."),
        ("causal", "The directed causal paths the graph states."),
        ("hypotheses", "The candidate hypotheses the graph licenses, ranked."),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("question", help="The question to reason about.")
        command.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    conflicts = sub.add_parser("conflicts", help="Contradictions the graph holds.")
    conflicts.add_argument("question", nargs="?", default=None, help="Scope to a question.")
    conflicts.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    path = sub.add_parser("path", help="The shortest chain between two concepts.")
    path.add_argument("source", help="The concept to start from.")
    path.add_argument("target", help="The concept to reach.")
    path.add_argument("--max-hops", type=int, default=3, help="How many hops at most.")
    path.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    verify = sub.add_parser("verify", help="Verify a conclusion against the graph.")
    verify.add_argument("source", help="The subject concept.")
    verify.add_argument("relation", help="The relation type (e.g. increases).")
    verify.add_argument("target", help="The object concept.")
    verify.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    choose = sub.add_parser("choose", help="Choose a tool, bounded by the sandbox allow-list.")
    choose.add_argument("tools", nargs="+", help="The proposed tool names, in order of preference.")
    choose.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    metrics = sub.add_parser("metrics", help="The measured reasoning metrics.")
    metrics.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    return parser


def _result_json(result: ReasoningResult) -> dict:
    return {
        "kind": result.kind,
        "question": result.question,
        "conclusion": result.conclusion,
        "confidence": result.confidence,
        "derived": result.derived,
        "trace": list(result.trace),
        "evidence": list(result.evidence),
        "limitations": list(result.limitations),
        "parts": [_result_json(part) for part in result.parts],
    }


def _emit(result: ReasoningResult, *, as_json: bool) -> int:
    if as_json:
        print(json.dumps(_result_json(result), ensure_ascii=False, indent=2))
    else:
        print(result.describe())
    return EXIT_OK


def run_decompose(args: argparse.Namespace) -> int:
    from core.problem_decomposition import decompose

    decomposition = decompose(args.question, ontology=ReasoningAPI().ontology)
    if decomposition is None:
        print("the question was empty; nothing to decompose.")
        return EXIT_OK
    if args.json:
        print(json.dumps(decomposition.to_primitive(), ensure_ascii=False, indent=2))
        return EXIT_OK
    print(decomposition.describe())
    return EXIT_OK


def run_reason(args: argparse.Namespace) -> int:
    return _emit(ReasoningAPI().reason(args.question), as_json=args.json)


def _simple(method_name: str):
    def handler(args: argparse.Namespace) -> int:
        result = getattr(ReasoningAPI(), method_name)(args.question)
        return _emit(result, as_json=args.json)

    return handler


def run_conflicts(args: argparse.Namespace) -> int:
    result = ReasoningAPI().conflicts(args.question)
    return _emit(result, as_json=args.json)


def run_path(args: argparse.Namespace) -> int:
    result = ReasoningAPI().path(args.source, args.target, max_hops=args.max_hops)
    return _emit(result, as_json=args.json)


def run_verify(args: argparse.Namespace) -> int:
    result = ReasoningAPI().verify(args.source, args.relation, args.target)
    return _emit(result, as_json=args.json)


def run_choose(args: argparse.Namespace) -> int:
    """Choose a tool, bounded by the sandbox's own allow-list.

    The sandbox is deny-by-default and read-only, so a proposal for a mutating or
    unknown tool is reported as refused -- the model proposes, the sandbox disposes.
    """
    import tempfile
    from pathlib import Path

    from core.tool_agent import ToolChooser, ToolProposal
    from core.tools import build_sandbox

    with tempfile.TemporaryDirectory() as folder:
        sandbox = build_sandbox(Path(folder), read_only=True)
        chooser = ToolChooser(sandbox)
        chosen = chooser.choose(ToolProposal(tool) for tool in args.tools)
    if args.json:
        print(
            json.dumps(
                [
                    {"tool": item.proposal.tool, "allowed": item.allowed, "reason": item.reason}
                    for item in chosen
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return EXIT_OK
    for item in chosen:
        print(item.describe())
    return EXIT_OK


def run_metrics(args: argparse.Namespace) -> int:
    from core.conclusion_verification import verification_accuracy
    from core.hypothesis_reasoning import hypothesis_accuracy
    from core.reasoning_api import facade_accuracy
    from core.reasoning_report import reasoning_trace_accuracy
    from core.tool_agent import tool_selection_accuracy

    metrics = {
        "decomposition": _decomposition_accuracy(),
        "conclusion_verification": verification_accuracy(),
        "reasoning_trace": reasoning_trace_accuracy(),
        "reasoning_facade": facade_accuracy(),
        "hypotheses": hypothesis_accuracy(),
        "tool_selection": tool_selection_accuracy(),
    }
    if args.json:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return EXIT_OK
    for name, value in metrics.items():
        print(f"{name:26} {value:g}")
    return EXIT_OK


def _decomposition_accuracy() -> float:
    from core.problem_decomposition import DECOMPOSITION_JUDGEMENTS, decomposition_accuracy

    return decomposition_accuracy(DECOMPOSITION_JUDGEMENTS)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])
    if not args.command:
        parser.print_help()
        return EXIT_OK
    handlers = {
        "decompose": run_decompose,
        "reason": run_reason,
        "plan": _simple("plan"),
        "decide": _simple("decide"),
        "believe": _simple("believe"),
        "compare": _simple("compare"),
        "facts": _simple("facts"),
        "infer": _simple("infer"),
        "connect": _simple("connect"),
        "causal": _simple("causal"),
        "hypotheses": _simple("hypotheses"),
        "conflicts": run_conflicts,
        "path": run_path,
        "verify": run_verify,
        "choose": run_choose,
        "metrics": run_metrics,
    }
    try:
        return handlers[args.command](args)
    except ValueError as error:
        print(f"invalid: {error}")
        return EXIT_USAGE
    except FileNotFoundError as error:
        print(f"unavailable: {error}")
        return EXIT_UNAVAILABLE
    except Exception as error:  # pragma: no cover - last-resort honesty
        print(f"failure: {error}")
        return EXIT_FAILURE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
