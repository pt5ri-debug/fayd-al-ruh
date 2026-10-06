"""Answer a question from the web, cite sources, rate trust, and learn it.

    python -m cli web "What is the relationship between porosity and permeability?"
    python -m cli web --approve "What is a gamma ray log?"
    python -m cli web --endpoint https://api.example.com/search "What is a gas kick?"

Search is an external tool, so it is **denied by default**: the command prints
the permission reason until ``--approve`` grants the capability and records a
human approval, exactly like the rest of the governed-tool path. Without a live
endpoint the command runs against a small offline corpus of sources, so the
whole path is testable on a fresh clone with no network and no API key.

An answer is folded into the project's corpus only when its trust clears the
threshold, and the command says plainly when it did not.
"""

from __future__ import annotations

import argparse
import json
import os

from core.human_approval import HumanApprovalGate
from core.permissions import Capability, PermissionPolicy
from core.web_source import HttpSearchProvider, StaticSearchProvider, WebResult
from core.web_verifier import WebAssessment, build_web_verifier

_ENDPOINT_ENV = "FAYD_SEARCH_ENDPOINT"
_API_KEY_ENV = "FAYD_SEARCH_API_KEY"

_DEMO_RESULTS: tuple[WebResult, ...] = (
    WebResult(
        url="https://www.spe.org/petrophysics/permeability",
        title="Porosity and permeability relationship",
        snippet=(
            "Porosity measures the void space in the rock, while permeability "
            "measures how easily fluid flows through the connected pores of the "
            "reservoir formation."
        ),
        credibility=0.9,
    ),
    WebResult(
        url="https://en.wikipedia.org/wiki/Permeability_(earth_sciences)",
        title="Permeability in petroleum reservoirs",
        snippet=(
            "In petroleum reservoirs, permeability measures how easily fluid "
            "flows through porous rock, while porosity measures the fraction of "
            "void space in the formation."
        ),
        credibility=0.7,
    ),
    WebResult(
        url="https://en.wikipedia.org/wiki/Gamma_ray_logging",
        title="Gamma ray logging",
        snippet=(
            "A gamma ray log measures the natural radioactivity of rocks in a "
            "well, helping to identify formation lithology such as shale and "
            "reservoir sandstone."
        ),
        credibility=0.7,
    ),
)


def _permissions(approved: bool) -> PermissionPolicy:
    if not approved:
        return PermissionPolicy()
    return PermissionPolicy(
        allowed=frozenset(
            {
                Capability.SYMBOLIC_SIMULATION,
                Capability.SANDBOX_TASK,
                Capability.EXTERNAL_TOOL,
            }
        ),
        allowed_external_tools=frozenset({"web_search"}),
    )


def _approvals(approved: bool) -> HumanApprovalGate:
    gate = HumanApprovalGate()
    if approved:
        request = gate.request("web_search", requested_by="cli-user")
        request.approve("cli-reviewer")
    return gate


def _provider(endpoint: str | None):
    if endpoint:
        return HttpSearchProvider(endpoint, api_key=os.environ.get(_API_KEY_ENV))
    return StaticSearchProvider(_DEMO_RESULTS)


def _print_assessment(assessment: WebAssessment, *, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "question": assessment.question,
                    "answer": assessment.answer,
                    "citations": list(assessment.citations),
                    "credibility": assessment.credibility,
                    "agreement": assessment.agreement,
                    "support": assessment.support,
                    "trust": assessment.trust,
                    "learned": assessment.learned,
                    "learned_doc_id": assessment.learned_doc_id,
                    "search_allowed": assessment.search_allowed,
                    "contradicts_internal": assessment.contradicts_internal,
                    "reason": assessment.reason,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(assessment.summary())
    if not assessment.search_allowed:
        print("\n(search was not allowed; pass --approve to grant the external capability)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Answer from the web, compare with internal knowledge, and rate trust."
    )
    parser.add_argument("question", nargs="*", help="the question to answer")
    parser.add_argument("--approve", action="store_true", help="grant the external search capability")
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.environ.get(_ENDPOINT_ENV),
        help=f"search endpoint URL (or set {_ENDPOINT_ENV}); default: offline corpus",
    )
    parser.add_argument("--threshold", type=float, default=0.6, help="trust threshold to learn")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not args.question:
        parser.error("a question is required")

    question = " ".join(args.question)
    verifier = build_web_verifier(
        provider=_provider(args.endpoint),
        trust_threshold=args.threshold,
        permissions=_permissions(args.approve),
        approvals=_approvals(args.approve),
    )
    assessment = verifier.verify(question)
    _print_assessment(assessment, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
