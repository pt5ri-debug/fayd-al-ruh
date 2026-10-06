"""Demonstrate the sandboxed tool layer and the tool agent.

    python -m ai.agent_demo

Runs a small plan inside a temporary, read/write sandbox: write a note, read it
back, list the directory, search for a term, and evaluate an expression. Every
step goes through the same allow-list, root-confinement, and capability gates the
real agent uses, and the full trace is printed.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from core.tool_agent import ToolAgent, ToolCall, plan_from_calls
from core.tools import build_sandbox


def run_demo(root: Path | None = None) -> int:
    created = root is None
    sandbox_root = (
        Path(tempfile.mkdtemp(prefix="fayd-sandbox-")) if root is None else Path(root)
    )
    sandbox = build_sandbox(sandbox_root, read_only=False)
    plan = plan_from_calls(
        "اكتب ملاحظة بترولية، ثم تحقّق منها وابحث فيها",
        (
            ToolCall("write_file", {"relative": "notes/well.txt", "content": "النفاذية عالية"}),
            ToolCall("read_file", {"relative": "notes/well.txt"}),
            ToolCall("list_dir", {"relative": "notes"}),
            ToolCall("search", {"query": "النفاذية", "relative": "."}),
            ToolCall("calculator", {"expression": "2 * (3 + 4)"}),
        ),
        reasons=("كتابة", "قراءة", "عرض", "بحث", "حساب"),
    )
    run = ToolAgent(sandbox).run(plan)
    print(f"الهدف: {run.goal}")
    for step in run.steps:
        status = "ok " if step.result.ok else "خطأ"
        detail = step.result.output if step.result.ok else step.result.error
        print(f"  [{status}] {step.call.tool} ({step.reason}): {detail}")
    print(f"النتيجة: {'نجح' if run.succeeded else 'فشل'}")
    if created:
        print(f"(صندوق مؤقت: {sandbox_root})")
    return 0 if run.succeeded else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Demo the sandboxed tool layer.")
    parser.add_argument("--root", type=Path, default=None, help="sandbox root directory")
    args = parser.parse_args(argv)
    return run_demo(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
