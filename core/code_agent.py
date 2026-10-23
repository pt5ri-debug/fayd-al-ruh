"""The Studio's coding agent: an OpenHands-style loop over a sandboxed workspace.

This is the part that turns a plain-language request ("add a `--json` flag to the
CLI") into real edits. It follows the same shape as the project's other agent
loops -- the model proposes one action, the sandbox runs it, the outcome is fed
back -- with three deliberate differences that make it safe to point at the
project's own source:

* **It only ever touches the workspace.** Every read and write goes through
  :class:`core.code_studio.CodeStudio`, which confines paths to the workspace
  root and refuses non-text files. The model cannot reach outside the isolated
  copy, so a bad suggestion cannot damage the running checkout.
* **It speaks a tiny, strict JSON protocol.** The model replies with one JSON
  object: ``{"action": "list_files"}``, ``{"action": "read_file", "path": ...}``,
  ``{"action": "write_file", "path": ..., "content": ...}``, or
  ``{"action": "done", "summary": ...}``. Parsing is defensive: a malformed reply
  is reported to the model as an error and the loop continues, rather than
  crashing. Nothing in the reply is ``eval``-ed or executed as code.
* **It is bounded and auditable.** The loop has a step budget, and every step --
  the model's own reasoning, the action, and the result -- is recorded in an
  :class:`AgentTrace` the window shows. A run that hits the budget stops honestly
  instead of looping forever.

The model itself is injected as a :class:`core.coder_model.CoderClient`, so a
test drives the whole agent with a scripted fake and needs no Ollama.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.code_studio import SKIP_DIRS, CodeStudio, StudioError, Workspace
from core.coder_model import CoderClient, CoderMessage, CoderUnavailable

#: How many model turns one request may take before the agent stops. Each turn is
#: one action, so this is also roughly the number of edits a run can make.
DEFAULT_MAX_STEPS = 24

#: How much of a file is shown to the model in one read. A whole small file fits;
#: a large one is truncated so the prompt cannot grow without bound.
DEFAULT_READ_CHARS = 12_000

#: The actions the model may ask for. ``done`` ends the run.
KNOWN_ACTIONS = frozenset(
    {
        "list_files",
        "find_files",
        "search",
        "read_file",
        "edit_file",
        "write_file",
        "diff",
        "run_checks",
        "done",
    }
)

#: Actions that name a file, so the parser can insist on a ``path``.
PATH_ACTIONS = frozenset({"read_file", "edit_file", "write_file"})

#: The protocol the model must follow. Kept short and explicit: a 7B code model
#: follows a small, literal instruction far more reliably than a loose one.
AGENT_SYSTEM_PROMPT = """You are a careful senior software engineer editing a project inside an \
isolated workspace (a copy of a git repository). You change real files, so be precise.

You reply with EXACTLY ONE JSON object and nothing else. No prose, no markdown fences.

Available actions:
  {"action": "list_files"}
  {"action": "find_files", "pattern": "**/*.py"}
  {"action": "search", "query": "some literal text", "path": "optional/subdir"}
  {"action": "read_file", "path": "relative/path.py"}
  {"action": "edit_file", "path": "relative/path.py", "old": "exact text to replace", "new": "replacement"}
  {"action": "write_file", "path": "relative/path.py", "content": "the full new file contents"}
  {"action": "diff"}
  {"action": "run_checks"}
  {"action": "done", "summary": "one line describing what you changed and why"}

Rules:
- Work one step at a time. Search or read a file before you change it.
- PREFER "edit_file" over "write_file": it replaces one exact snippet, so you do
  not have to reproduce a whole file and cannot lose lines by accident. The
  "old" text must match the file EXACTLY (including indentation) and must appear
  once; add surrounding lines to make it unique, or set "replace_all": true.
- Use "write_file" only for a NEW file or when most of the file changes.
- Use "search" to find where something lives, "find_files" to find files by name.
- Use "diff" to review your own change as a unified diff before you finish.
- Run "run_checks" before you finish, and fix anything it reports.
- Only edit text files inside the workspace. Paths are relative to the workspace root.
- When the change is complete and the checks pass, reply with the "done" action.
"""


class AgentError(RuntimeError):
    """The agent could not carry out the request."""


@dataclass(frozen=True)
class AgentStep:
    """One turn of the loop: what the model said, did, and got back."""

    index: int
    action: str
    detail: str = ""
    result: str = ""
    ok: bool = True

    def live_lines(self, *, max_result_lines: int = 12) -> list[str]:
        """A short, human-readable rendering of this step for a live transcript.

        The window shows one of these per turn as the model works, so the reader
        sees the same shape as a code host's agent chat: the action it took, then
        what came back. A long observation is trimmed, so one big file read cannot
        flood the transcript.
        """
        if self.action == "done":
            return [f"✓ Done: {self.detail}"]
        if self.action == "invalid":
            return [f"⚠ Asked the model to reply again: {self.detail}"]
        mark = "→" if self.ok else "✗"
        head = f"{mark} {self.action} {self.detail}".rstrip()
        lines = [head]
        body = (self.result or "").rstrip()
        if body:
            observed = body.splitlines()
            if len(observed) > max_result_lines:
                kept = observed[:max_result_lines]
                kept.append(f"… [{len(observed) - max_result_lines} more line(s)]")
                observed = kept
            lines.extend(f"   {line}" for line in observed)
        return lines


@dataclass
class AgentTrace:
    """The full, inspectable record of one agent run."""

    task: str
    steps: list[AgentStep] = field(default_factory=list)
    summary: str = ""
    stopped_reason: str = ""
    writes: list[str] = field(default_factory=list)
    #: True once the model has run the project's checks at least once.
    checked: bool = False
    #: Whether the most recent ``run_checks`` passed.
    checks_ok: bool = False

    @property
    def ok(self) -> bool:
        """True when the agent finished on its own ``done`` action."""
        return bool(self.summary) and not self.stopped_reason

    def render(self) -> str:
        """A readable transcript for the window and the CLI."""
        lines = [f"Task: {self.task}"]
        for step in self.steps:
            mark = "ok" if step.ok else "!!"
            lines.append(f"[{step.index}] {step.action} {step.detail} -> {mark} {step.result}".rstrip())
        if self.summary:
            lines.append(f"Done: {self.summary}")
        if self.stopped_reason:
            lines.append(f"Stopped: {self.stopped_reason}")
        if self.writes:
            lines.append("Files written: " + ", ".join(self.writes))
        if self.checked:
            lines.append("Checks: " + ("passed" if self.checks_ok else "failed"))
        return "\n".join(lines)


@dataclass(frozen=True)
class AgentAction:
    """A parsed action from the model's JSON reply."""

    name: str
    path: str = ""
    content: str = ""
    summary: str = ""
    query: str = ""
    pattern: str = ""
    old: str = ""
    new: str = ""
    replace_all: bool = False

    def describe(self) -> str:
        """A short, human-readable subject for the trace line."""
        if self.name == "search":
            return f"{self.query!r}" + (f" in {self.path}" if self.path else "")
        if self.name == "find_files":
            return self.pattern
        if self.name in {"read_file", "write_file", "edit_file"}:
            return self.path
        if self.name == "diff":
            return "the workspace diff"
        if self.name == "run_checks":
            return "the project's checks"
        return ""


def parse_action(reply: str) -> AgentAction:
    """Parse the model's reply into one action, or raise :class:`AgentError`.

    The parser is forgiving about surrounding text -- a model sometimes wraps the
    object in a sentence or a code fence -- but strict about the object itself: an
    unknown action, a missing path, or content that is not a string is an error the
    loop reports back to the model.
    """
    text = (reply or "").strip()
    if not text:
        raise AgentError("The model returned an empty reply.")
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1] if text.lstrip().startswith("json") else text
        text = text.strip("`").strip()

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise AgentError("The model did not return a JSON object.")
    try:
        data: Any = json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        raise AgentError(f"The model's JSON did not parse: {error}") from error
    if not isinstance(data, dict):
        raise AgentError("The model's JSON was not an object.")

    name = str(data.get("action", "")).strip().lower()
    if name not in KNOWN_ACTIONS:
        raise AgentError(
            f"Unknown action {name!r}. Use one of: {', '.join(sorted(KNOWN_ACTIONS))}."
        )
    path = str(data.get("path", "")).strip()
    content = data.get("content", "")
    query = str(data.get("query", "")).strip()
    pattern = str(data.get("pattern", "")).strip()
    old = data.get("old", "")
    new = data.get("new", "")
    if name in PATH_ACTIONS and not path:
        raise AgentError(f"The {name} action needs a 'path'.")
    if name == "search" and not query:
        raise AgentError("The search action needs a 'query'.")
    if name == "find_files" and not pattern:
        raise AgentError("The find_files action needs a 'pattern' (e.g. '**/*.py').")
    if name == "edit_file":
        if not isinstance(old, str) or not isinstance(new, str):
            raise AgentError("The edit_file action needs 'old' and 'new' as strings.")
        if not old:
            raise AgentError("The edit_file action needs a non-empty 'old' snippet to replace.")
    if name == "write_file":
        if "content" not in data:
            raise AgentError("The write_file action needs 'content' (the full file text).")
        if not isinstance(content, str):
            raise AgentError("The write_file action needs 'content' as a string.")
    return AgentAction(
        name=name,
        path=path,
        content=content if isinstance(content, str) else "",
        summary=str(data.get("summary", "")).strip(),
        query=query,
        pattern=pattern,
        old=old if isinstance(old, str) else "",
        new=new if isinstance(new, str) else "",
        replace_all=bool(data.get("replace_all", False)),
    )


class CodingAgent:
    """Drives one model through a bounded edit loop inside a workspace."""

    def __init__(
        self,
        studio: CodeStudio,
        client: CoderClient,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        read_chars: int = DEFAULT_READ_CHARS,
        temperature: float = 0.2,
        check_command: tuple[str, ...] | None = None,
        allow_checks: bool = True,
    ) -> None:
        self.studio = studio
        self.client = client
        self.max_steps = max_steps
        self.read_chars = read_chars
        self.temperature = temperature
        self.check_command = check_command
        self.allow_checks = allow_checks

    def _find_files(self, workspace: Workspace, pattern: str) -> str:
        """Files matching a glob, confined to the workspace and its editable set."""
        text = pattern.strip()
        if not text:
            return "ERROR: the find_files action needs a 'pattern'."
        # A pattern is a *name* filter, never an escape hatch: refuse anything that
        # climbs out of the workspace before it is ever expanded.
        if text.startswith(("/", "~")) or ".." in Path(text).parts:
            return "ERROR: the pattern must be relative to the workspace root."
        root = workspace.path
        matches: list[str] = []
        for path in sorted(root.glob(text)):
            if len(matches) >= 200:
                break
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(part in SKIP_DIRS for part in relative.parts):
                continue
            try:
                self.studio._check_suffix(path)
            except StudioError:
                continue
            matches.append(relative.as_posix())
        return "\n".join(matches) or f"(no files match {text})"

    def _observe(self, workspace: Workspace, action: AgentAction) -> str:
        """Run one action inside the workspace and return the text fed back."""
        if action.name == "list_files":
            entries = self.studio.list_files(workspace)
            return "\n".join(entry.path for entry in entries) or "(no files)"
        if action.name == "find_files":
            return self._find_files(workspace, action.pattern)
        if action.name == "search":
            try:
                hits = self.studio.search(
                    workspace, action.query, relative=action.path, context_lines=1
                )
            except StudioError as error:
                return f"ERROR: {error}"
            if not hits:
                return f"(no matches for {action.query!r})"
            return "\n--\n".join(hit.render() for hit in hits)
        if action.name == "read_file":
            try:
                text = self.studio.read_file(workspace, action.path)
            except StudioError as error:
                return f"ERROR: {error}"
            if len(text) > self.read_chars:
                return text[: self.read_chars] + "\n… [file truncated]"
            return text
        if action.name == "edit_file":
            try:
                made = self.studio.edit_file(
                    workspace, action.path, action.old, action.new, replace_all=action.replace_all
                )
            except StudioError as error:
                return f"ERROR: {error}"
            return f"Replaced {made} occurrence(s) in {action.path}."
        if action.name == "write_file":
            try:
                written = self.studio.write_file(workspace, action.path, action.content)
            except StudioError as error:
                return f"ERROR: {error}"
            return f"Wrote {written} bytes to {action.path}."
        if action.name == "diff":
            text = self.studio.diff(workspace)
            if len(text) > self.read_chars:
                return text[: self.read_chars] + "\n… [diff truncated]"
            return text
        if action.name == "run_checks":
            if not self.allow_checks:
                return "ERROR: running the checks is disabled for this run."
            result = self.studio.run_checks(workspace, self.check_command)
            tail = result.output.strip()[-4000:]
            return f"{result.summary()}\n{tail}".strip()
        return ""

    @staticmethod
    def _checks_ok(result: str) -> bool:
        """Whether a ``run_checks`` observation reports the checks as passed.

        The observation begins with the check's own summary, so the outcome is read
        from that line -- not from the ``ERROR:`` prefix, which a *failing test run*
        would not carry (a failed run is a normal, truthful observation).
        """
        first = result.splitlines()[0] if result.strip() else ""
        return first.startswith("Checks passed")

    def run(
        self,
        workspace: Workspace,
        task: str,
        *,
        context: str = "",
        on_step: Callable[[AgentStep], None] | None = None,
    ) -> AgentTrace:
        """Carry out ``task`` in ``workspace``, returning the full trace.

        ``context`` is extra, optional background (for example the workspace's
        current diff) placed in the first message. ``on_step``, when given, is
        called with each :class:`AgentStep` as it happens, so a caller can show the
        model's work live instead of only when the run ends.
        """
        if not task.strip():
            raise AgentError("The task cannot be empty.")
        trace = AgentTrace(task=task.strip())
        messages: list[CoderMessage] = [CoderMessage("system", AGENT_SYSTEM_PROMPT)]
        first = f"Task: {task.strip()}"
        if context.strip():
            first += f"\n\nContext:\n{context.strip()}"
        messages.append(CoderMessage("user", first))

        def report(step: AgentStep) -> None:
            trace.steps.append(step)
            if on_step is not None:
                on_step(step)

        for index in range(1, self.max_steps + 1):
            try:
                reply = self.client.chat(messages, temperature=self.temperature)
            except CoderUnavailable as error:
                trace.stopped_reason = f"The coding model could not answer: {error}"
                return trace

            try:
                action = parse_action(reply)
            except AgentError as error:
                report(AgentStep(index, "invalid", detail=str(error), result="asked again", ok=False))
                messages.append(CoderMessage("assistant", reply))
                messages.append(
                    CoderMessage(
                        "user",
                        f"That was not valid: {error} Reply with one JSON object only.",
                    )
                )
                continue

            if action.name == "done":
                trace.summary = action.summary or "The model reported the task complete."
                report(AgentStep(index, "done", detail=trace.summary))
                return trace

            result = self._observe(workspace, action)
            ok = not result.startswith("ERROR:")
            if action.name in {"write_file", "edit_file"} and ok:
                if action.path not in trace.writes:
                    trace.writes.append(action.path)
            if action.name == "run_checks" and self.allow_checks:
                trace.checked = True
                trace.checks_ok = ok and self._checks_ok(result)
            report(AgentStep(index, action.name, detail=action.describe(), result=result, ok=ok))
            messages.append(CoderMessage("assistant", reply))
            messages.append(CoderMessage("user", result))

        trace.stopped_reason = (
            f"Reached the {self.max_steps}-step limit without the model finishing. "
            "The changes made so far are in the workspace; review them and run again."
        )
        return trace


__all__ = [
    "AGENT_SYSTEM_PROMPT",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_READ_CHARS",
    "KNOWN_ACTIONS",
    "PATH_ACTIONS",
    "AgentAction",
    "AgentError",
    "AgentStep",
    "AgentTrace",
    "CodingAgent",
    "parse_action",
]
