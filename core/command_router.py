"""Route a free-language utterance to either the assistant or a tool plan.

The agent layer already runs a *declared* plan; the router is what turns an
ordinary sentence into that plan. It is deliberately conservative:

* every candidate tool must be on the sandbox's allow-list, so a proposal can
  never widen the agent's authority;
* an utterance the router does not confidently understand becomes a normal
  grounded-assistant question, never a guessed tool call;
* a :class:`RoutedCommand` carries the reason it was chosen, so the decision is
  inspectable.

The router recognises English and Arabic intents (list/read/write/search/compute)
and extracts only the arguments it can see in the text. Anything ambiguous falls
back to answering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from core.tool_agent import ToolCall, ToolPlan, plan_from_calls
from core.tools import ToolSandbox


class RouteKind(str, Enum):
    """What the router decided an utterance is."""

    ANSWER = "answer"
    TOOL = "tool"


@dataclass(frozen=True)
class RoutedCommand:
    """The router's decision for one utterance."""

    kind: RouteKind
    utterance: str
    reason: str
    plan: ToolPlan | None = None


_EXPRESSION_CHARACTERS = set("0123456789+-*/(). ")
_OPERATORS = set("+-*/")
_PATH_PATTERN = re.compile(r"[\w./-]+\.\w+|[\w./-]+/")


def _has_operator(expression: str) -> bool:
    """True when the expression has an operator beyond a leading sign."""
    return any(character in _OPERATORS for character in expression.lstrip("+-"))


def _extract_expression(text: str) -> str | None:
    """Return the longest arithmetic span in ``text``, or ``None``.

    We scan for maximal runs of expression characters and keep the longest one
    that contains a digit and an operator, so "calculate 2 * (3 + 4)" yields the
    whole expression rather than stopping before the closing parenthesis.
    """
    best: str | None = None
    current = ""
    for character in text:
        if character in _EXPRESSION_CHARACTERS:
            current += character
            continue
        candidate = current.strip()
        if candidate and any(c.isdigit() for c in candidate) and _has_operator(candidate):
            if best is None or len(candidate) > len(best):
                best = candidate
        current = ""
    candidate = current.strip()
    if candidate and any(c.isdigit() for c in candidate) and _has_operator(candidate):
        if best is None or len(candidate) > len(best):
            best = candidate
    return best


def _extract_path(text: str, *, default: str = ".") -> str:
    match = _PATH_PATTERN.search(text)
    return match.group(0).strip() if match else default


def _after_marker(text: str, markers: tuple[str, ...]) -> str:
    lowered = text.casefold()
    for marker in markers:
        index = lowered.find(marker.casefold())
        if index != -1:
            return text[index + len(marker):].strip(" :؟?\"'")
    return ""


class NaturalLanguageRouter:
    """Turns a sentence into a tool plan, or leaves it to the assistant."""

    def __init__(self, sandbox: ToolSandbox) -> None:
        self.sandbox = sandbox

    @property
    def allowed_tools(self) -> frozenset[str]:
        return self.sandbox.policy.allowed_tools

    def _allowed(self, tool: str) -> bool:
        return tool in self.allowed_tools

    @property
    def _can_write(self) -> bool:
        return self._allowed("write_file") and not self.sandbox.policy.read_only

    def route(self, utterance: str) -> RoutedCommand:
        text = utterance.strip()
        if not text:
            raise ValueError("Cannot route an empty utterance.")
        lowered = text.casefold()

        if self._allowed("calculator") and any(
            keyword in lowered for keyword in ("calculate", "compute", "what is", "احسب", "كم يساوي")
        ):
            expression = _extract_expression(text)
            if expression is not None:
                call = ToolCall("calculator", {"expression": expression})
                return RoutedCommand(
                    RouteKind.TOOL,
                    text,
                    f"arithmetic request: {expression}",
                    plan_from_calls(text, (call,), ("compute",)),
                )

        if self._allowed("search") and any(
            keyword in lowered for keyword in ("search for", "find ", "look for", "ابحث عن", "جد ")
        ):
            query = _after_marker(text, ("search for", "look for", "find", "ابحث عن", "جد"))
            if query:
                call = ToolCall("search", {"query": query, "relative": "."})
                return RoutedCommand(
                    RouteKind.TOOL,
                    text,
                    f"search request: {query!r}",
                    plan_from_calls(text, (call,), ("search",)),
                )

        if self._allowed("read_file") and any(
            keyword in lowered for keyword in ("read file", "read the file", "show me file", "اقرأ الملف", "اعرض الملف")
        ):
            relative = _extract_path(text)
            call = ToolCall("read_file", {"relative": relative})
            return RoutedCommand(
                RouteKind.TOOL,
                text,
                f"read request: {relative}",
                plan_from_calls(text, (call,), ("read",)),
            )

        if self._allowed("list_dir") and any(
            keyword in lowered for keyword in ("list files", "list the files", "list directory", "show files", "اعرض الملفات", "اسرد الملفات")
        ):
            relative = _extract_path(text, default=".")
            call = ToolCall("list_dir", {"relative": relative})
            return RoutedCommand(
                RouteKind.TOOL,
                text,
                f"list request: {relative}",
                plan_from_calls(text, (call,), ("list",)),
            )

        if self._can_write and any(
            keyword in lowered for keyword in ("write to file", "write file", "save to file", "اكتب في الملف", "احفظ في الملف")
        ):
            relative = _extract_path(text)
            content = _after_marker(text, ("write to file", "write file", "save to file", "اكتب في الملف", "احفظ في الملف"))
            content = content.lstrip(":").strip()
            content = content[len(relative):].lstrip(" :").strip() if content.startswith(relative) else content
            if content:
                call = ToolCall("write_file", {"relative": relative, "content": content})
                return RoutedCommand(
                    RouteKind.TOOL,
                    text,
                    f"write request: {relative}",
                    plan_from_calls(text, (call,), ("write",)),
                )

        return RoutedCommand(RouteKind.ANSWER, text, "no tool intent matched; answering from knowledge")


__all__ = ["NaturalLanguageRouter", "RouteKind", "RoutedCommand"]
