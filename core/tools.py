"""A real, sandboxed tool layer.

The cognitive core can reason but cannot *do* anything outside memory. This module
gives it a small set of real tools -- read a file, write a file, list a directory,
search text, do arithmetic -- and, more importantly, a sandbox that every call
must pass through.

The sandbox is deny-by-default and enforces four things:

1. **Allow-list.** Only tools the policy names may run.
2. **Root confinement.** Every path is resolved under one root directory; a path
   that escapes it (``..``, an absolute path elsewhere, a symlink out) is refused.
3. **Read/write separation.** A read-only policy refuses any mutating tool.
4. **Capability + approval gates.** The project's :class:`core.permissions.PermissionPolicy`
   decides whether a capability is granted, and external tools additionally need a
   live human approval (:class:`core.human_approval.HumanApprovalGate`).

Tools return a :class:`ToolResult`; they never raise into the caller, so an agent
loop can always continue and record what happened.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from core.human_approval import HumanApprovalGate
from core.permissions import Capability, PermissionPolicy

Handler = Callable[..., str]


class ToolError(Exception):
    """Raised inside a tool; caught by the sandbox and returned as a failed result."""


@dataclass(frozen=True)
class ToolSpec:
    """A named, described tool bound to a handler."""

    name: str
    description: str
    handler: Handler
    capability: Capability = Capability.SANDBOX_TASK
    mutates: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Tool name cannot be empty.")
        if not self.description.strip():
            raise ValueError(f"Tool '{self.name}' needs a description.")


@dataclass(frozen=True)
class ToolResult:
    """The outcome of one sandboxed tool call."""

    tool: str
    ok: bool
    output: str = ""
    error: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


class ToolRegistry:
    """A name -> tool map."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._tools:
            raise ValueError(f"Tool '{spec.name}' is already registered.")
        self._tools[spec.name] = spec
        return spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._tools[name] for name in self.names())

    def __len__(self) -> int:
        return len(self._tools)


@dataclass(frozen=True)
class SandboxPolicy:
    """Deny-by-default rules for what a sandbox may run and touch."""

    root: Path
    allowed_tools: frozenset[str]
    read_only: bool = True
    max_output_bytes: int = 16_384
    allowed_suffixes: frozenset[str] | None = None
    max_file_bytes: int = 262_144

    def __post_init__(self) -> None:
        if self.max_output_bytes < 1:
            raise ValueError("max_output_bytes must be at least 1.")
        if self.max_file_bytes < 1:
            raise ValueError("max_file_bytes must be at least 1.")


class ToolSandbox:
    """Runs tools against a registry only after the policy and gates allow it."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: SandboxPolicy,
        *,
        permissions: PermissionPolicy | None = None,
        approvals: HumanApprovalGate | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.permissions = permissions or PermissionPolicy()
        self.approvals = approvals or HumanApprovalGate()

    def run(self, name: str, **arguments) -> ToolResult:
        spec = self.registry.get(name)
        if spec is None:
            return ToolResult(name, False, error=f"Unknown tool '{name}'.")
        if name not in self.policy.allowed_tools:
            return ToolResult(name, False, error=f"Tool '{name}' is not on the allow-list.")
        if spec.mutates and self.policy.read_only:
            return ToolResult(name, False, error=f"Tool '{name}' mutates and the sandbox is read-only.")

        decision = self.permissions.decide(spec.capability, name)
        if not decision.allowed:
            return ToolResult(name, False, error=decision.reason)
        if spec.capability is Capability.EXTERNAL_TOOL:
            approval = self.approvals.decide(name)
            if not approval.approved:
                return ToolResult(name, False, error=approval.reason)

        try:
            output = spec.handler(**arguments)
        except ToolError as error:
            return ToolResult(name, False, error=str(error))
        except Exception as error:  # a tool must never crash the caller
            return ToolResult(name, False, error=f"Tool '{name}' failed: {error}")

        encoded = output.encode("utf-8")
        if len(encoded) > self.policy.max_output_bytes:
            output = encoded[: self.policy.max_output_bytes].decode("utf-8", errors="ignore")
            return ToolResult(
                name, True, output=output, metadata={"truncated": "true"}
            )
        return ToolResult(name, True, output=output)

    # --- path safety ----------------------------------------------------
    def resolve_path(self, relative: str) -> Path:
        """Resolve ``relative`` under the sandbox root, refusing any escape."""
        if not relative.strip():
            raise ToolError("Path cannot be empty.")
        candidate = (self.policy.root / relative).resolve()
        root = self.policy.root.resolve()
        if candidate != root and root not in candidate.parents:
            raise ToolError(f"Path '{relative}' escapes the sandbox root.")
        return candidate

    def check_suffix(self, path: Path) -> None:
        allowed = self.policy.allowed_suffixes
        if allowed is not None and path.suffix.casefold() not in allowed:
            raise ToolError(f"File type '{path.suffix}' is not allowed.")


def _read_file(sandbox: ToolSandbox, relative: str) -> str:
    path = sandbox.resolve_path(relative)
    sandbox.check_suffix(path)
    if not path.is_file():
        raise ToolError(f"Not a file: {relative}")
    if path.stat().st_size > sandbox.policy.max_file_bytes:
        raise ToolError(f"File '{relative}' exceeds the size limit.")
    return path.read_text(encoding="utf-8", errors="replace")


def _write_file(sandbox: ToolSandbox, relative: str, content: str) -> str:
    path = sandbox.resolve_path(relative)
    sandbox.check_suffix(path)
    encoded = content.encode("utf-8")
    if len(encoded) > sandbox.policy.max_file_bytes:
        raise ToolError("Content exceeds the size limit.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Wrote {len(encoded)} bytes to {relative}"


def _list_dir(sandbox: ToolSandbox, relative: str = ".") -> str:
    path = sandbox.resolve_path(relative)
    if not path.is_dir():
        raise ToolError(f"Not a directory: {relative}")
    entries = sorted(
        f"{item.name}/" if item.is_dir() else item.name for item in path.iterdir()
    )
    return "\n".join(entries) if entries else "(empty)"


def _search(sandbox: ToolSandbox, query: str, relative: str = ".") -> str:
    if not query.strip():
        raise ToolError("Search query cannot be empty.")
    root = sandbox.resolve_path(relative)
    matches: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        sandbox.check_suffix(path)
        if path.stat().st_size > sandbox.policy.max_file_bytes:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if query in line:
                matches.append(f"{path.relative_to(sandbox.policy.root)}:{number}: {line.strip()}")
                if len(matches) >= 50:
                    return "\n".join(matches)
    return "\n".join(matches) if matches else "(no matches)"


_OPERATORS: dict[type, Callable[..., float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ToolError("Only arithmetic expressions are allowed.")


def _calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ToolError(f"Invalid expression: {error}") from error
    value = _evaluate(tree)
    return str(int(value)) if value.is_integer() else str(value)


def build_sandbox_tools(sandbox: "ToolSandbox") -> tuple[ToolSpec, ...]:
    """The built-in file/search/calculator tools, bound to a sandbox root."""

    def bind(handler: Callable[..., str]) -> Handler:
        def wrapped(**kwargs) -> str:
            return handler(sandbox, **kwargs)

        return wrapped

    return (
        ToolSpec(
            "read_file",
            "Read a UTF-8 text file inside the sandbox root.",
            bind(_read_file),
            Capability.SANDBOX_TASK,
        ),
        ToolSpec(
            "write_file",
            "Write a UTF-8 text file inside the sandbox root.",
            bind(_write_file),
            Capability.SANDBOX_TASK,
            mutates=True,
        ),
        ToolSpec(
            "list_dir",
            "List the entries of a directory inside the sandbox root.",
            bind(_list_dir),
            Capability.SANDBOX_TASK,
        ),
        ToolSpec(
            "search",
            "Search for a literal string in files inside the sandbox root.",
            bind(_search),
            Capability.SANDBOX_TASK,
        ),
        ToolSpec(
            "calculator",
            "Evaluate a safe arithmetic expression.",
            lambda expression: _calculator(expression),
            Capability.SYMBOLIC_SIMULATION,
        ),
    )


def build_sandbox(
    root,
    *,
    read_only: bool = True,
    allowed_tools: Iterable[str] | None = None,
    permissions: PermissionPolicy | None = None,
    approvals: HumanApprovalGate | None = None,
    allowed_suffixes: frozenset[str] | None = None,
) -> ToolSandbox:
    """Assemble a sandbox with the built-in tools and a deny-by-default policy."""
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    registry = ToolRegistry()
    names = frozenset(allowed_tools) if allowed_tools is not None else frozenset(
        {"read_file", "write_file", "list_dir", "search", "calculator"}
    )
    policy = SandboxPolicy(
        root=root_path,
        allowed_tools=names,
        read_only=read_only,
        allowed_suffixes=allowed_suffixes,
    )
    sandbox = ToolSandbox(
        registry, policy, permissions=permissions, approvals=approvals
    )
    for spec in build_sandbox_tools(sandbox):
        if spec.name in names:
            registry.register(spec)
    return sandbox


__all__ = [
    "SandboxPolicy",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolSandbox",
    "ToolSpec",
    "build_sandbox",
    "build_sandbox_tools",
]
