"""The Code Studio service: the Tk-free brain behind the Studio window.

Everything the Studio does -- make a workspace, read and write files, run the
coding model, run the checks, and merge -- lives here, so the window and the CLI
share one implementation and the logic is testable with no display. This mirrors
the split the rest of the project already uses (``gui/import_center.py`` beside
``gui/import_dialog.py``, ``gui/sql_import.py`` beside ``gui/sql_import_dialog.py``).

The service never merges on its own and never edits the main checkout: it only
drives :class:`core.code_studio.CodeStudio` and :class:`core.code_agent.CodingAgent`,
both of which are confined to a workspace. A merge happens only when the caller
asks for one and the gates in the engine allow it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.code_agent import AgentError, AgentStep, AgentTrace, CodingAgent
from core.code_studio import (
    CheckResult,
    CodeStudio,
    FileEntry,
    StudioError,
    Workspace,
    default_check_command,
)
from core.coder_model import (
    DEFAULT_CODER_MODEL,
    CoderClient,
    CoderStatus,
    CoderUnavailable,
    coder_status,
)
from core.coder_llama import (
    BackendStatus,
    LlamaCppCoderClient,
    coder_backend_status,
)


@dataclass(frozen=True)
class StudioReport:
    """The outcome of one service call, for the window's transcript."""

    ok: bool
    message: str
    detail: str = ""


@dataclass
class StudioState:
    """A snapshot the window renders: what exists and what the model can do."""

    workspaces: tuple[Workspace, ...] = ()
    coder: CoderStatus | None = None
    repository: bool = False
    base_branch: str = "main"
    base_revision: str = ""
    backend: BackendStatus | None = None


class CodeStudioService:
    """Drives the Studio's workspaces, editor, agent, checks, and merge.

    ``client`` is injectable so a test (or a user with a different local runtime)
    can supply the coding model. When it is omitted, an **in-process** client is
    built lazily on first use -- a GGUF file loaded in this process, so no
    ``ollama serve`` daemon has to sit in the background holding RAM. A user who
    already runs Ollama is still served: :meth:`resolve_client` falls back to it
    when no in-process model is ready, so an existing setup keeps working.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        studio: CodeStudio | None = None,
        client: CoderClient | None = None,
        check_command: tuple[str, ...] | None = None,
        max_steps: int = 12,
        prefer_in_process: bool = True,
    ) -> None:
        self.studio = studio or CodeStudio(root)
        self._client = client
        self._explicit_client = client is not None
        self.check_command = check_command or default_check_command(root)
        self.max_steps = max_steps
        self.prefer_in_process = prefer_in_process

    # --- model ----------------------------------------------------------
    def resolve_client(self) -> CoderClient:
        """The coding model client, built on first use (never at construction).

        The in-process model is preferred; when it cannot run, a reachable Ollama
        server is used instead, so an existing Ollama setup is not discarded.
        """
        if self._client is not None:
            return self._client
        if self.prefer_in_process:
            candidate = LlamaCppCoderClient()
            if candidate.is_installed():
                self._client = candidate
                return self._client
        from core.coder_model import OllamaCoderClient

        self._client = OllamaCoderClient()
        return self._client

    @property
    def client(self) -> CoderClient:
        """The resolved coding model client."""
        return self.resolve_client()

    def backend(self) -> BackendStatus:
        """Which runtime can answer, and the next step when none can."""
        if self._explicit_client:
            status = self.status()
            return BackendStatus("injected", status.available, detail=status.summary())
        return coder_backend_status(probe_ollama=True)

    def status(self) -> CoderStatus:
        """An honest report of whether the coding model can answer.

        The preferred runtime is the **in-process** model, which needs no
        background daemon, so a machine with no Ollama at all is not told to
        start a server it never needed. Ollama is only named when it is the
        runtime that can actually answer, or when the in-process model is not
        ready *and* Ollama is the only thing left to fall back to. The backend is
        resolved once, so this never probes the same server twice.
        """
        if self.prefer_in_process and not self._explicit_client:
            backend = self.backend()
            if backend.backend == "ollama":
                return CoderStatus(
                    available=backend.available,
                    server_reachable=True,
                    model=DEFAULT_CODER_MODEL,
                    installed=backend.available,
                    detail="" if backend.available else backend.detail,
                    hint=backend.hint,
                    backend="ollama",
                )
            # Either the in-process model is the answer, or neither runtime is
            # ready -- and then ``backend.detail`` already names the in-process
            # step first, so Ollama is never presented as the only way.
            return CoderStatus(
                available=backend.available,
                server_reachable=True,
                model=DEFAULT_CODER_MODEL,
                installed=backend.available,
                detail=backend.detail,
                hint=backend.hint,
                backend="in-process",
            )
        return coder_status(self.resolve_client())

    def state(self) -> StudioState:
        """A snapshot of the Studio for the window to render."""
        repository = self.studio.is_repository()
        base_revision = ""
        if repository:
            try:
                base_revision = self.studio.base_revision()
            except StudioError:
                base_revision = ""
        return StudioState(
            workspaces=self.studio.list_workspaces() if repository else (),
            coder=self.status(),
            repository=repository,
            base_branch=self.studio.base_branch,
            base_revision=base_revision,
            backend=self.backend(),
        )

    # --- workspaces -----------------------------------------------------
    def create_workspace(self, task: str, *, slug: str | None = None) -> StudioReport:
        """Create a new isolated workspace for ``task``."""
        try:
            workspace = self.studio.create(task, slug=slug)
        except StudioError as error:
            return StudioReport(False, str(error))
        return StudioReport(
            True,
            f"Created workspace `{workspace.slug}` on `{workspace.branch}` "
            f"from {workspace.base_revision}.",
        )

    def workspace(self, slug: str) -> Workspace:
        return self.studio.get(slug)

    def remove_workspace(self, slug: str) -> StudioReport:
        """Discard a workspace and its branch."""
        try:
            self.studio.remove(slug)
        except StudioError as error:
            return StudioReport(False, str(error))
        return StudioReport(True, f"Discarded workspace `{slug}`.")

    # --- editor ---------------------------------------------------------
    def list_files(self, slug: str) -> tuple[FileEntry, ...]:
        return self.studio.list_files(self.workspace(slug))

    def read_file(self, slug: str, relative: str) -> StudioReport:
        try:
            text = self.studio.read_file(self.workspace(slug), relative)
        except StudioError as error:
            return StudioReport(False, str(error))
        return StudioReport(True, f"Read {relative}.", detail=text)

    def save_file(self, slug: str, relative: str, content: str) -> StudioReport:
        """Write the editor's text into the workspace."""
        try:
            written = self.studio.write_file(self.workspace(slug), relative, content)
        except StudioError as error:
            return StudioReport(False, str(error))
        return StudioReport(True, f"Saved {relative} ({written} bytes).")

    def changed_files(self, slug: str) -> tuple[str, ...]:
        return self.studio.changed_files(self.workspace(slug))

    def search(self, slug: str, query: str, *, relative: str = "") -> StudioReport:
        """Find a literal string in the workspace, for the window's search box."""
        try:
            hits = self.studio.search(self.workspace(slug), query, relative=relative, context_lines=1)
        except StudioError as error:
            return StudioReport(False, str(error))
        if not hits:
            return StudioReport(True, f"No matches for {query!r}.", detail="(no matches)")
        detail = "\n--\n".join(hit.render() for hit in hits)
        return StudioReport(True, f"{len(hits)} match(es) for {query!r}.", detail=detail)

    def edit_file(
        self, slug: str, relative: str, old: str, new: str, *, replace_all: bool = False
    ) -> StudioReport:
        """Replace one exact snippet in a file, without rewriting the whole file."""
        try:
            made = self.studio.edit_file(
                self.workspace(slug), relative, old, new, replace_all=replace_all
            )
        except StudioError as error:
            return StudioReport(False, str(error))
        return StudioReport(True, f"Replaced {made} occurrence(s) in {relative}.")

    def diff(self, slug: str) -> StudioReport:
        """The workspace's change as a unified diff."""
        try:
            text = self.studio.diff(self.workspace(slug))
        except StudioError as error:
            return StudioReport(False, str(error))
        return StudioReport(True, "Diff ready.", detail=text)

    # --- agent ----------------------------------------------------------
    def run_agent(
        self,
        slug: str,
        task: str,
        *,
        use_diff_context: bool = True,
        on_step: Callable[[AgentStep], None] | None = None,
    ) -> tuple[StudioReport, AgentTrace | None]:
        """Ask the coding model to carry out ``task`` in the workspace.

        Returns the report *and* the trace, so the window can show every step the
        model took. ``on_step`` is forwarded to the agent, so a caller can render
        each step live as the run proceeds. A missing model is reported honestly
        instead of failing.
        """
        try:
            workspace = self.workspace(slug)
        except StudioError as error:
            return StudioReport(False, str(error)), None

        status = self.status()
        if not status.available:
            return StudioReport(False, status.summary()), None

        context = ""
        if use_diff_context:
            try:
                current = self.studio.diff(workspace)
                if current and not current.startswith("(no changes"):
                    context = current
            except StudioError:
                context = ""

        agent = CodingAgent(
            self.studio,
            self.resolve_client(),
            max_steps=self.max_steps,
            check_command=self.check_command,
        )
        try:
            trace = agent.run(workspace, task, context=context, on_step=on_step)
        except AgentError as error:
            return StudioReport(False, str(error)), None
        except CoderUnavailable as error:
            return StudioReport(False, f"The coding model could not answer: {error}"), None

        changed = self.studio.changed_files(workspace)
        if trace.ok:
            message = f"The model finished: {trace.summary}"
        else:
            message = trace.stopped_reason or "The model did not finish."
        if changed:
            message += f" Changed files: {', '.join(changed)}."
        return StudioReport(trace.ok, message, detail=trace.render()), trace

    # --- checks and merge -----------------------------------------------
    def run_checks(self, slug: str) -> tuple[StudioReport, CheckResult | None]:
        """Run the project's checks inside the workspace, in the background."""
        try:
            workspace = self.workspace(slug)
        except StudioError as error:
            return StudioReport(False, str(error)), None
        result = self.studio.run_checks(workspace, self.check_command)
        return StudioReport(result.ok, result.summary(), detail=result.output), result

    def can_merge(self, slug: str) -> StudioReport:
        """Whether this workspace may be merged now, with the reason either way."""
        try:
            allowed, reason = self.studio.can_merge(self.workspace(slug))
        except StudioError as error:
            return StudioReport(False, str(error))
        return StudioReport(allowed, reason)

    def merge(self, slug: str) -> StudioReport:
        """Merge the workspace into the base branch, if the gates allow it."""
        try:
            result = self.studio.merge(self.workspace(slug))
        except StudioError as error:
            return StudioReport(False, str(error))
        detail = "\n".join(
            f"$ {' '.join(command.argv)}  (exit {command.returncode})" for command in result.commands
        )
        return StudioReport(result.ok, result.message, detail=detail)

    # --- in-process model management -------------------------------------
    def available_models(self) -> tuple[tuple[str, str, float, str], ...]:
        """The coding models ``download_model`` can fetch, for a menu."""
        from core.coder_llama import spec_summaries

        return spec_summaries()

    def download_model(self, spec_key: str = "", *, approved: bool = False) -> StudioReport:
        """Download a coding model's weights for the in-process runtime.

        Leaves the machine, so it runs only with ``approved=True`` (the window's
        explicit button, or the CLI's ``--approve``).
        """
        from core.coder_llama import DEFAULT_SPEC_KEY, CoderModelUnavailable, download_model

        try:
            path = download_model(spec_key or DEFAULT_SPEC_KEY, approved=approved)
        except CoderModelUnavailable as error:
            return StudioReport(False, str(error))
        return StudioReport(True, f"Coding model ready: {path}")

    def local_model_file(self) -> str:
        """The in-process model file on disk, or an empty string."""
        from core.coder_llama import find_model_file

        found = find_model_file()
        return str(found) if found is not None else ""


__all__ = [
    "CodeStudioService",
    "StudioReport",
    "StudioState",
]