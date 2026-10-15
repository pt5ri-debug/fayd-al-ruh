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

from dataclasses import dataclass
from pathlib import Path

from core.code_agent import AgentError, AgentTrace, CodingAgent
from core.code_studio import (
    CheckResult,
    CodeStudio,
    FileEntry,
    StudioError,
    Workspace,
    default_check_command,
)
from core.coder_model import CoderClient, CoderStatus, CoderUnavailable, coder_status


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


class CodeStudioService:
    """Drives the Studio's workspaces, editor, agent, checks, and merge.

    ``client`` is injectable so a test (or a user with a different local runtime)
    can supply the coding model; when it is omitted, the default Ollama client for
    ``qwen2.5-coder:7b`` is built lazily on first use, so constructing the service
    never touches the network.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        studio: CodeStudio | None = None,
        client: CoderClient | None = None,
        check_command: tuple[str, ...] | None = None,
        max_steps: int = 12,
    ) -> None:
        self.studio = studio or CodeStudio(root)
        self._client = client
        self.check_command = check_command or default_check_command(root)
        self.max_steps = max_steps

    # --- model ----------------------------------------------------------
    @property
    def client(self) -> CoderClient:
        """The coding model client, built on first use (never at construction)."""
        if self._client is None:
            from core.coder_model import OllamaCoderClient

            self._client = OllamaCoderClient()
        return self._client

    def status(self) -> CoderStatus:
        """An honest report of whether the coding model can answer."""
        return coder_status(self.client)

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

    def diff(self, slug: str) -> StudioReport:
        """The workspace's change as a unified diff."""
        try:
            text = self.studio.diff(self.workspace(slug))
        except StudioError as error:
            return StudioReport(False, str(error))
        return StudioReport(True, "Diff ready.", detail=text)

    # --- agent ----------------------------------------------------------
    def run_agent(self, slug: str, task: str, *, use_diff_context: bool = True) -> tuple[StudioReport, AgentTrace | None]:
        """Ask the coding model to carry out ``task`` in the workspace.

        Returns the report *and* the trace, so the window can show every step the
        model took. A missing model is reported honestly instead of failing.
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

        agent = CodingAgent(self.studio, self.client, max_steps=self.max_steps)
        try:
            trace = agent.run(workspace, task, context=context)
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


__all__ = [
    "CodeStudioService",
    "StudioReport",
    "StudioState",
]
