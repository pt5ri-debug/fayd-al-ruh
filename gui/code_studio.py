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
from core.coder_llama import (
    BackendStatus,
    LlamaCppCoderClient,
    coder_backend_status,
)
from core.coder_model import (
    DEFAULT_CODER_MODEL,
    CoderClient,
    CoderStatus,
    CoderUnavailable,
    coder_status,
)
from core.studio_files import (
    BatchReport,
    FileEngine,
    NoteScan,
    OutlineReport,
    TreeEntry,
)
from core.studio_git import (
    BranchInfo,
    CommitInfo,
    ConflictReport,
    GitEngine,
    GitError,
    RemoteInfo,
    StashEntry,
    StatusEntry,
)
from core.studio_tasks import (
    CommandOutcome,
    CommandRunner,
    CommandSpec,
    ProjectProfile,
    TaskError,
    catalog_for,
    custom_spec,
    detect_project,
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
    #: What kind of checkout this is (python/node/rust/...), for the task menu.
    project: ProjectProfile | None = None
    #: The current branch of the main checkout, and whether it is clean.
    current_branch: str = ""
    dirty: bool = False


@dataclass(frozen=True)
class ChangeResult:
    """One whole chat turn: the model's run *and* the checks that follow it.

    The window shows a request as a chat turn, so the service runs the agent and
    then the project's checks as one unit and hands back everything the window
    needs to decide whether to ask the user to apply the change. ``ready_to_apply``
    is the honest signal: there is a change, and the checks passed.
    """

    report: StudioReport
    trace: AgentTrace | None = None
    checks: StudioReport | None = None
    changed: tuple[str, ...] = ()

    @property
    def ready_to_apply(self) -> bool:
        """True when there is a real change and the checks passed."""
        return bool(self.changed) and self.checks is not None and self.checks.ok


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
        runner: CommandRunner | None = None,
    ) -> None:
        self.root = Path(root)
        self.studio = studio or CodeStudio(root)
        self._client = client
        self._explicit_client = client is not None
        self.check_command = check_command or default_check_command(root)
        self.max_steps = max_steps
        self.prefer_in_process = prefer_in_process
        # The three engines the window and the CLI both drive: git, the file
        # tree, and the project's own commands. Each is a thin wrapper, built
        # once, so no surface re-implements the logic.
        self.git = GitEngine(self.root)
        self.files = FileEngine(self.studio)
        self.runner = runner or CommandRunner()

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
            project=detect_project(self.root),
            current_branch=self.git.current_branch() if repository else "",
            dirty=(not self.git.is_clean()) if repository else False,
        )

    # --- workspaces -----------------------------------------------------
    def create_workspace(self, task: str = "", *, slug: str | None = None) -> StudioReport:
        """Create a new isolated workspace for ``task``.

        The name is assigned automatically (``PR1``, ``PR2``, ...) unless ``slug``
        is given, so the caller never has to invent one. ``task`` may be empty when
        the workspace is opened ahead of the first request.
        """
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

    def run_change(
        self,
        slug: str,
        task: str,
        *,
        on_step: Callable[[AgentStep], None] | None = None,
        on_checks: Callable[[], None] | None = None,
    ) -> ChangeResult:
        """Run one chat turn: the model edits, then the project's checks run.

        This is the whole flow the window's composer drives. The model's steps are
        streamed through ``on_step``; ``on_checks`` is called once the model has
        finished and the checks are about to run, so the window can say so. The
        result carries ``ready_to_apply``, the honest gate for asking the user to
        apply the change to the real project.
        """
        report, trace = self.run_agent(slug, task, on_step=on_step)
        try:
            changed = tuple(self.changed_files(slug))
        except StudioError:
            changed = ()
        if not changed:
            return ChangeResult(report=report, trace=trace, checks=None, changed=())

        if on_checks is not None:
            on_checks()
        checks_report, _result = self.run_checks(slug)
        return ChangeResult(
            report=report, trace=trace, checks=checks_report, changed=changed
        )

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

    # --- git (per workspace) --------------------------------------------
    def _workspace_git(self, slug: str) -> GitEngine:
        """A git engine pointed at one workspace's worktree."""
        return GitEngine(self.workspace(slug).path)

    def git_status(self, slug: str) -> tuple[StatusEntry, ...]:
        """The workspace's ``git status``, as structured entries."""
        try:
            return self._workspace_git(slug).status()
        except (StudioError, GitError):
            return ()

    def git_log(self, slug: str, *, limit: int = 30, path: str = "") -> tuple[CommitInfo, ...]:
        """The workspace's recent commits, newest first."""
        try:
            return self._workspace_git(slug).log(limit=limit, path=path)
        except (StudioError, GitError):
            return ()

    def git_branches(self, slug: str, *, include_remote: bool = True) -> tuple[BranchInfo, ...]:
        """The branches in the workspace's repository, with tracking state."""
        try:
            return self._workspace_git(slug).branches(include_remote=include_remote)
        except (StudioError, GitError):
            return ()

    def git_remotes(self, slug: str) -> tuple[RemoteInfo, ...]:
        """The workspace's configured remotes, each URL sanitized of any credential."""
        try:
            return self._workspace_git(slug).remotes()
        except (StudioError, GitError):
            return ()

    def git_stash_list(self, slug: str) -> tuple[StashEntry, ...]:
        try:
            return self._workspace_git(slug).stash_list()
        except (StudioError, GitError):
            return ()

    def git_conflicts(self, slug: str) -> ConflictReport:
        try:
            return self._workspace_git(slug).conflicts()
        except (StudioError, GitError):
            return ConflictReport()

    def git_show(self, slug: str, ref: str = "HEAD", *, stat: bool = True) -> StudioReport:
        try:
            text = self._workspace_git(slug).show(ref, stat=stat)
        except (StudioError, GitError) as error:
            return StudioReport(False, str(error))
        return StudioReport(True, f"Showing {ref}.", detail=text)

    def git_blame(self, slug: str, path: str) -> StudioReport:
        try:
            text = self._workspace_git(slug).blame(path)
        except (StudioError, GitError) as error:
            return StudioReport(False, str(error))
        return StudioReport(True, f"Blame for {path}.", detail=text)

    def stage(self, slug: str, *paths: str) -> StudioReport:
        """Stage files (or everything) in the workspace."""
        try:
            result = self._workspace_git(slug).stage(*paths)
        except (StudioError, GitError) as error:
            return StudioReport(False, str(error))
        target = ", ".join(paths) if paths else "everything"
        return StudioReport(result.ok, f"Staged {target}." if result.ok else result.summary())

    def unstage(self, slug: str, *paths: str) -> StudioReport:
        try:
            result = self._workspace_git(slug).unstage(*paths)
        except (StudioError, GitError) as error:
            return StudioReport(False, str(error))
        target = ", ".join(paths) if paths else "everything"
        return StudioReport(result.ok, f"Unstaged {target}." if result.ok else result.summary())

    def commit(self, slug: str, message: str) -> StudioReport:
        """Commit the workspace's staged changes (author supplied if unset)."""
        try:
            result = self._workspace_git(slug).commit(message)
        except (StudioError, GitError) as error:
            return StudioReport(False, str(error))
        if result.ok:
            self.studio._invalidate_checks(self.workspace(slug))
            return StudioReport(True, f"Committed: {result.stdout.splitlines()[0] if result.stdout else message}")
        return StudioReport(False, result.summary())

    def discard_changes(self, slug: str, *paths: str) -> StudioReport:
        """Discard worktree changes (``git restore``) in the workspace."""
        try:
            result = self._workspace_git(slug).restore(*paths)
            self.studio._invalidate_checks(self.workspace(slug))
        except (StudioError, GitError) as error:
            return StudioReport(False, str(error))
        target = ", ".join(paths) if paths else "all files"
        return StudioReport(result.ok, f"Discarded changes in {target}." if result.ok else result.summary())

    def stash(self, slug: str, message: str = "", *, include_untracked: bool = False) -> StudioReport:
        try:
            result = self._workspace_git(slug).stash(message, include_untracked=include_untracked)
        except (StudioError, GitError) as error:
            return StudioReport(False, str(error))
        return StudioReport(result.ok, result.stdout.strip() or result.summary())

    def stash_pop(self, slug: str, index: int = 0) -> StudioReport:
        try:
            result = self._workspace_git(slug).stash_pop(index)
        except (StudioError, GitError) as error:
            return StudioReport(False, str(error))
        return StudioReport(result.ok, result.stdout.strip() or result.summary())

    def create_branch(self, slug: str, name: str, *, checkout: bool = True) -> StudioReport:
        """Create a branch in the workspace's repository."""
        try:
            result = self._workspace_git(slug).create_branch(name, checkout=checkout)
        except (StudioError, GitError) as error:
            return StudioReport(False, str(error))
        return StudioReport(result.ok, f"Created branch `{name}`." if result.ok else result.summary())

    def switch_branch(self, slug: str, name: str) -> StudioReport:
        try:
            result = self._workspace_git(slug).switch_branch(name)
        except (StudioError, GitError) as error:
            return StudioReport(False, str(error))
        return StudioReport(result.ok, f"Switched to `{name}`." if result.ok else result.summary())

    def apply_patch(self, slug: str, patch_text: str, *, check: bool = False) -> StudioReport:
        """Apply a unified diff to the workspace (a dry run with ``check``)."""
        try:
            result = self._workspace_git(slug).apply_patch(patch_text, check=check)
        except (StudioError, GitError) as error:
            return StudioReport(False, str(error))
        return StudioReport(result.ok, result.message, detail=result.detail)

    def export_patch(self, slug: str) -> StudioReport:
        """The workspace's whole change as a patch that can be saved and re-applied."""
        try:
            text = self._workspace_git(slug).diff()
        except (StudioError, GitError) as error:
            return StudioReport(False, str(error))
        return StudioReport(True, "Patch ready.", detail=text)

    def remote_action(self, slug: str, action: str, *, remote: str = "origin") -> StudioReport:
        """Fetch/pull/push the workspace's repository.

        The caller must have passed the approval gate first: this method only
        performs the action, so a network call can never happen by accident.
        """
        try:
            engine = self._workspace_git(slug)
            if action == "fetch":
                result = engine.fetch(remote=remote)
            elif action == "pull":
                result = engine.pull(remote=remote)
            elif action == "push":
                result = engine.push(remote=remote, branch=engine.current_branch())
            else:
                return StudioReport(False, f"Unknown remote action {action!r}.")
        except (StudioError, GitError) as error:
            return StudioReport(False, str(error))
        return StudioReport(result.ok, f"{action} {remote}." if result.ok else result.summary())

    # --- files (per workspace) ------------------------------------------
    def file_tree(self, slug: str, *, relative: str = "") -> tuple[TreeEntry, ...]:
        """The workspace's nested editable-file tree."""
        try:
            return self.files.tree(self.workspace(slug), relative=relative)
        except (StudioError, GitError):
            return ()

    def create_file(self, slug: str, relative: str, content: str = "") -> StudioReport:
        try:
            written = self.files.create_file(self.workspace(slug), relative, content)
        except StudioError as error:
            return StudioReport(False, str(error))
        return StudioReport(True, f"Created {relative} ({written} bytes).")

    def create_directory(self, slug: str, relative: str) -> StudioReport:
        try:
            self.files.create_directory(self.workspace(slug), relative)
        except StudioError as error:
            return StudioReport(False, str(error))
        return StudioReport(True, f"Created folder {relative}.")

    def rename_file(self, slug: str, relative: str, new_relative: str) -> StudioReport:
        try:
            moved = self.files.rename(self.workspace(slug), relative, new_relative)
        except StudioError as error:
            return StudioReport(False, str(error))
        return StudioReport(True, f"Renamed {relative} to {moved}.")

    def copy_file(self, slug: str, relative: str, new_relative: str) -> StudioReport:
        try:
            made = self.files.copy(self.workspace(slug), relative, new_relative)
        except StudioError as error:
            return StudioReport(False, str(error))
        return StudioReport(True, f"Copied {relative} to {made}.")

    def delete_file(self, slug: str, relative: str, *, directory: bool = False) -> StudioReport:
        try:
            if directory:
                removed = self.files.delete_directory(self.workspace(slug), relative)
                return StudioReport(True, f"Deleted folder {relative} ({removed} file(s)).")
            self.files.delete(self.workspace(slug), relative)
        except StudioError as error:
            return StudioReport(False, str(error))
        return StudioReport(True, f"Deleted {relative}.")

    def replace_across(
        self, slug: str, old: str, new: str, *, relative: str = "", apply: bool = False
    ) -> BatchReport:
        """Replace a literal string across the workspace, previewing by default."""
        try:
            return self.files.replace_across(
                self.workspace(slug), old, new, relative=relative, apply=apply
            )
        except StudioError as error:
            return BatchReport(False, str(error))

    def outline(self, slug: str, relative: str) -> OutlineReport:
        try:
            return self.files.outline(self.workspace(slug), relative)
        except StudioError as error:
            return OutlineReport(relative, error=str(error))

    def scan_notes(self, slug: str, *, relative: str = "") -> NoteScan:
        try:
            return self.files.scan_notes(self.workspace(slug), relative=relative)
        except StudioError:
            return NoteScan()

    # --- tasks (project commands) ---------------------------------------
    def project_profile(self) -> ProjectProfile:
        return detect_project(self.root)

    def command_catalog(self) -> tuple[CommandSpec, ...]:
        """The development commands that fit this project."""
        return catalog_for(self.root, check_command=self.check_command)

    def run_command(
        self,
        slug: str,
        key: str,
        *,
        on_line: Callable[[str], None] | None = None,
        command_line: str = "",
    ) -> CommandOutcome:
        """Run one catalog command (or a custom one) inside the workspace.

        The command runs in the workspace's own directory, so a test or a lint
        sees the change and not the main checkout. A custom command is parsed
        with :func:`shlex.split` and never handed to a shell.
        """
        if key == "custom":
            spec = custom_spec(command_line)
        else:
            found = next((s for s in self.command_catalog() if s.key == key), None)
            if found is None:
                raise TaskError(f"No task named {key!r} fits this project.")
            spec = found
        return self.runner.run(spec, cwd=self.workspace(slug).path, on_line=on_line)

    def cancel_command(self) -> None:
        """Stop the running command, if any."""
        self.runner.cancel()

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
    "BatchReport",
    "BranchInfo",
    "ChangeResult",
    "CodeStudioService",
    "CommandOutcome",
    "CommandSpec",
    "CommitInfo",
    "ConflictReport",
    "NoteScan",
    "OutlineReport",
    "ProjectProfile",
    "RemoteInfo",
    "StashEntry",
    "StatusEntry",
    "StudioReport",
    "StudioState",
    "TreeEntry",
]
