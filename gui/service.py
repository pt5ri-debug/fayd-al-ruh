"""Headless service layer behind the GUI.

The GUI window is a thin shell over this class. Keeping the logic here means it
can be exercised without a display, and it reuses the exact session the text
``chat`` command builds -- same assistant, same learner, same sandbox, same
``ModelSwitcher`` -- so a GUI answer is produced by the same code path a CLI
answer is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai import chat as chat_module
from ai import mind as mind_module
from ai.jarvis import build_session
from core.jarvis import JarvisMode, JarvisTurn
from gui.attachments import Attachment, AttachmentNotice, attach_document, remove_documents


@dataclass
class ChatReply:
    """One rendered assistant reply, ready for the GUI to display."""

    text: str
    citations: tuple[str, ...] = ()
    used_web: bool = False
    learned: bool = False
    mode: str = "sources"
    hint: str = ""
    is_control: bool = False
    steps: list[dict[str, Any]] = field(default_factory=list)


class ChatService:
    """Owns one conversation session and answers questions for the GUI.

    ``source`` selects which mouth answers, mirroring the CLI's switches:

    - ``"sources"`` -- grounded answers with citations (default).
    - ``"model"`` -- answers from the trained language model, no citations.
    """

    def __init__(
        self,
        *,
        mode: JarvisMode | None = None,
        no_web: bool = False,
        no_feedback: bool = False,
        root: Path | None = None,
        read_write: bool = False,
        model_checkpoint: str | None = None,
        max_new_tokens: int = 96,
        session: str | None = None,
    ) -> None:
        self.mode = mode or build_session(
            root=root,
            read_write=read_write,
            no_web=no_web,
            no_feedback=no_feedback,
            text_only=True,
        )
        self.switcher = chat_module.ModelSwitcher(
            self.mode,
            loader=lambda: mind_module.load_mind_model(
                model_checkpoint,
                max_new_tokens=max_new_tokens,
            ),
            max_new_tokens=max_new_tokens,
        )
        self.session = session
        if session and not self.mode.resume_session(session):
            self.session = None
        # Files the user attached to this conversation, kept so they can be listed
        # in the UI and detached again. Their text is already in the assistant's
        # store (see attach_file), so they need no separate retrieval path.
        self.attachments: list[Attachment] = []

    @property
    def source(self) -> str:
        """Which mouth currently answers: ``"model"`` or ``"sources"``."""
        return "model" if self.switcher.active else "sources"

    def use_model(self) -> None:
        """Answer from the trained model. Raises when the checkpoint is missing."""
        self.switcher.use_model()

    def use_sources(self) -> None:
        """Answer from retrieved passages with citations."""
        self.switcher.use_sources()

    def ask(self, message: str) -> ChatReply:
        """Handle one user message and return the rendered reply."""
        turn = self.mode.handle(message, require_wake=False, speak=False)
        return self._to_reply(turn)

    def save(self) -> str:
        """Persist the current session; returns the status line."""
        result = self.mode.save_session()
        if result.startswith("Saved") and self.session is None:
            self.session = self.mode.conversation.id
        return result

    def new_conversation(self) -> str:
        """Start a fresh session and return its id."""
        conversation = self.mode.reset_conversation()
        self.session = None
        return conversation.id

    def list_sessions(self) -> tuple[str, ...]:
        """The ids of saved conversations available to resume."""
        return self.mode.list_sessions()

    def attach_file(self, path: str | Path) -> AttachmentNotice:
        """Read a file, index its text, and remember it as an attachment.

        The document's chunks join the assistant's own store, so a question can
        retrieve and cite them exactly like the built-in corpus. A read failure
        comes back as ``ok=False`` with a message, never as an exception.
        """
        notice = attach_document(path)
        if not notice.ok or not notice.attachments:
            return notice
        retriever = self._retriever()
        if retriever is None:
            return AttachmentNotice(
                filename=notice.filename,
                ok=False,
                message="Attachments need the source assistant, which is not available here.",
            )
        for attachment in notice.attachments:
            for document in attachment.documents:
                try:
                    retriever.index_document(document)
                except (ValueError, OSError) as error:
                    return AttachmentNotice(
                        filename=notice.filename,
                        ok=False,
                        message=f"Could not index the file: {error}",
                    )
            self.attachments.append(attachment)
        return notice

    def attach_document(self, attachment: Attachment) -> None:
        """Index an already-read :class:`Attachment` (used when wiring in tests)."""
        retriever = self._retriever()
        if retriever is None:
            raise ValueError("Attachments need the source assistant, which is not available here.")
        for document in attachment.documents:
            retriever.index_document(document)
        self.attachments.append(attachment)

    def remove_attachment(self, attachment: Attachment) -> int:
        """Detach a file: drop its documents from the store. Returns how many left."""
        retriever = self._retriever()
        removed = remove_documents(retriever, attachment) if retriever is not None else 0
        self.attachments = [item for item in self.attachments if item is not attachment]
        return removed

    def clear_attachments(self) -> int:
        """Detach every file; returns how many were removed."""
        removed = 0
        for attachment in list(self.attachments):
            removed += self.remove_attachment(attachment)
        return removed

    def _retriever(self):
        """The grounded assistant's retriever, when one is wired in.

        Taken from the switcher's grounded assistant rather than the active one,
        so attaching a file works even while the window is answering from the
        trained model -- the evidence is there the moment the user switches back.
        """
        grounded = getattr(self.switcher, "grounded", None)
        return getattr(grounded, "retriever", None)

    def resume(self, session_id: str) -> bool:
        """Make a saved conversation current. Returns whether it was found."""
        if not self.mode.resume_session(session_id):
            return False
        self.session = session_id
        return True

    def _to_reply(self, turn: JarvisTurn) -> ChatReply:
        text = chat_module.format_turn(turn)
        if turn.routed == "control":
            return ChatReply(text=text, mode=self.source, is_control=True)
        hint = ""
        if (
            self.source == "sources"
            and turn.citations
            and chat_module._NO_SOURCE_MARKER in turn.response
        ):
            hint = chat_module.MIND_HINT
        steps = [
            {
                "tool": step.call.tool,
                "ok": step.result.ok,
                "output": step.result.output,
                "error": step.result.error,
            }
            for step in (turn.agent_run.steps if turn.agent_run is not None else [])
        ]
        return ChatReply(
            text=text,
            citations=tuple(turn.citations),
            used_web=bool(turn.used_web),
            learned=bool(turn.learned),
            mode=self.source,
            hint=hint,
            steps=steps,
        )


__all__ = ["ChatReply", "ChatService"]
