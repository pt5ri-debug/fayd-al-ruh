"""Jarvis mode: one loop that listens, understands, acts, and speaks.

This is the layer the earlier phases were building toward. It does not add a new
reasoning core; it *orchestrates* the pieces that already exist:

* the grounded, cited assistant (:mod:`core.assistant`) answers questions;
* the natural-language router (:mod:`core.command_router`) turns a sentence into
  a sandboxed tool plan when -- and only when -- the intent is clear;
* the tool agent (:mod:`core.tool_agent`) runs that plan under every existing
  guarantee (allow-list, root confinement, capability and approval gates);
* speech I/O (:mod:`core.speech`) and the wake-word gate (:mod:`core.wake_word`)
  make it voice-first without ever widening the model's authority;
* user preferences (:mod:`core.preferences`) let it remember how to serve.

The four hard gates of the project stay in force: the sandbox, the groundedness
gate, the explicit-feedback rule, and the regression monitor. Jarvis mode can
refuse, fall back, and say plainly when it cannot do something -- it never lets a
sentence turn into an action the sandbox forbids.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.command_router import NaturalLanguageRouter, RouteKind
from core.conversation import Conversation, ConversationMemory
from core.continuous_learning import ContinuousLearner, Feedback
from core.jarvis_controls import JarvisControl, ControlIntent, JarvisControls
from core.speech import AudioClip, Speaker, VoiceAssistant
from core.tool_agent import AgentRun, ToolAgent
from core.wake_word import AudioSpeechGate, WakeWordDetector


HELP_TEXT = (
    "Conversation controls: say 'stop' to mute my voice, 'speak' to turn it back "
    "on, 'repeat' to hear the last answer again, 'new conversation' to start over, "
    "'save' to persist this session, 'sessions' to list saved ones, 'that's right' "
    "to confirm my last answer, 'that's wrong, it's ...' to correct it, or 'help' "
    "for this list. Anything else, ask me a question or give me a command."
)


@dataclass
class JarvisTurn:
    """The full record of one Jarvis exchange."""

    heard: str
    response: str
    woke: bool
    routed: str
    intent: str = ""
    answer_text: str = ""
    agent_run: AgentRun | None = None
    citations: tuple[str, ...] = field(default_factory=tuple)
    spoken: AudioClip | None = None
    playback_error: str = ""
    conversation_id: str = ""


class JarvisMode:
    """A single conversational loop over the assistant, tools, and voice."""

    def __init__(
        self,
        assistant,
        *,
        router: NaturalLanguageRouter,
        agent: ToolAgent | None = None,
        voice: VoiceAssistant | None = None,
        wake: WakeWordDetector | None = None,
        gate: AudioSpeechGate | None = None,
        speaker: Speaker | None = None,
        memory: ConversationMemory | None = None,
        controls: JarvisControls | None = None,
        learner: ContinuousLearner | None = None,
        mute: bool = False,
    ) -> None:
        self.assistant = assistant
        self.router = router
        self.agent = agent or ToolAgent(router.sandbox)
        self.voice = voice or VoiceAssistant(assistant)
        self.wake = wake or WakeWordDetector()
        self.gate = gate or AudioSpeechGate()
        self.speaker = speaker
        self.memory = memory or ConversationMemory()
        self.controls = controls or JarvisControls()
        self.learner = learner
        self.mute = mute
        self._conversation: Conversation | None = None

    @property
    def conversation(self) -> Conversation:
        """The current session, created on first use so follow-ups keep context."""
        if self._conversation is None:
            self._conversation = self.memory.start("Jarvis session")
        return self._conversation

    def reset_conversation(self) -> Conversation:
        """Start a new memory session, leaving earlier turns behind."""
        self._conversation = self.memory.start("Jarvis session")
        return self._conversation

    def save_session(self) -> str:
        """Persist the current session. Returns a status line.

        Saving is a no-op when the memory has no store, and it reports a write
        failure instead of raising, so a read-only data directory cannot crash a
        spoken turn.
        """
        if not self.memory.persistent:
            return "This session is not persistent (no store is configured)."
        try:
            self.memory.save(self.conversation)
        except OSError as error:
            return f"Could not save this conversation: {error}"
        return f"Saved this conversation as {self.conversation.id}."

    def resume_session(self, conversation_id: str) -> bool:
        """Make a saved session current. Returns whether it was found."""
        conversation = self.memory.load(conversation_id)
        if conversation is None:
            return False
        self._conversation = conversation
        return True

    def list_sessions(self) -> tuple[str, ...]:
        """The ids of sessions saved with a persistent store."""
        if not self.memory.persistent:
            return ()
        return tuple(self.memory.list_saved())

    def handle(self, utterance: str, *, require_wake: bool = False, speak: bool = True) -> JarvisTurn:
        """Process one utterance end to end and return the recorded turn."""
        text = utterance.strip()
        if not text:
            raise ValueError("Cannot handle an empty utterance.")

        woke = True
        if require_wake:
            match = self.wake.detect(text)
            if not match.woke:
                return JarvisTurn(
                    heard=text,
                    response="",
                    woke=False,
                    routed="ignored",
                )
            text = match.remainder or text

        control = self.controls.match(text)
        if control is not None:
            return self._handle_control(control, utterance, speak=speak)

        routed = self.router.route(text)
        if routed.kind is RouteKind.TOOL and routed.plan is not None:
            run = self.agent.run(routed.plan)
            response = self._describe_run(run)
            conversation = self.conversation
            conversation.add_user(text)
            conversation.add_assistant(response)
            self.memory.save(conversation)
            return JarvisTurn(
                heard=utterance.strip(),
                response=response,
                woke=woke,
                routed="tool",
                agent_run=run,
                spoken=self._speak(response, speak),
                conversation_id=conversation.id,
            )

        conversation = self.conversation
        answer = self.assistant.ask(text, conversation=conversation)
        return JarvisTurn(
            heard=utterance.strip(),
            response=answer.text,
            woke=woke,
            routed="answer",
            answer_text=answer.text,
            citations=answer.citations,
            spoken=self._speak(answer.text, speak),
            conversation_id=conversation.id,
        )

    def handle_audio(self, clip: AudioClip, *, require_wake: bool = True, speak: bool = True) -> JarvisTurn:
        """Transcribe a clip, then handle it (respecting the speech gate)."""
        if require_wake and not self.gate.has_speech(clip):
            return JarvisTurn(heard="", response="", woke=False, routed="silence")
        heard = self.voice.recognizer.transcribe(clip)
        return self.handle(heard, require_wake=require_wake, speak=speak)

    def _handle_control(self, control: JarvisControl, utterance: str, *, speak: bool) -> JarvisTurn:
        """Act on a conversation-level control without touching the sandbox."""
        intent = control.intent
        if intent is ControlIntent.MUTE:
            self.mute = True
            return self._control_turn(utterance, intent, "Muted. Say 'speak' when you want my voice back.", None)
        if intent is ControlIntent.SPEAK:
            self.mute = False
            response = "Voice on."
            return self._control_turn(utterance, intent, response, self._speak(response, speak))
        if intent is ControlIntent.REPEAT:
            last = self.conversation.last_assistant_message()
            if last is None:
                return self._control_turn(utterance, intent, "There is nothing to repeat yet.", None)
            return self._control_turn(
                utterance, intent, last.text, self._speak(last.text, speak), citations=last.citations
            )
        if intent is ControlIntent.RESET:
            self.reset_conversation()
            response = "Started a new conversation."
            return self._control_turn(utterance, intent, response, self._speak(response, speak))
        if intent is ControlIntent.SAVE:
            response = self.save_session()
            return self._control_turn(utterance, intent, response, self._speak(response, speak))
        if intent is ControlIntent.SESSIONS:
            ids = self.list_sessions()
            if not ids:
                response = "No saved sessions yet."
            else:
                response = "Saved sessions: " + ", ".join(ids)
            return self._control_turn(utterance, intent, response, self._speak(response, speak))
        if intent is ControlIntent.CONFIRM:
            response = self._handle_feedback(control.utterance, confirmed=True)
            return self._control_turn(utterance, intent, response, self._speak(response, speak))
        if intent is ControlIntent.CORRECT:
            response = self._handle_feedback(control.utterance, confirmed=False, correction=control.correction)
            return self._control_turn(utterance, intent, response, self._speak(response, speak))
        # HELP
        return self._control_turn(utterance, intent, HELP_TEXT, self._speak(HELP_TEXT, speak))

    def _handle_feedback(
        self,
        utterance: str,
        *,
        confirmed: bool,
        correction: str = "",
    ) -> str:
        """Fold explicit feedback on the last answer into the corpus.

        Teaching needs two things that come from outside the conversation: the
        last exchange to attach the verdict to, and a learner whose store feeds
        the assistant's own retrieval. When either is missing we say so plainly
        rather than recording feedback that could never change an answer.
        """
        if self.learner is None:
            return "Feedback is not enabled in this session, so I cannot record that."
        exchange = self.conversation.last_exchange()
        if exchange is None:
            return "There is no recent answer to confirm or correct."
        question, answer = exchange
        statement = answer if confirmed else correction
        try:
            step = self.learner.learn(
                Feedback(
                    question=question,
                    answer=answer,
                    confirmed=confirmed,
                    correction=correction,
                )
            )
        except ValueError as error:
            return f"Could not record that feedback: {error}"
        self._record_exchange(utterance, statement)
        outcome = step.update
        if outcome is not None and outcome.action in {"added", "reinforced"}:
            return f"Thank you -- I recorded that ({outcome.action}). I will use it next time."
        if outcome is not None and outcome.action == "rejected":
            return (
                "Thank you -- I recorded your verdict, but I did not add the "
                f"statement ({outcome.reason})."
            )
        return "Thank you -- I recorded your feedback."

    def _record_exchange(self, feedback: str, statement: str) -> None:
        """Append the feedback turn, refreshing the retriever when one is wired."""
        conversation = self.conversation
        conversation.add_user(feedback)
        conversation.add_assistant(statement)
        self._refresh_knowledge()
        self.memory.save(conversation)

    def _refresh_knowledge(self) -> None:
        """Rebuild the assistant's retrieval index from its document store.

        A new or changed document is only visible to retrieval after the
        retriever re-indexes, so teaching would otherwise not affect the next
        answer. Best-effort: a custom assistant without a retriever is left alone.
        """
        retriever = getattr(self.assistant, "retriever", None)
        index_document = getattr(retriever, "index_document", None)
        if index_document is None:
            return
        documents = getattr(getattr(retriever, "documents", None), "documents", None)
        if not callable(documents):
            return
        for document in documents():
            try:
                index_document(document)
            except (ValueError, KeyError):
                continue

    def _control_turn(
        self,
        utterance: str,
        intent: ControlIntent,
        response: str,
        spoken: AudioClip | None,
        *,
        citations: tuple[str, ...] = (),
    ) -> JarvisTurn:
        return JarvisTurn(
            heard=utterance.strip(),
            response=response,
            woke=True,
            routed="control",
            intent=intent.value,
            citations=citations,
            spoken=spoken,
            conversation_id=self.conversation.id,
        )

    def _speak(self, text: str, speak: bool) -> AudioClip | None:
        if not speak or self.mute:
            return None
        return self.voice.synthesizer.synthesize(text)

    def play(self, turn: JarvisTurn) -> str:
        """Play a turn's spoken clip. Returns an error message, or "" on success.

        Playback is best-effort and never raises: a missing or failing player
        leaves the text answer intact and reports why.
        """
        if turn.spoken is None or self.speaker is None:
            return ""
        try:
            self.speaker.play(turn.spoken)
        except ValueError as error:
            turn.playback_error = str(error)
            return turn.playback_error
        return ""

    @staticmethod
    def _describe_run(run: AgentRun) -> str:
        if not run.steps:
            return "I could not run that request."
        if run.succeeded:
            last = run.steps[-1]
            return f"Done: {last.result.output}"
        failed = next(step for step in run.steps if not step.result.ok)
        return f"I could not do that: {failed.result.error}"


__all__ = ["HELP_TEXT", "JarvisMode", "JarvisTurn"]
