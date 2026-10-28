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
from core.continuous_learning import ContinuousLearner, Feedback
from core.conversation import Conversation, ConversationMemory
from core.jarvis_controls import ControlIntent, JarvisControl, JarvisControls
from core.speech import AudioClip, Speaker, VoiceAssistant
from core.tool_agent import AgentRun, ToolAgent
from core.wake_word import AudioSpeechGate, WakeWordDetector

HELP_TEXT = (
    "Conversation controls: say 'stop' to mute my voice, 'speak' to turn it back "
    "on, 'repeat' to hear the last answer again, 'why' to hear how I reached it, "
    "'new conversation' to start over, "
    "'save' to persist this session, 'sessions' to list saved ones, 'web on' or "
    "'web off' to toggle internet lookups, 'that's right' to confirm my last "
    "answer, 'that's wrong, it's ...' to correct it, or 'help' for this list. "
    "Anything else, ask me a question or give me a command."
)

#: The same controls phrased for a text-only conversation, where there is no
#: voice to mute and nothing to replay out loud.
TEXT_HELP_TEXT = (
    "Conversation controls: type 'new conversation' to start over, 'save' to "
    "persist this session, 'sessions' to list saved ones, 'repeat' to show the "
    "last answer again, 'why' to show how I reached it, 'web on' or 'web off' to "
    "toggle internet lookups, "
    "'that's right' to confirm my last answer, 'that's wrong, it's ...' to correct "
    "it, or 'quit' to leave. Anything else, ask me a question or give me a command."
)


def _web_note(used_web: bool, learned: bool) -> str:
    """The provenance line appended to a web-augmented answer."""
    if used_web and learned:
        return "Note: I searched the web, compared it with what I knew, and added the new knowledge."
    if used_web:
        return "Note: I answered this partly from the web, not only from what I knew."
    return "Note: I recorded a new fact in my own knowledge."


def _belief_lines(belief: object) -> tuple[str, ...]:
    """The evidence-weighed belief as readable lines, or ``()`` when there is none.

    Takes the answer's own ``AnswerBelief`` (``core.answer_belief``) through
    ``getattr`` so the turn carries only what the answer actually produced -- a
    question that asks no verdict yields no lines rather than a guessed belief.
    """
    if belief is None:
        return ()
    statement = getattr(belief, "statement", "")
    confidence = getattr(belief, "confidence", None)
    status = getattr(belief, "status", "")
    if not statement or confidence is None:
        return ()
    lines = [f"{statement} — confidence {confidence:.2f} ({status})"]
    lines.extend(
        f"supports: {line}" for line in getattr(belief, "supporting", ()) or ()
    )
    lines.extend(
        f"contradicts: {line}" for line in getattr(belief, "contradicting", ()) or ()
    )
    return tuple(lines)


def _temporal_lines(temporal: object) -> tuple[str, ...]:
    """The conversation's timeline as readable lines, or ``()`` when there is none.

    Takes the answer's own ``AnswerTemporal`` (``core.answer_temporal``) through
    ``getattr`` so the turn carries only what the answer actually produced -- a
    question that asks nothing temporal, or a conversation too short to have a
    timeline, yields no lines rather than a guessed one.
    """
    if temporal is None:
        return ()
    entries = getattr(temporal, "entries", ()) or ()
    if not entries:
        return ()
    lines = [f"{entry.role}: {entry.text}" for entry in entries]
    change = getattr(temporal, "change", "")
    relation = getattr(temporal, "relation", "")
    if change:
        lines.append(f"latest change: {change} ({relation})")
    pattern = getattr(temporal, "pattern", None)
    if pattern is not None:
        lines.append(
            f"pattern \"{pattern.description}\" x{pattern.occurrences} ({pattern.direction})"
        )
    prediction = getattr(temporal, "prediction", None)
    if prediction is not None:
        lines.append(f"next expected: {prediction.predicted_time.isoformat()}")
    return tuple(lines)


def _analogy_lines(analogy: object) -> tuple[str, ...]:
    """The structural comparison as readable lines, or ``()`` when there is none.

    Takes the answer's own ``AnswerAnalogy`` (``core.answer_analogy``) through
    ``getattr`` so the turn carries only what the answer actually produced -- a
    question that asks no comparison, or names fewer than two concepts, yields no
    lines rather than a guessed one.
    """
    if analogy is None:
        return ()
    matched = getattr(analogy, "matched", ()) or ()
    if not matched:
        return ()
    base = getattr(analogy, "base", "")
    target = getattr(analogy, "target", "")
    verdict = "meaningful analogy" if getattr(analogy, "meaningful", False) else "not an analogy"
    lines = [f"{base} ~ {target} ({verdict}, score {getattr(analogy, 'score', 0.0):.2f})"]
    lines.extend(f"matched: {line}" for line in matched)
    lines.extend(
        f"candidate: {source} --{relation}--> {target} (confidence {confidence:.2f})"
        for source, relation, target, confidence in getattr(analogy, "transferred", ())
    )
    return tuple(lines)


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
    relation_facts: tuple[str, ...] = field(default_factory=tuple)
    """The knowledge graph's typed relations on the question's concepts, as
    readable lines (``porosity —increases→ permeability``). Rendered after the
    answer so a reader can tell a stored link from a cited sentence."""
    relation_conflicts: tuple[str, ...] = field(default_factory=tuple)
    """Contradictory claims the graph holds on the question's concepts, as
    readable lines. Rendered as an honest warning, never resolved for the user."""
    inferred: tuple[str, ...] = field(default_factory=tuple)
    """Relations the graph *entails* on the question's concepts, as readable lines
    with the steps they rest on. Rendered under a label that says they are derived,
    never stored, so a reader can tell an inference from a written claim."""
    connections: tuple[str, ...] = field(default_factory=tuple)
    """Chains of typed relations connecting the concepts the question names, as
    readable lines with each step's direction. Rendered under a label that says the
    chain was walked, never stored, so a reader can tell a multi-hop conclusion from
    a single written edge."""
    causal_paths: tuple[str, ...] = field(default_factory=tuple)
    """Directed causal chains the graph states between the question's concepts, as
    readable lines with the path and the mechanism. Rendered under a label that says
    the path is directed (cause → effect), so an association is never read as a
    cause."""
    plan: tuple[str, ...] = field(default_factory=tuple)
    """The ordered procedure the corpus states for a question that asks *how*, as
    readable numbered steps. Empty when the question asks no procedure, so a
    definitional question carries no plan rather than a guessed one."""
    decision: tuple[str, ...] = field(default_factory=tuple)
    """The ranked recommendation the graph states for a question that asks what to
    do, as readable lines (``acid_stimulation (increases, confidence 0.70)``).
    Empty when the question asks nothing to do, so a definitional question carries
    no recommendation rather than a guessed one."""
    belief: tuple[str, ...] = field(default_factory=tuple)
    """The evidence-weighed belief the graph supports for a question that asks
    *whether a claim holds*, as readable lines (the claim, its confidence and
    status, and the edges it rests on). Empty when the question asks no verdict, so
    a definitional question carries no belief rather than a guessed one."""
    lead: str = ""
    """The reasoning the question warranted, as it opens the answer body -- the
    strongest of the plan, decision, belief, and timeline bridges, already at the
    head of ``response`` (``core.answer_lead``). Empty when the question warranted
    no reasoning, so a definitional answer carries no lead rather than a guessed
    one. Carried separately so a machine-readable turn can show the reasoning *as
    the answer's own opening*, not only in the trailing ``reasoning_trace``."""
    temporal: tuple[str, ...] = field(default_factory=tuple)
    """The timeline the conversation holds for a question that asks *what changed*,
    as readable lines (each turn, the latest change, and any pattern the temporal
    reasoner found). Empty when the question asks nothing temporal or the
    conversation holds fewer than two turns, so a definitional question carries no
    timeline rather than a guessed one (``core.answer_temporal``)."""
    analogy: tuple[str, ...] = field(default_factory=tuple)
    """The structural comparison the graph holds for a question that asks how two
    things are alike, as readable lines (the engine's verdict, its matched
    relations, and its candidate inferences). Empty when the question asks no
    comparison or names fewer than two concepts, so a definitional question carries
    no comparison rather than a guessed one (``core.answer_analogy``)."""
    confidence: float = 0.0
    """The answer's confidence: the retrieved evidence weighed with the graph's own
    reasoning and its contradictions (``core.answer_confidence``). ``0.0`` when the
    turn carried no answer (a control, a silence, a tool run)."""
    confidence_note: str = ""
    """Why the confidence is what it is -- the evidence term, the graph support it
    gained, the contradiction penalty it lost. Rendered under a label that says it is
    a confidence, so a reader can see which term moved the number rather than taking
    a bare figure on trust. Empty when no answer was produced."""
    reasoning_trace: str = ""
    """The whole trace of *why* this answer -- what was read, what was derived, and
    how the confidence was reached -- as one ordered, labelled block
    (``core.reasoning_report``). Rendered under a heading that says it is the
    reasoning, so a reader sees the steps rather than taking the answer on trust.
    Empty when no answer was produced or the answer carried no trace."""
    spoken: AudioClip | None = None
    playback_error: str = ""
    conversation_id: str = ""
    used_web: bool = False
    learned: bool = False


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
        help_text: str = HELP_TEXT,
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
        self.help_text = help_text
        self._conversation: Conversation | None = None
        # The last answer's reasoning trace, kept so the ``why`` control can repeat
        # it on demand. Held as text (not the whole answer) so nothing else is kept
        # alive and a new conversation drops it.
        self._last_reasoning = ""

    @property
    def conversation(self) -> Conversation:
        """The current session, created on first use so follow-ups keep context."""
        if self._conversation is None:
            self._conversation = self.memory.start("Jarvis session")
        return self._conversation

    def reset_conversation(self) -> Conversation:
        """Start a new memory session, leaving earlier turns behind."""
        self._conversation = self.memory.start("Jarvis session")
        self._last_reasoning = ""
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
        response = answer.text
        used_web = bool(getattr(answer, "used_web", False))
        learned = bool(getattr(answer, "learned", False))
        if used_web or learned:
            response = f"{response}\n\n{_web_note(used_web, learned)}"
        facts = tuple(
            fact.describe() for fact in getattr(answer, "relation_facts", ()) or ()
        )
        conflicts = tuple(
            conflict.describe() for conflict in getattr(answer, "relation_conflicts", ()) or ()
        )
        inferences = tuple(
            item.explain() for item in getattr(answer, "inferred", ()) or ()
        )
        connections = tuple(
            item.describe() for item in getattr(answer, "connections", ()) or ()
        )
        causal = tuple(
            item.describe() for item in getattr(answer, "causal_paths", ()) or ()
        )
        plan = tuple(
            f"{index}. {step}"
            for index, step in enumerate(
                getattr(getattr(answer, "plan", None), "steps", ()) or (), start=1
            )
        )
        decision = tuple(
            f"{option} ({relation}, confidence {confidence:.2f})"
            for option, relation, confidence in (
                getattr(getattr(answer, "decision", None), "options", ()) or ()
            )
        )
        belief = _belief_lines(getattr(answer, "belief", None))
        temporal = _temporal_lines(getattr(answer, "temporal", None))
        analogy = _analogy_lines(getattr(answer, "analogy", None))
        lead = getattr(getattr(answer, "lead", None), "describe", lambda: "")()
        report = getattr(answer, "confidence_report", None)
        confidence_note = report.describe() if report is not None else ""
        trace = getattr(answer, "reasoning", None)
        reasoning_trace = trace.describe() if trace is not None else ""
        self._last_reasoning = reasoning_trace
        return JarvisTurn(
            heard=utterance.strip(),
            response=response,
            woke=woke,
            routed="answer",
            answer_text=response,
            citations=answer.citations,
            relation_facts=facts,
            relation_conflicts=conflicts,
            inferred=inferences,
            connections=connections,
            causal_paths=causal,
            plan=plan,
            decision=decision,
            belief=belief,
            lead=lead,
            temporal=temporal,
            analogy=analogy,
            confidence=float(getattr(answer, "confidence", 0.0) or 0.0),
            confidence_note=confidence_note,
            reasoning_trace=reasoning_trace,
            spoken=self._speak(response, speak),
            conversation_id=conversation.id,
            used_web=used_web,
            learned=learned,
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
        if intent is ControlIntent.WEB:
            response = self._handle_web_control(control)
            return self._control_turn(utterance, intent, response, self._speak(response, speak))
        if intent is ControlIntent.CONFIRM:
            response = self._handle_feedback(control.utterance, confirmed=True)
            return self._control_turn(utterance, intent, response, self._speak(response, speak))
        if intent is ControlIntent.CORRECT:
            response = self._handle_feedback(control.utterance, confirmed=False, correction=control.correction)
            return self._control_turn(utterance, intent, response, self._speak(response, speak))
        if intent is ControlIntent.WHY:
            if not self._last_reasoning:
                response = "There is no recent answer to explain yet."
            else:
                response = self._last_reasoning
            return self._control_turn(utterance, intent, response, self._speak(response, speak))
        # HELP is the remaining control.
        return self._control_turn(utterance, intent, self.help_text, self._speak(self.help_text, speak))

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

    def _handle_web_control(self, control: JarvisControl) -> str:
        """Turn the assistant's internet lookups on or off, honestly."""
        assistant = self.assistant
        if getattr(assistant, "web", None) is None:
            return "Internet lookups are not configured in this session."
        tokens = control.phrase.split()
        enable = not ("off" in tokens or control.phrase == "offline")
        assistant.web_enabled = enable
        if enable:
            return "Internet lookups are on: I will search, compare, and learn trusted answers."
        return "Internet lookups are off: I will answer only from what I already know."

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


__all__ = ["HELP_TEXT", "TEXT_HELP_TEXT", "JarvisMode", "JarvisTurn"]
