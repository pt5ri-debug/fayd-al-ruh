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
from core.speech import AudioClip, Speaker, VoiceAssistant
from core.tool_agent import AgentRun, ToolAgent
from core.wake_word import AudioSpeechGate, WakeWordDetector


@dataclass
class JarvisTurn:
    """The full record of one Jarvis exchange."""

    heard: str
    response: str
    woke: bool
    routed: str
    answer_text: str = ""
    agent_run: AgentRun | None = None
    citations: tuple[str, ...] = field(default_factory=tuple)
    spoken: AudioClip | None = None
    playback_error: str = ""


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
    ) -> None:
        self.assistant = assistant
        self.router = router
        self.agent = agent or ToolAgent(router.sandbox)
        self.voice = voice or VoiceAssistant(assistant)
        self.wake = wake or WakeWordDetector()
        self.gate = gate or AudioSpeechGate()
        self.speaker = speaker

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

        routed = self.router.route(text)
        if routed.kind is RouteKind.TOOL and routed.plan is not None:
            run = self.agent.run(routed.plan)
            response = self._describe_run(run)
            return JarvisTurn(
                heard=utterance.strip(),
                response=response,
                woke=woke,
                routed="tool",
                agent_run=run,
                spoken=self._speak(response, speak),
            )

        answer = self.assistant.ask(text)
        return JarvisTurn(
            heard=utterance.strip(),
            response=answer.text,
            woke=woke,
            routed="answer",
            answer_text=answer.text,
            citations=answer.citations,
            spoken=self._speak(answer.text, speak),
        )

    def handle_audio(self, clip: AudioClip, *, require_wake: bool = True, speak: bool = True) -> JarvisTurn:
        """Transcribe a clip, then handle it (respecting the speech gate)."""
        if require_wake and not self.gate.has_speech(clip):
            return JarvisTurn(heard="", response="", woke=False, routed="silence")
        heard = self.voice.recognizer.transcribe(clip)
        return self.handle(heard, require_wake=require_wake, speak=speak)

    def _speak(self, text: str, speak: bool) -> AudioClip | None:
        if not speak:
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


__all__ = ["JarvisMode", "JarvisTurn"]
