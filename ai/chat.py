"""Talk with the project's AI, like Jarvis but text-first.

    python -m cli chat                          # interactive conversation
    python -m cli chat "what is porosity?"      # one question, then exit
    python -m cli chat --no-web                 # answer only from what it knows
    python -m cli chat --mind                   # answer from the trained model, not the sources
    python -m cli chat --ultron                 # answer from the general local model (any subject)
    python -m cli chat --vision                 # answer from the petroleum local model (reservoir, drilling, PVT)
    python -m cli chat --banner                 # answer from the presentation local model (outlines, slide content)
    python -m cli chat --cloud                  # answer from a remote cloud model (ChatGPT/Claude/Gemini/...)
    python -m cli chat --natasha                # talk with Natasha, the companion (memory + tone)
    python -m cli chat --hakeem                 # talk with Hakeem, the scholar of religion and history
    python -m cli chat --session demo           # resume/autosave a named session
    python -m cli chat --list-sessions          # show saved conversations

This is the plain conversation entry point: it runs the same grounded loop as
Jarvis mode -- one session, follow-up questions keep context, every answer cites
its sources -- but stays in text. No microphone, no speaker, no wake word: it
uses the transcript recognizer and the silent synthesizer, so it behaves the same
on a machine with no audio at all.

The same spoken controls still work as typed commands: "new conversation" starts
over, "save" persists the session, "sessions" lists saved ones, "web on"/"web
off" toggle internet lookups, "that's right" and "that's wrong, it's ..." teach it
from explicit feedback, and "help" lists them. Type "quit" to leave.

Seven mouths answer in the *same* conversation, swapped in **here**, in this
module, without touching the Jarvis loop or the session builder, so Jarvis mode
is unaffected:

* the sources (default) -- retrieved passages, with citations;
* ``--mind`` -- the small in-project trained language model, no citations;
* ``--ultron`` -- the real, general local model (``llama-cpp-python``), which
  talks about every field of life, no citations;
* ``--vision`` -- the real, petroleum-engineering local model
  (``llama-cpp-python``, built on OGAI Reasoner), no citations;
* ``--banner`` -- the real, presentation local model (``llama-cpp-python``, the
  Presentation Studio's own Qwen3 weights), which drafts outlines and slide
  content, no citations;
* ``--cloud`` -- a **remote** OpenAI-compatible model (ChatGPT, Claude, Gemini,
  Groq, ...), opt-in and off by default, no citations;
* ``--natasha`` -- the project's **companion**: a stable personality, a durable
  memory, and a changing tone, through the local model, no citations;
* ``--hakeem`` -- the project's **scholar of religion and history**: he compares
  the Abrahamic faiths and discusses the ancient civilisations, through the *same*
  local model the companion uses, no citations.

Type "from the model", "from ultron", "from vision", "from banner", "from cloud",
"from natasha", "from hakeem", or "from the sources" to switch mouths
mid-conversation.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from pathlib import Path

from companion.companion_agent import CompanionError
from core.banner import BannerUnavailable
from core.cloud_model import CloudUnavailable
from core.composition import sources_block
from core.jarvis import JarvisMode, JarvisTurn
from core.ultron import UltronUnavailable
from core.vision_model import VisionUnavailable
from hakeem.hakeem_agent import HakeemError

from ai import banner_mouth, cloud_mouth, hakeem_mouth, natasha_mouth, ultron_mouth, vision_mouth
from ai import mind as mind_module
from ai.jarvis import build_session

INTRO = "Chat with the project's AI. Ask a question, or type 'help' for controls; 'quit' to exit."
MIND_HINT = (
    "AI: (the sources could not answer; type 'from the model' to answer from the "
    "trained model, 'from ultron' for the general local model, 'from vision' for the "
    "petroleum local model, 'from banner' for the presentation local model, "
    "'from natasha' for the companion, 'from hakeem' for the scholar of religion and "
    "history, or 'from the sources' to go back)"
)

# The composer's wording when retrieval found nothing to stand on.
_NO_SOURCE_MARKER = "not enough evidence"

# Strips punctuation so "From the model." and "model on" match the same phrase.
_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)

#: The five answering mouths. The sources are the grounded default; ``model`` is
#: the small in-project checkpoint; ``ultron`` is the general local model;
#: ``vision`` is the petroleum-engineering local model; ``banner`` is the
#: Presentation Studio's own local model.
MOUTH_SOURCES = "sources"
MOUTH_MODEL = "model"
MOUTH_ULTRON = "ultron"
MOUTH_VISION = "vision"
MOUTH_BANNER = "banner"
MOUTH_CLOUD = "cloud"
MOUTH_NATASHA = "natasha"
MOUTH_HAKEEM = "hakeem"


def _normalize(text: str) -> str:
    cleaned = _NON_WORD.sub(" ", text.strip().casefold())
    return " ".join(cleaned.split())


def _match_switch(text: str) -> str | None:
    """Return the mouth a switch line selects, else ``None``.

    Recognised only for the whole line, so an ordinary question is never captured.
    The phrases live in :mod:`ai.mind`, :mod:`ai.ultron_mouth`,
    :mod:`ai.vision_mouth`, and :mod:`ai.banner_mouth`, not in the shared Jarvis
    controls, so the Jarvis command's wording never changes.
    """
    normalized = _normalize(text)
    if not normalized:
        return None
    if normalized in _normalize_set(mind_module.SWITCH_TO_MODEL):
        return MOUTH_MODEL
    if normalized in _normalize_set(ultron_mouth.SWITCH_TO_ULTRON):
        return MOUTH_ULTRON
    if normalized in _normalize_set(vision_mouth.SWITCH_TO_VISION):
        return MOUTH_VISION
    if normalized in _normalize_set(banner_mouth.SWITCH_TO_BANNER):
        return MOUTH_BANNER
    if normalized in _normalize_set(cloud_mouth.SWITCH_TO_CLOUD):
        return MOUTH_CLOUD
    if normalized in _normalize_set(natasha_mouth.SWITCH_TO_NATASHA):
        return MOUTH_NATASHA
    if normalized in _normalize_set(hakeem_mouth.SWITCH_TO_HAKEEM):
        return MOUTH_HAKEEM
    if normalized in _normalize_set(mind_module.SWITCH_TO_SOURCES):
        return MOUTH_SOURCES
    return None


def _normalize_set(phrases: tuple[str, ...]) -> set[str]:
    return {_normalize(phrase) for phrase in phrases}


class ModelSwitcher:
    """Swaps the session's answering mouth between the sources and a model.

    Seven mouths are offered: the grounded sources (default), the small
    in-project trained model, **Ultron**, the general local model, **Vision**, the
    petroleum-engineering local model, **Banner**, the Presentation Studio's own
    local model, **Cloud**, a remote OpenAI-compatible model, and **Natasha**, the
    companion. The
    grounded assistant and its help text are kept so switching back is exact.
    Only ``mode.assistant``/``mode.voice.assistant``/``mode.help_text`` change;
    nothing in the Jarvis loop, the controls, or the session builder is touched,
    so Jarvis mode is unaffected.

    A model is loaded lazily on the first switch to it, so an interactive chat
    that never asks for one never pays for loading it -- and a *failed* load
    raises before the mouth is swapped, so the chat keeps answering from the
    mouth that already works.
    """

    def __init__(
        self,
        mode: JarvisMode,
        *,
        loader,
        max_new_tokens: int = 96,
        ultron_loader: Callable[[], ultron_mouth.UltronAssistant] | None = None,
        vision_loader: Callable[[], vision_mouth.VisionAssistant] | None = None,
        banner_loader: Callable[[], banner_mouth.BannerAssistant] | None = None,
        cloud_loader: Callable[[], cloud_mouth.CloudAssistant] | None = None,
        natasha_loader: Callable[[], natasha_mouth.NatashaAssistant] | None = None,
        hakeem_loader: Callable[[], hakeem_mouth.HakeemAssistant] | None = None,
    ) -> None:
        self.mode = mode
        self.grounded = mode.assistant
        self.grounded_help = mode.help_text
        self._loader = loader
        self._ultron_loader = ultron_loader or ultron_mouth.load_ultron_assistant
        self._vision_loader = vision_loader or vision_mouth.load_vision_assistant
        self._banner_loader = banner_loader or banner_mouth.load_banner_assistant
        self._cloud_loader = cloud_loader or cloud_mouth.load_cloud_assistant
        self._natasha_loader = natasha_loader or natasha_mouth.load_natasha_assistant
        self._hakeem_loader = hakeem_loader or hakeem_mouth.load_hakeem_assistant
        self._max_new_tokens = max_new_tokens
        self._mind: mind_module.MindAssistant | None = None
        self._ultron: ultron_mouth.UltronAssistant | None = None
        self._vision: vision_mouth.VisionAssistant | None = None
        self._banner: banner_mouth.BannerAssistant | None = None
        self._cloud: cloud_mouth.CloudAssistant | None = None
        self._natasha: natasha_mouth.NatashaAssistant | None = None
        self._hakeem: hakeem_mouth.HakeemAssistant | None = None
        self._mouth = MOUTH_SOURCES

    @property
    def mouth(self) -> str:
        """Which mouth currently answers: ``"sources"``, ``"model"``, ``"ultron"``, ``"vision"``, ``"banner"``, or ``"cloud"``."""
        return self._mouth

    @property
    def active(self) -> bool:
        """True when a model answers instead of the grounded sources."""
        return self._mouth != MOUTH_SOURCES

    @property
    def mind(self) -> mind_module.MindAssistant:
        if self._mind is None:
            self._mind = mind_module.MindAssistant(self._loader(), max_new_tokens=self._max_new_tokens)
        return self._mind

    @property
    def ultron(self) -> ultron_mouth.UltronAssistant:
        if self._ultron is None:
            self._ultron = self._ultron_loader()
        return self._ultron

    @property
    def vision(self) -> vision_mouth.VisionAssistant:
        if self._vision is None:
            self._vision = self._vision_loader()
        return self._vision

    @property
    def banner(self) -> banner_mouth.BannerAssistant:
        if self._banner is None:
            self._banner = self._banner_loader()
        return self._banner

    @property
    def cloud(self) -> cloud_mouth.CloudAssistant:
        if self._cloud is None:
            self._cloud = self._cloud_loader()
        return self._cloud

    @property
    def natasha(self) -> natasha_mouth.NatashaAssistant:
        if self._natasha is None:
            self._natasha = self._natasha_loader()
        return self._natasha

    @property
    def hakeem(self) -> hakeem_mouth.HakeemAssistant:
        if self._hakeem is None:
            self._hakeem = self._hakeem_loader()
        return self._hakeem

    def _swap(self, assistant, mouth: str, help_suffix: str) -> None:
        """Point the session at ``assistant`` and record which mouth is active."""
        self.mode.assistant = assistant
        self.mode.voice.assistant = assistant
        self.mode.help_text = f"{self.grounded_help} {help_suffix}"
        self._mouth = mouth

    def use_model(self) -> None:
        """Activate the trained-model mouth. Raises when the checkpoint cannot load."""
        assistant = self.mind  # load now, so a failure is reported before switching
        self._swap(assistant, MOUTH_MODEL, mind_module.MIND_HELP)

    def use_ultron(self) -> None:
        """Activate the Ultron mouth. Raises when the runtime or model is missing."""
        assistant = self.ultron  # load/check now, before the swap
        self._swap(assistant, MOUTH_ULTRON, ultron_mouth.ULTRON_HELP)

    def use_vision(self) -> None:
        """Activate the Vision mouth. Raises when the runtime or model is missing."""
        assistant = self.vision  # load/check now, before the swap
        self._swap(assistant, MOUTH_VISION, vision_mouth.VISION_HELP)

    def use_banner(self) -> None:
        """Activate the Banner mouth. Raises when the runtime or model is missing."""
        assistant = self.banner  # load/check now, before the swap
        self._swap(assistant, MOUTH_BANNER, banner_mouth.BANNER_HELP)

    def use_cloud(self) -> None:
        """Activate the Cloud mouth. Raises when no key is set or the endpoint refuses."""
        assistant = self.cloud  # load/check now, before the swap
        self._swap(assistant, MOUTH_CLOUD, cloud_mouth.CLOUD_HELP)

    def use_natasha(self) -> None:
        """Activate the Natasha mouth. Raises when the local model cannot answer."""
        assistant = self.natasha  # load/check now, before the swap
        self._swap(assistant, MOUTH_NATASHA, natasha_mouth.NATASHA_HELP)

    def use_hakeem(self) -> None:
        """Activate the Hakeem mouth. Raises when the local model cannot answer."""
        assistant = self.hakeem  # load/check now, before the swap
        self._swap(assistant, MOUTH_HAKEEM, hakeem_mouth.HAKEEM_HELP)

    def invalidate_cloud(self) -> None:
        """Forget the cached Cloud mouth, so the next load picks up a new provider.

        Called when the user changes the cloud provider: without this the mouth
        would keep answering from the provider it was first built with.
        """
        self._cloud = None

    def invalidate_mouth(self, mouth: str) -> None:
        """Forget a cached mouth, so the next load rebuilds it.

        Called when the user points a mouth at a *different* language model (see
        ``gui.service.ChatService.set_mouth_model``): the assistant is cached after
        its first load, so without dropping it the reload would hand back the model
        it was first built with rather than the newly chosen one. Only the named
        mouth is dropped, so the others keep their loaded model.
        """
        if mouth == MOUTH_MODEL:
            self._mind = None
        elif mouth == MOUTH_ULTRON:
            self._ultron = None
        elif mouth == MOUTH_VISION:
            self._vision = None
        elif mouth == MOUTH_BANNER:
            self._banner = None
        elif mouth == MOUTH_CLOUD:
            self._cloud = None
        elif mouth == MOUTH_NATASHA:
            self._natasha = None
        elif mouth == MOUTH_HAKEEM:
            self._hakeem = None

    def use_sources(self) -> None:
        """Answer from retrieved passages with citations again."""
        self.mode.assistant = self.grounded
        self.mode.voice.assistant = self.grounded
        self.mode.help_text = self.grounded_help
        self._mouth = MOUTH_SOURCES


def format_turn(turn: JarvisTurn) -> str:
    """Render one turn as text: the answer, then its sources."""
    if turn.routed == "ignored":
        return "(no wake word; ignored)"
    if turn.routed == "silence":
        return "(no speech detected)"
    if turn.routed == "control":
        return f"[{turn.intent}] {turn.response}"
    if turn.agent_run is not None:
        lines = []
        for step in turn.agent_run.steps:
            status = "ok " if step.result.ok else "ERR"
            detail = step.result.output if step.result.ok else step.result.error
            lines.append(f"  [{status}] {step.call.tool}: {detail}")
        return "\n".join(lines) if lines else turn.response

    parts = [turn.response]
    if turn.relation_conflicts:
        parts.append(
            "Note: the knowledge graph holds contradictory claims on these concepts, "
            "so treat them as uncertain:\n"
            + "\n".join(f"  - {conflict}" for conflict in turn.relation_conflicts)
        )
    if turn.relation_facts:
        parts.append(
            "Knowledge graph on file:\n"
            + "\n".join(f"  - {fact}" for fact in turn.relation_facts)
        )
    if turn.inferred:
        parts.append(
            "Inferred from the knowledge graph (derived, not stored):\n"
            + "\n".join(f"  - {item}" for item in turn.inferred)
        )
    if turn.connections:
        parts.append(
            "How the graph connects the concepts in the question (walked, not stored):\n"
            + "\n".join(f"  - {item}" for item in turn.connections)
        )
    if turn.causal_paths:
        parts.append(
            "Causal paths the graph states (directed, cause → effect):\n"
            + "\n".join(f"  - {item}" for item in turn.causal_paths)
        )
    if turn.plan:
        parts.append(
            "The procedure the sources state (ordered by the planner):\n"
            + "\n".join(f"  {item}" for item in turn.plan)
        )
    if turn.decision:
        parts.append(
            "The options the graph states (ranked by the decision engine):\n"
            + "\n".join(f"  - {item}" for item in turn.decision)
        )
    if turn.confidence_note:
        parts.append(f"Confidence in this answer (evidence weighed with the graph): {turn.confidence_note}")
    if turn.reasoning_trace:
        parts.append(turn.reasoning_trace)
    block = sources_block(turn.citations)
    if block:
        parts.append(block)
    return "\n\n".join(parts)


def turn_payload(turn: JarvisTurn) -> dict:
    """A machine-readable record of one turn."""
    payload = {
        "heard": turn.heard,
        "routed": turn.routed,
        "intent": turn.intent,
        "response": turn.response,
        "citations": list(turn.citations),
        "relation_facts": list(turn.relation_facts),
        "relation_conflicts": list(turn.relation_conflicts),
        "inferred": list(turn.inferred),
        "connections": list(turn.connections),
        "causal_paths": list(turn.causal_paths),
        "plan": list(turn.plan),
        "decision": list(turn.decision),
        "confidence": turn.confidence,
        "confidence_note": turn.confidence_note,
        "reasoning_trace": turn.reasoning_trace,
        "used_web": turn.used_web,
        "learned": turn.learned,
        "conversation_id": turn.conversation_id,
    }
    if turn.agent_run is not None:
        payload["steps"] = [
            {
                "tool": step.call.tool,
                "ok": step.result.ok,
                "output": step.result.output,
                "error": step.result.error,
            }
            for step in turn.agent_run.steps
        ]
    return payload


def print_turn(turn: JarvisTurn, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(turn_payload(turn), ensure_ascii=False))
        return
    print(format_turn(turn))


def _print_switch(kind: str, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"routed": "switch", "mode": kind}, ensure_ascii=False))
        return
    if kind == MOUTH_MODEL:
        print("AI: [model] Now answering from the trained model only (no sources).")
    elif kind == MOUTH_ULTRON:
        print("AI: [ultron] Now answering from the general local model (any subject, no sources).")
    elif kind == MOUTH_VISION:
        print("AI: [vision] Now answering from the petroleum local model (reservoir, drilling, PVT; no sources).")
    elif kind == MOUTH_BANNER:
        print("AI: [banner] Now answering from the presentation local model (outlines and slide content; no sources).")
    elif kind == MOUTH_CLOUD:
        print("AI: [cloud] Now answering from a remote cloud model over the network (no sources).")
    elif kind == MOUTH_NATASHA:
        print("AI: [natasha] Now talking with Natasha, the companion (stable personality, lasting memory, changing tone).")
    elif kind == MOUTH_HAKEEM:
        print("AI: [hakeem] Now talking with Hakeem, the scholar of religion and history (compares the Abrahamic faiths; ancient civilisations).")
    else:
        print("AI: [sources] Now answering from retrieved passages with citations.")
    print()


#: What to say when a mouth could not be loaded, and how to fix it. Ultron's,
#: Vision's, and Banner's messages name their own real next step (install the
#: extra, or fetch a model), so a failed switch is never a dead end.
_MOUTH_FAILURE = {
    MOUTH_MODEL: (
        "Could not load the trained model: {error}",
        "Train it first with `python -m cli llm` (or point --model-checkpoint at one).",
    ),
    MOUTH_ULTRON: (
        "Could not load Ultron: {error}",
        "Install the optional runtime with `python -m pip install -r requirements-ultron.txt`, "
        "then fetch a model with `python -m cli ultron download --approve`.",
    ),
    MOUTH_VISION: (
        "Could not load Vision: {error}",
        "Install the optional runtime with `python -m pip install -r requirements-vision.txt`, "
        "then fetch a model with `python -m cli vision download --approve`.",
    ),
    MOUTH_BANNER: (
        "Could not load Banner: {error}",
        "Install the optional runtime with `python -m pip install -r requirements-banner.txt`, "
        "then fetch a model with `python -m cli banner download --approve`.",
    ),
    MOUTH_CLOUD: (
        "Could not reach the cloud model: {error}",
        "Set the provider's API key in an environment variable (e.g. OPENAI_API_KEY, "
        "ANTHROPIC_API_KEY, GEMINI_API_KEY), then run `python -m cli cloud status` "
        "to check the endpoint.",
    ),
    MOUTH_NATASHA: (
        "Could not reach Natasha's local model: {error}",
        "Fetch the in-process weights with `python -m cli natasha download --approve`, "
        "or run a local Ollama server and pull the tag with "
        "`python -m cli natasha download --approve --backend ollama` "
        "(run `python -m cli natasha status` to see which runtime answers).",
    ),
    MOUTH_HAKEEM: (
        "Could not reach Hakeem's local model: {error}",
        "He speaks through the same model as the companion, so fetch the in-process "
        "weights with `python -m cli hakeem download --approve`, or run a local Ollama "
        "server and pull the tag with `python -m cli hakeem download --approve --backend "
        "ollama` (run `python -m cli hakeem status` to see which runtime answers).",
    ),
}

#: The confirmation shown once a mouth is active. Shared by the text loop and the
#: window, so both describe the swap with the same words.
_MOUTH_SWITCHED = {
    MOUTH_MODEL: "Now answering from the trained model (no sources).",
    MOUTH_ULTRON: "Now answering from Ultron, the general local model (any subject, no citations).",
    MOUTH_VISION: "Now answering from Vision, the petroleum local model (reservoir, drilling, PVT; no citations).",
    MOUTH_BANNER: "Now answering from Banner, the presentation local model (outlines and slide content; no citations).",
    MOUTH_CLOUD: "Now answering from a remote cloud model over the network (no citations).",
    MOUTH_NATASHA: "Now talking with Natasha, the companion (stable personality, lasting memory, changing tone; no citations).",
    MOUTH_HAKEEM: "Now talking with Hakeem, the scholar of religion and history (compares the Abrahamic faiths and discusses the ancient civilisations; no citations).",
    MOUTH_SOURCES: "Now answering from sources, with citations.",
}


def switch_failure_message(kind: str, error: Exception) -> str:
    """A short, actionable message for a mouth that could not be loaded."""
    lines = _MOUTH_FAILURE.get(kind)
    if lines is None:
        return f"Could not switch mouths: {error}"
    return "\n".join(line.format(error=error) for line in lines)


def mouth_switched_message(kind: str) -> str:
    """The confirmation shown once ``kind`` is the active mouth."""
    return _MOUTH_SWITCHED.get(kind, _MOUTH_SWITCHED[MOUTH_SOURCES])


def _apply_switch(switcher, kind: str) -> str:
    """Swap the session's mouth. Returns an error message, or ``""`` on success."""
    try:
        if kind == MOUTH_MODEL:
            switcher.use_model()
        elif kind == MOUTH_ULTRON:
            switcher.use_ultron()
        elif kind == MOUTH_VISION:
            switcher.use_vision()
        elif kind == MOUTH_BANNER:
            switcher.use_banner()
        elif kind == MOUTH_CLOUD:
            switcher.use_cloud()
        elif kind == MOUTH_NATASHA:
            switcher.use_natasha()
        elif kind == MOUTH_HAKEEM:
            switcher.use_hakeem()
        else:
            switcher.use_sources()
    except (FileNotFoundError, OSError, ValueError, UltronUnavailable, VisionUnavailable, BannerUnavailable, CloudUnavailable, CompanionError, HakeemError) as error:
        return switch_failure_message(kind, error)
    return ""


def run_loop(mode, *, as_json: bool, autosave: bool = False, switcher=None) -> int:
    """Read questions from the terminal and answer until 'quit' or EOF.

    With a ``switcher`` the typed controls "from the model", "from ultron", and
    "from the sources" change which mouth answers; without one the loop is
    exactly as before.
    """
    if not as_json:
        print(INTRO)
    while True:
        try:
            line = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.strip().casefold() in {"quit", "exit", "q", ""}:
            return 0
        if switcher is not None:
            switch = _match_switch(line)
            if switch is not None:
                failure = _apply_switch(switcher, switch)
                if failure:
                    print(failure)
                    print()
                    continue
                _print_switch(switch, as_json=as_json)
                continue
        turn = mode.handle(line, require_wake=False, speak=False)
        if not as_json:
            print("AI:", end=" ")
        print_turn(turn, as_json=as_json)
        if not as_json:
            print()
            if switcher is not None and not switcher.active and _NO_SOURCE_MARKER in turn.response:
                print(MIND_HINT)
                print()
        if autosave:
            mode.save_session()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chat",
        description="Talk with the project's AI in text (like Jarvis, without voice).",
    )
    parser.add_argument("message", nargs="*", help="one message; omit to enter the interactive loop")
    parser.add_argument("--root", default=None, help="sandbox root directory for tools")
    parser.add_argument("--read-write", action="store_true", help="allow writing tools in the sandbox")
    parser.add_argument("--no-web", action="store_true", help="answer only from internal knowledge; do not search the internet")
    parser.add_argument(
        "--web-threshold",
        type=float,
        default=None,
        help="top internal score below which it searches the web (0-1; higher searches more)",
    )
    parser.add_argument("--no-feedback", action="store_true", help="disable learning from confirm/correct controls")
    parser.add_argument("--session", default=None, help="resume a saved session by id, or start a new one")
    parser.add_argument("--list-sessions", action="store_true", help="list saved sessions and exit")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--mind",
        action="store_true",
        help="answer from the trained model itself (no retrieval, no citations), not from the sources",
    )
    parser.add_argument("--model-checkpoint", default=None, help="path to a language-model checkpoint to answer with")
    parser.add_argument("--mind-temperature", type=float, default=0.0, help="sampling temperature for model answers (0 = deterministic)")
    parser.add_argument("--mind-tokens", type=int, default=96, help="maximum tokens to generate per model answer")
    parser.add_argument(
        "--ultron",
        action="store_true",
        help="answer from Ultron, the general local model (any subject, no citations), not from the sources",
    )
    parser.add_argument("--ultron-model", default=None, help="a specific .gguf file for Ultron to load")
    parser.add_argument("--ultron-tokens", type=int, default=None, help="maximum tokens per Ultron answer")
    parser.add_argument("--ultron-temperature", type=float, default=None, help="sampling temperature for Ultron answers")
    parser.add_argument(
        "--vision",
        action="store_true",
        help="answer from Vision, the petroleum local model (reservoir, drilling, production, PVT; no citations), not from the sources",
    )
    parser.add_argument("--vision-model", default=None, help="a specific .gguf file for Vision to load")
    parser.add_argument("--vision-tokens", type=int, default=None, help="maximum tokens per Vision answer")
    parser.add_argument("--vision-temperature", type=float, default=None, help="sampling temperature for Vision answers")
    parser.add_argument(
        "--banner",
        action="store_true",
        help="answer from Banner, the presentation local model (outlines and slide content; no citations), not from the sources",
    )
    parser.add_argument("--banner-model", default=None, help="a specific .gguf file for Banner to load")
    parser.add_argument("--banner-tokens", type=int, default=None, help="maximum tokens per Banner answer")
    parser.add_argument("--banner-temperature", type=float, default=None, help="sampling temperature for Banner answers")
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="answer from a remote cloud model (ChatGPT/Claude/Gemini/... over the network; no citations), not from the sources",
    )
    parser.add_argument("--cloud-provider", default=None, help="which cloud provider to use (e.g. openai, anthropic, gemini, groq)")
    parser.add_argument("--cloud-model", default=None, help="the cloud model to ask for (overrides the provider default)")
    parser.add_argument("--cloud-base-url", default=None, help="an OpenAI-compatible base URL to use instead")
    parser.add_argument("--cloud-api-key-env", default=None, help="the environment variable holding the cloud API key")
    parser.add_argument("--cloud-tokens", type=int, default=None, help="maximum tokens per cloud answer")
    parser.add_argument("--cloud-temperature", type=float, default=None, help="sampling temperature for cloud answers")
    parser.add_argument(
        "--natasha",
        action="store_true",
        help="talk with Natasha, the companion (stable personality, lasting memory, changing tone; no citations), not from the sources",
    )
    parser.add_argument("--natasha-session", default=None, help="the companion session id to resume or create")
    parser.add_argument("--natasha-model", default=None, help="the local model Natasha speaks through (default qwen3:4b)")
    parser.add_argument(
        "--hakeem",
        action="store_true",
        help="talk with Hakeem, the scholar of religion and history (compares the Abrahamic faiths, discusses the ancient civilisations; no citations), not from the sources",
    )
    parser.add_argument("--hakeem-session", default=None, help="the scholar session id to resume or create")
    parser.add_argument("--hakeem-model", default=None, help="the local model Hakeem speaks through (default qwen3:4b)")
    args = parser.parse_args(argv)

    if args.list_sessions:
        mode = build_session(no_web=args.no_web, no_feedback=args.no_feedback, text_only=True)
        saved = mode.list_sessions()
        if saved:
            print("Saved sessions:")
            for session_id in saved:
                print(f"  - {session_id}")
        else:
            print("No saved sessions yet.")
        return 0

    mode = build_session(
        root=Path(args.root) if args.root else None,
        read_write=args.read_write,
        no_web=args.no_web,
        web_threshold=args.web_threshold,
        no_feedback=args.no_feedback,
        text_only=True,
    )

    switcher = ModelSwitcher(
        mode,
        loader=lambda: mind_module.load_mind_model(
            args.model_checkpoint,
            temperature=args.mind_temperature,
            max_new_tokens=args.mind_tokens,
        ),
        max_new_tokens=args.mind_tokens,
        ultron_loader=lambda: ultron_mouth.load_ultron_assistant(
            args.ultron_model,
            max_tokens=args.ultron_tokens,
            temperature=args.ultron_temperature,
        ),
        vision_loader=lambda: vision_mouth.load_vision_assistant(
            args.vision_model,
            max_tokens=args.vision_tokens,
            temperature=args.vision_temperature,
        ),
        banner_loader=lambda: banner_mouth.load_banner_assistant(
            args.banner_model,
            max_tokens=args.banner_tokens,
            temperature=args.banner_temperature,
        ),
        cloud_loader=lambda: cloud_mouth.load_cloud_assistant(
            args.cloud_model,
            provider=args.cloud_provider,
            base_url=args.cloud_base_url,
            api_key_env=args.cloud_api_key_env,
            max_tokens=args.cloud_tokens,
            temperature=args.cloud_temperature,
        ),
        natasha_loader=lambda: natasha_mouth.load_natasha_assistant(
            session_id=args.natasha_session,
            model=args.natasha_model,
        ),
        hakeem_loader=lambda: hakeem_mouth.load_hakeem_assistant(
            session_id=args.hakeem_session,
            model=args.hakeem_model,
        ),
    )
    if args.mind:
        failure = _apply_switch(switcher, MOUTH_MODEL)
        if failure:
            print(failure)
            return 1
        if not args.json:
            print(mind_module.status_line(mind_module.mind_status(args.model_checkpoint)))
    if args.ultron:
        failure = _apply_switch(switcher, MOUTH_ULTRON)
        if failure:
            print(failure)
            return 1
        if not args.json:
            print(switcher.ultron.describe())
    if args.vision:
        failure = _apply_switch(switcher, MOUTH_VISION)
        if failure:
            print(failure)
            return 1
        if not args.json:
            print(switcher.vision.describe())
    if args.banner:
        failure = _apply_switch(switcher, MOUTH_BANNER)
        if failure:
            print(failure)
            return 1
        if not args.json:
            print(switcher.banner.describe())
    if args.cloud:
        failure = _apply_switch(switcher, MOUTH_CLOUD)
        if failure:
            print(failure)
            return 1
        if not args.json:
            print(switcher.cloud.describe())
    if args.natasha:
        failure = _apply_switch(switcher, MOUTH_NATASHA)
        if failure:
            print(failure)
            return 1
        if not args.json:
            print(switcher.natasha.describe())
    if args.hakeem:
        failure = _apply_switch(switcher, MOUTH_HAKEEM)
        if failure:
            print(failure)
            return 1
        if not args.json:
            print(switcher.hakeem.describe())

    if args.session:
        if not mode.resume_session(args.session):
            if not args.json:
                print(f"No saved session '{args.session}'. Starting a new one.")
        elif not args.json:
            print(f"Resumed session {mode.conversation.id}.")

    if args.message:
        turn = mode.handle(" ".join(args.message), require_wake=False, speak=False)
        print_turn(turn, as_json=args.json)
        if args.session:
            mode.save_session()
        return 0

    return run_loop(mode, as_json=args.json, autosave=args.session is not None, switcher=switcher)


__all__ = [
    "INTRO",
    "MOUTH_BANNER",
    "MOUTH_CLOUD",
    "MOUTH_HAKEEM",
    "MOUTH_MODEL",
    "MOUTH_NATASHA",
    "MOUTH_SOURCES",
    "MOUTH_ULTRON",
    "MOUTH_VISION",
    "ModelSwitcher",
    "format_turn",
    "main",
    "mouth_switched_message",
    "print_turn",
    "run_loop",
    "switch_failure_message",
    "turn_payload",
]


if __name__ == "__main__":
    raise SystemExit(main())
