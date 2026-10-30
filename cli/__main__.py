"""Entry point for the portable build and ``python -m cli``.

Usage:
    python -m cli                     # interactive numbered menu
    python -m cli gui                 # open the ChatGPT-style graphical interface
    python -m cli voice "your question" # one voice turn
    python -m cli speak "text" --voice man-1  # read text aloud with a chosen voice
    python -m cli voices              # list the reading voices (men/women/boys/girls)
    python -m cli agent               # tool-agent demo in the safety sandbox
    python -m cli jarvis              # Jarvis mode: listen, understand, act, speak
    python -m cli ultron              # Ultron: a general, advanced local model (any subject)
    python -m cli ultron "who wrote Hamlet?"   # one question, answered locally
    python -m cli ultron download --approve    # one-time: fetch Ultron's model (~5 GB)
    python -m cli vision              # Vision: a petroleum-engineering local model
    python -m cli vision "what is the MBE?"    # one question, answered locally
    python -m cli vision download --approve    # one-time: fetch Vision's model (~5 GB)
    python -m cli banner              # Banner: the Presentation Studio's local model
    python -m cli banner "an outline for a deck on well testing"   # one question
    python -m cli banner download --approve    # one-time: fetch Banner's model (~5 GB)
    python -m cli cloud               # Cloud: a remote model (ChatGPT, Claude, Gemini, ...)
    python -m cli cloud "who wrote Hamlet?"    # one question, answered by the remote model
    python -m cli cloud --provider groq "..."  # choose the provider
    python -m cli cloud status        # is a key set and the endpoint reachable
    python -m cli natasha             # chat with Natasha, the companion (local model)
    python -m cli natasha "مرحبا"      # one message; --session <id> keeps one session
    python -m cli natasha status      # which runtime answers, and the next step
    python -m cli natasha models      # list the models she can fetch
    python -m cli natasha download --approve  # one-time: fetch the GGUF weights (~2.6 GB)
    python -m cli hakeem              # chat with Hakeem, the scholar of religion and history
    python -m cli hakeem "قارن بين الأديان"  # one message; --session <id> keeps one session
    python -m cli hakeem status       # which runtime answers, and the next step
    python -m cli hakeem models       # list the models he can fetch
    python -m cli hakeem download --approve   # one-time: fetch the shared weights (~2.6 GB)
    python -m cli chat                # chat with the AI in text (Jarvis, no voice)
    python -m cli chat --ultron       # chat answering from Ultron (general local model)
    python -m cli chat --vision       # chat answering from Vision (petroleum local model)
    python -m cli chat --banner       # chat answering from Banner (presentation local model)
    python -m cli chat --cloud        # chat answering from a remote cloud model
    python -m cli chat --hakeem       # chat answering as Hakeem (scholar of religion and history)
    python -m cli ultron-chat         # the same, from the menu/CLI command
    python -m cli vision-chat         # chat answering from Vision (petroleum local model)
    python -m cli banner-chat         # chat answering from Banner (presentation local model)
    python -m cli cloud-chat          # chat answering from a remote cloud model
    python -m cli hakeem-chat         # chat answering as Hakeem (scholar of religion and history)
    python -m cli mind-chat           # chat answering from the trained model (no sources)
    python -m cli learn               # continuous learning + regression gate
    python -m cli insight             # what the AI can do and what it has learned
    python -m cli insight "question"  # ask the AI, with citations
    python -m cli internet-learner    # learn petroleum facts from trusted sites
    python -m cli web "question"      # web answer, cited and trust-scored (governed)
    python -m cli knowledge build     # build the bulk knowledge base (100 MB+)
    python -m cli knowledge stats     # show the bulk knowledge base size
    python -m cli knowledge search Q  # search the built knowledge base
    python -m cli knowledge build-index # build the fast full-text index
    python -m cli ingest add DOCS       # ingest files/folders/SQL (streaming)
    python -m cli ingest list-connectors # show the available connectors
    python -m cli ingest stats          # ingestion store + ledger status
    python -m cli ingest sql-drivers    # list the supported SQL drivers
    python -m cli ingest sql-inspect DB  # show a database's schema (no rows read)
    python -m cli ingest sql-preview DB --table T  # preview how rows will be stored
    python -m cli ingest sql-import DB --job J     # resumable SQLite import
    python -m cli ingest sql-status J   # one SQL import job's status
    python -m cli ingest sql-resume J   # continue an interrupted job
    python -m cli ingest sql-cancel J   # ask a running job to stop safely
    python -m cli studio status         # Code Studio: workspaces + coding-model status
    python -m cli studio new "task"     # make an isolated workspace (a git worktree)
    python -m cli studio ask <slug> "…" # ask qwen2.5-coder:7b to change the workspace
    python -m cli studio diff <slug>    # review the change as a unified diff
    python -m cli studio check <slug>   # run the project's tests inside the workspace
    python -m cli studio merge <slug>   # merge it into the project (gated by checks)
    python -m cli presentation new "T"  # Presentation Studio: a new deck project
    python -m cli presentation outline P --text-file plan.txt   # set the outline
    python -m cli presentation approve-outline P   # approve it, then draft slides
    python -m cli presentation draft P  # ask the assistant for slide content
    python -m cli presentation approve-content P   # approve the content
    python -m cli presentation export P --output deck.pptx      # write the .pptx
    python -m cli deps                  # show which optional features are installed
    python -m cli deps --install ocr    # install the OCR packages (scanned PDFs)
    python -m cli paths               # show asset/data locations
    python -m cli demo                # local cognitive demo
    python -m cli ml --episodes 60    # ML pipeline
    python -m cli llm --steps 200     # language-model trainer
    python -m cli petroleum           # petroleum-engineering task training
    python -m cli curriculum          # show the seeded degree curriculum
    python -m cli release             # train everything for a release
    python -m cli eval                # evaluation suite
    python -m cli update              # check GitHub for program updates
    python -m cli update --apply      # download and apply available updates

The commands mirror the modules developers already use, so nothing about the
in-repo workflow changes.
"""

from __future__ import annotations

import sys

from cli.menu import (
    describe_paths,
    menu_loop,
    missing_dependency_message,
    run_agent,
    run_analogy,
    run_audio_speech,
    run_banner,
    run_banner_chat,
    run_characters,
    run_chat,
    run_cloud,
    run_cloud_chat,
    run_code_studio,
    run_curriculum,
    run_demo,
    run_deps,
    run_docs,
    run_embeddings,
    run_evaluation,
    run_gui,
    run_hakeem,
    run_hakeem_chat,
    run_image,
    run_import_center,
    run_ingest,
    run_insight,
    run_internet_learner,
    run_jarvis,
    run_knowledge,
    run_learn,
    run_llm,
    run_mind_chat,
    run_ml,
    run_model_chat,
    run_natasha,
    run_petroleum,
    run_piper_speech,
    run_presentation,
    run_qwen_speech,
    run_relations,
    run_release,
    run_scene3d,
    run_selfcheck,
    run_speak,
    run_ultron,
    run_ultron_chat,
    run_update,
    run_vision,
    run_vision_chat,
    run_voice,
    run_voice_studio,
    run_web,
)

_COMMANDS = {
    "voice": run_voice,
    "speak": run_speak,
    "voices": run_speak,
    "qwen-asr": lambda argv: run_qwen_speech(("qwen-asr", *argv)),
    "qwen-tts": lambda argv: run_qwen_speech(("qwen-tts", *argv)),
    "qwen-speech": run_qwen_speech,
    "piper": run_piper_speech,
    "audio": run_audio_speech,
    "audio.cpp": run_audio_speech,
    "agent": run_agent,
    "jarvis": run_jarvis,
    "ultron": run_ultron,
    "vision": run_vision,
    "banner": run_banner,
    "cloud": run_cloud,
    "natasha": run_natasha,
    "hakeem": run_hakeem,
    "chat": run_chat,
    "gui": run_gui,
    "mind-chat": run_mind_chat,
    "ultron-chat": run_ultron_chat,
    "vision-chat": run_vision_chat,
    "banner-chat": run_banner_chat,
    "cloud-chat": run_cloud_chat,
    "hakeem-chat": run_hakeem_chat,
    "learn": run_learn,
    "insight": run_insight,
    "internet-learner": run_internet_learner,
    "web": run_web,
    "knowledge": run_knowledge,
    "ingest": run_ingest,
    "import-center": run_import_center,
    "studio": run_code_studio,
    "presentation": run_presentation,
    "docs": run_docs,
    "analogy": run_analogy,
    "embeddings": run_embeddings,
    "relations": run_relations,
    "scene3d": run_scene3d,
    "scene": run_scene3d,
    "characters": run_characters,
    "character": run_characters,
    "model-chat": run_model_chat,
    "mouths": run_model_chat,
    "voice-studio": run_voice_studio,
    "voice-studios": run_voice_studio,
    "speech-studio": run_voice_studio,
    "image": run_image,
    "deps": run_deps,
    "demo": run_demo,
    "ml": run_ml,
    "llm": run_llm,
    "petroleum": run_petroleum,
    "curriculum": run_curriculum,
    "release": run_release,
    "eval": run_evaluation,
    "evaluation": run_evaluation,
    "selfcheck": run_selfcheck,
    "update": run_update,
}


def _use_utf8_console() -> None:
    """Make stdout/stderr tolerate Arabic on legacy terminals (e.g. Windows cp1252).

    The project prints Arabic labels, but a frozen Windows build inherits a
    code-page encoding that cannot represent them, which would crash the
    portable bundle. Reconfiguring to UTF-8 with a safe fallback keeps output
    readable instead of raising ``UnicodeEncodeError``.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    _use_utf8_console()
    # First run after an upgrade: pull any pre-existing in-repo training under
    # the git-ignored data folder, so it survives and never dirties the tree.
    from core.paths import ensure_data_layout

    ensure_data_layout()
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return menu_loop()
    command, *rest = args
    if command in {"paths", "info"}:
        print(describe_paths())
        return 0
    if command in {"-h", "--help", "help"}:
        print(__doc__)
        return 0
    action = _COMMANDS.get(command)
    if action is None:
        print(f"Unknown command: {command!r}\n")
        print(__doc__)
        return 1
    try:
        return action(rest)
    except ModuleNotFoundError as error:
        print(missing_dependency_message(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
