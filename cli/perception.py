"""The perception CLI (``python -m cli perception …``): the headless twin.

It drives the **same** modules the coordinator and the bridge use, so the CLI and
any window share one router, one set of pipelines, and one cognitive bridge -- no
perception logic is reimplemented here.

Commands:

* ``status`` -- what this build can perceive right now, engine by engine.
* ``detect <path>`` -- file type detection and integrity (no content read).
* ``read <path>`` -- perceive a document/PDF/Word/text file.
* ``classify <pdf>`` -- whether a PDF is text, scanned, or mixed.
* ``ocr <image>`` -- OCR an image.
* ``describe <image>`` -- ask the local vision model to describe an image.
* ``audio <file>`` -- audio metadata, and a transcript unless ``--no-transcribe``.
* ``speak <text>`` -- render text to a WAV with a local engine.
* ``ingest <path>`` -- perceive a file *and* submit it to the Cognitive Core.
* ``ingest-text <text>`` -- ingest a bare string through the same bridge.

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` an engine/model is not
available, ``4`` a gate refused the operation, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
import json
import sys

from perception.coordinator import PerceptionCoordinator
from perception.detector import detect_file, validate_integrity
from perception.schema import PerceptionResult
from perception.validator import perception_status

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3
EXIT_REFUSED = 4
EXIT_FAILURE = 5


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cli perception",
        description="Local multimodal perception: documents, images, audio, and speech.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="what this build can perceive right now")

    p_detect = sub.add_parser("detect", help="detect a file's type and integrity")
    p_detect.add_argument("path")
    p_detect.add_argument("--json", action="store_true")

    p_read = sub.add_parser("read", help="perceive a document, PDF, Word, or text file")
    p_read.add_argument("path")
    p_read.add_argument("--no-ocr", action="store_true", help="refuse OCR")
    p_read.add_argument("--ocr", action="store_true", help="force OCR on")
    p_read.add_argument("--languages", default="eng+ara")
    p_read.add_argument("--normalize", action="store_true")
    p_read.add_argument("--json", action="store_true")

    p_classify = sub.add_parser("classify", help="classify a PDF (text/scanned/mixed)")
    p_classify.add_argument("path")
    p_classify.add_argument("--json", action="store_true")

    p_ocr = sub.add_parser("ocr", help="OCR a plain image file")
    p_ocr.add_argument("path")
    p_ocr.add_argument("--languages", default="eng+ara")
    p_ocr.add_argument("--preprocess", action="store_true")
    p_ocr.add_argument("--region", default="")
    p_ocr.add_argument("--json", action="store_true")

    p_describe = sub.add_parser("describe", help="describe an image with the local vision model")
    p_describe.add_argument("path")
    p_describe.add_argument("--question", default="Describe this image in one or two sentences.")
    p_describe.add_argument("--json", action="store_true")

    p_audio = sub.add_parser("audio", help="read audio metadata and (optionally) transcribe")
    p_audio.add_argument("path")
    p_audio.add_argument("--no-transcribe", action="store_true")
    p_audio.add_argument("--language", default="")
    p_audio.add_argument("--json", action="store_true")

    p_speak = sub.add_parser("speak", help="render text to a WAV with a local engine")
    p_speak.add_argument("text")
    p_speak.add_argument("--voice", default="")
    p_speak.add_argument("--output", default="")
    p_speak.add_argument("--json", action="store_true")

    p_ingest = sub.add_parser("ingest", help="perceive a file and submit it to the Cognitive Core")
    p_ingest.add_argument("path")
    p_ingest.add_argument("--remember", action="store_true", help="also capture an episodic memory")
    p_ingest.add_argument("--no-claims", action="store_true", help="do not submit candidate relations")
    p_ingest.add_argument("--json", action="store_true")

    p_text = sub.add_parser("ingest-text", help="ingest a bare string through the same bridge")
    p_text.add_argument("text")
    p_text.add_argument("--json", action="store_true")

    return parser


def _languages(value: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in value.replace(",", "+").split("+") if part.strip())
    return parts or ("eng", "ara")


def _print_result(result: PerceptionResult, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result.to_primitive(), ensure_ascii=False, indent=2))
        return
    print(result.describe())
    if result.content:
        print("-" * 60)
        print(result.content)
    if result.metadata:
        print("-" * 60)
        print("metadata: " + json.dumps(result.metadata, ensure_ascii=False))


def _read(path: str, args: argparse.Namespace) -> int:
    ocr = True if args.ocr else (False if args.no_ocr else None)
    result = PerceptionCoordinator().perceive(
        path, ocr=ocr, languages=_languages(args.languages), normalize=args.normalize
    )
    _print_result(result, as_json=args.json)
    if result.status.value in ("failed", "unsupported"):
        return EXIT_UNAVAILABLE if "not installed" in " ".join(result.errors) else EXIT_FAILURE
    return EXIT_OK


def _detect(path: str, args: argparse.Namespace) -> int:
    detected = detect_file(path)
    report = validate_integrity(path)
    if args.json:
        print(
            json.dumps(
                {
                    "kind": detected.kind.value,
                    "media_type": detected.media_type,
                    "suffix": detected.suffix,
                    "sniffed": detected.sniffed,
                    "readable": report.ok,
                    "reason": report.reason,
                    "size_bytes": report.size_bytes,
                    "digest": report.digest,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(detected.describe())
        print(report.describe())
    return EXIT_OK if report.ok else EXIT_FAILURE


def _classify(path: str, args: argparse.Namespace) -> int:
    from perception.pdf import PdfPerceptionError, classify_pdf

    try:
        kind = classify_pdf(path)
    except PdfPerceptionError as error:
        print(f"Could not classify {path}: {error}", file=sys.stderr)
        return EXIT_FAILURE
    if args.json:
        print(json.dumps({"kind": kind, "path": path}, ensure_ascii=False))
    else:
        print(f"{path}: {kind} PDF")
        if kind in ("scanned", "mixed"):
            print("  (OCR is needed for the pages without a text layer)")
    return EXIT_OK


def _ocr(path: str, args: argparse.Namespace) -> int:
    from perception.ocr import ocr_available, ocr_install_hint, perceive_image_ocr

    if not ocr_available():
        print(ocr_install_hint(), file=sys.stderr)
        return EXIT_UNAVAILABLE
    result = perceive_image_ocr(
        path,
        languages=_languages(args.languages),
        preprocess=args.preprocess,
        region=args.region,
    )
    _print_result(result, as_json=args.json)
    return EXIT_OK if result.ok else EXIT_FAILURE


def _describe(path: str, args: argparse.Namespace) -> int:
    from perception.images import describe_image_with_model

    result = describe_image_with_model(path, question=args.question)
    _print_result(result, as_json=args.json)
    if result.status.value == "failed":
        return EXIT_UNAVAILABLE
    return EXIT_OK


def _audio(path: str, args: argparse.Namespace) -> int:
    from perception.audio import perceive_audio

    result = perceive_audio(path, transcribe=not args.no_transcribe, language=args.language)
    _print_result(result, as_json=args.json)
    if result.status.value == "failed":
        return EXIT_UNAVAILABLE
    return EXIT_OK


def _speak(args: argparse.Namespace) -> int:
    from perception.speech import SpeechPerceptionError, synthesize_to_file

    try:
        output = synthesize_to_file(args.text, voice=args.voice, output=args.output or None)
    except SpeechPerceptionError as error:
        print(str(error), file=sys.stderr)
        return EXIT_FAILURE
    if args.json:
        print(
            json.dumps(
                {
                    "path": str(output.path) if output.path else "",
                    "engine": output.engine,
                    "voice": output.voice,
                    "language": output.language,
                    "duration_seconds": output.duration_seconds,
                    "silent": output.silent,
                },
                ensure_ascii=False,
            )
        )
    else:
        print(output.describe())
        if output.path:
            print(f"written to {output.path}")
    return EXIT_OK


def _ingest(path: str, args: argparse.Namespace) -> int:
    from core.perception_bridge import default_bridge

    result = PerceptionCoordinator().perceive(path)
    if result.status.value in ("failed", "unsupported"):
        _print_result(result, as_json=args.json)
        return EXIT_FAILURE
    outcome = default_bridge().ingest(
        result, submit_claims=not args.no_claims, remember=args.remember
    )
    if args.json:
        print(
            json.dumps(
                {
                    "result": result.to_primitive(),
                    "outcome": {
                        "documents": list(outcome.documents),
                        "evidence_ids": list(outcome.evidence_ids),
                        "candidate_claims": outcome.candidate_claims,
                        "accepted_claims": outcome.accepted_claims,
                        "review_claims": outcome.review_claims,
                        "rejected_claims": outcome.rejected_claims,
                        "memory_ids": list(outcome.memory_ids),
                        "skipped_reason": outcome.skipped_reason,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        _print_result(result, as_json=False)
        print("-" * 60)
        print(outcome.describe())
    return EXIT_OK


def _ingest_text(args: argparse.Namespace) -> int:
    from core.perception_bridge import default_bridge

    result = PerceptionCoordinator().perceive_text(args.text)
    outcome = default_bridge().ingest(result, submit_claims=True)
    if args.json:
        print(json.dumps({"outcome": outcome.describe()}, ensure_ascii=False))
    else:
        print(result.summary())
        print(outcome.describe())
    return EXIT_OK


def _status(args: argparse.Namespace) -> int:
    report = perception_status()
    print(report.describe())
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Run the perception CLI; returns the process exit code."""
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.command:
        parser.print_help()
        return EXIT_USAGE
    handlers = {
        "status": lambda a: _status(a),
        "detect": lambda a: _detect(a.path, a),
        "read": lambda a: _read(a.path, a),
        "classify": lambda a: _classify(a.path, a),
        "ocr": lambda a: _ocr(a.path, a),
        "describe": lambda a: _describe(a.path, a),
        "audio": lambda a: _audio(a.path, a),
        "speak": _speak,
        "ingest": lambda a: _ingest(a.path, a),
        "ingest-text": _ingest_text,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:  # pragma: no cover - a user interrupt
        print("interrupted.", file=sys.stderr)
        return EXIT_FAILURE
    except Exception as error:  # never dump a bare traceback on the user
        print(f"perception failed: {error}", file=sys.stderr)
        return EXIT_FAILURE


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
