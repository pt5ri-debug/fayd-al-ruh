"""The Training Center CLI: the headless twin of the window.

It drives the *same* :class:`gui.training_center.TrainingCenterService` the window
drives, so the CLI and the window share one review queue, one dataset registry,
one run log, and one history -- no workflow logic is reimplemented here.

Commands:

* ``dashboard`` -- the dashboard's real figures, read from the stores;
* ``catalog`` -- every control, what it does, and the command that runs it;
* ``settings [--set key=value]`` -- the Training Center's own preferences;
* ``import <path>...`` -- import files or folders into the knowledge base;
* ``sources --url <url> --approve`` -- the governed web path;
* ``knowledge [--query Q] [--reindex] [--export DIR] [--stats]`` -- the knowledge
  base;
* ``ask --question Q [--provider P]`` / ``ask --topic T [--count N]`` -- gather
  answers into the review queue;
* ``review [--pending] [--approve ID] [--reject ID] [--correct ID --answer A]``;
* ``export --dataset ID`` -- build a governed dataset from approved examples;
* ``datasets [--validate ID] [--export ID DIR]`` -- the Dataset Studio;
* ``jobs [--cancel ID] [--feasibility]`` -- the training jobs;
* ``evaluate <run_id>`` -- evaluate a run on its held-out split;
* ``pipeline [--list] [--create ID] [--run ID --approve]``;
* ``history [--kind K] [--query Q]``;
* ``resources`` -- this machine's own figures;
* ``gate`` -- the held-out gate report.

The **language model** (the one home of every training surface):

* ``lm-status`` -- the checkpoint the project trains and the corpus it uses;
* ``lm-settings [--set K=V] [--reset]`` -- the whole training option set;
* ``lm-corpus`` -- how much text a run would train on, source by source;
* ``lm-train [--approve] [--fresh] [--shards N] [overrides…]`` -- train or continue;
* ``lm-evaluate`` -- perplexity and bits per byte on held-out data;
* ``lm-generate --prompt P`` -- sample a continuation from the model;
* ``lm-release [--approve]`` -- train every learning layer and write the manifest;
* ``lm-weights --install KEY | --import FILE | --fetch URL | --build TIER | --list``;
* ``lm-plan`` -- the headless command equal to the settings in force.

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` the model, dataset, or
run is unknown, ``4`` a gate refused the operation, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

#: The real stdout, captured before any trainer redirects it, so a live progress
#: line reaches the terminal instead of being swallowed by the redirect.
_STDOUT = sys.stdout

if TYPE_CHECKING:
    from gui.training_center import TrainingCenterService

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3
EXIT_GATE = 4
EXIT_FAILURE = 5


def _service(root: str = "") -> TrainingCenterService:
    from gui.training_center import TrainingCenterService

    return TrainingCenterService(Path(root)) if root else TrainingCenterService()


def _fail(message: str, code: int) -> int:
    print(message)
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cli training-center",
        description="The Training Center: learn from sources, build governed datasets, "
        "train, and evaluate -- all over the project's own machinery.",
    )
    parser.add_argument("--root", default="", help="the data root (default: data/)")
    sub = parser.add_subparsers(dest="command")

    p_dash = sub.add_parser("dashboard", help="the dashboard's real figures")
    p_dash.add_argument("--json", action="store_true")
    p_dash.set_defaults(func=_cmd_dashboard)

    p_cat = sub.add_parser("catalog", help="what every control does")
    p_cat.add_argument("--json", action="store_true")
    p_cat.set_defaults(func=_cmd_catalog)

    p_set = sub.add_parser("settings", help="the Training Center's own preferences")
    p_set.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p_set.add_argument("--json", action="store_true")
    p_set.set_defaults(func=_cmd_settings)

    p_web = sub.add_parser("web", help="the internet switch (cloud, URL import, KB build)")
    p_web.add_argument("--on", action="store_true")
    p_web.add_argument("--off", action="store_true")
    p_web.add_argument("--json", action="store_true")
    p_web.set_defaults(func=_cmd_web)

    p_prov = sub.add_parser("providers", help="the answer providers (for a dropdown)")
    p_prov.add_argument("--json", action="store_true")
    p_prov.set_defaults(func=_cmd_providers)

    p_imp = sub.add_parser("import", help="import files or folders")
    p_imp.add_argument("paths", nargs="+")
    p_imp.add_argument("--json", action="store_true")
    p_imp.set_defaults(func=_cmd_import)

    p_src = sub.add_parser("sources", help="learn from the internet (governed)")
    p_src.add_argument("--url", action="append", default=[])
    p_src.add_argument("--approve", action="store_true")
    p_src.add_argument("--json", action="store_true")
    p_src.set_defaults(func=_cmd_sources)

    p_kn = sub.add_parser("knowledge", help="the knowledge base")
    p_kn.add_argument("--query", default="")
    p_kn.add_argument("--reindex", action="store_true")
    p_kn.add_argument("--export", default="")
    p_kn.add_argument("--stats", action="store_true")
    p_kn.add_argument("--json", action="store_true")
    p_kn.set_defaults(func=_cmd_knowledge)

    p_ask = sub.add_parser("ask", help="gather answers into the review queue")
    p_ask.add_argument("--question", default="")
    p_ask.add_argument("--topic", default="")
    p_ask.add_argument("--count", type=int, default=5)
    p_ask.add_argument("--provider", default="")
    p_ask.add_argument("--json", action="store_true")
    p_ask.set_defaults(func=_cmd_ask)

    p_rev = sub.add_parser("review", help="review gathered answers")
    p_rev.add_argument("--pending", action="store_true")
    p_rev.add_argument("--all", action="store_true")
    p_rev.add_argument("--approve", default="")
    p_rev.add_argument("--reject", default="")
    p_rev.add_argument("--correct", default="")
    p_rev.add_argument("--answer", default="")
    p_rev.add_argument("--json", action="store_true")
    p_rev.set_defaults(func=_cmd_review)

    p_exp = sub.add_parser("export", help="build a dataset from approved examples")
    p_exp.add_argument("--dataset", required=True)
    p_exp.add_argument("--json", action="store_true")
    p_exp.set_defaults(func=_cmd_export)

    p_ds = sub.add_parser("datasets", help="the Dataset Studio")
    p_ds.add_argument("--validate", default="")
    p_ds.add_argument("--export", nargs=2, default=None, metavar=("ID", "DIR"))
    p_ds.add_argument("--json", action="store_true")
    p_ds.set_defaults(func=_cmd_datasets)

    p_jobs = sub.add_parser("jobs", help="the training jobs")
    p_jobs.add_argument("--cancel", default="")
    p_jobs.add_argument("--json", action="store_true")
    p_jobs.set_defaults(func=_cmd_jobs)

    p_eval = sub.add_parser("evaluate", help="evaluate a run on its held-out split")
    p_eval.add_argument("run_id")
    p_eval.add_argument("--json", action="store_true")
    p_eval.set_defaults(func=_cmd_evaluate)

    p_pipe = sub.add_parser("pipeline", help="the learning pipelines")
    p_pipe.add_argument("--list", action="store_true")
    p_pipe.add_argument("--create", default="")
    p_pipe.add_argument("--run", default="")
    p_pipe.add_argument("--approve", action="store_true")
    p_pipe.add_argument("--json", action="store_true")
    p_pipe.set_defaults(func=_cmd_pipeline)

    p_hist = sub.add_parser("history", help="the durable history")
    p_hist.add_argument("--kind", default="")
    p_hist.add_argument("--query", default="")
    p_hist.add_argument("--limit", type=int, default=30)
    p_hist.add_argument("--json", action="store_true")
    p_hist.set_defaults(func=_cmd_history)

    p_res = sub.add_parser("resources", help="this machine's own figures")
    p_res.add_argument("--json", action="store_true")
    p_res.set_defaults(func=_cmd_resources)

    p_gate = sub.add_parser("gate", help="the held-out gate report")
    p_gate.add_argument("--json", action="store_true")
    p_gate.set_defaults(func=_cmd_gate)

    # -- the language model (every training surface in one place) ----------
    p_lm_status = sub.add_parser("lm-status", help="the checkpoint and the corpus it trains on")
    p_lm_status.add_argument("--json", action="store_true")
    p_lm_status.set_defaults(func=_cmd_lm_status)

    p_lm_settings = sub.add_parser("lm-settings", help="the language-model training settings")
    p_lm_settings.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p_lm_settings.add_argument("--reset", action="store_true")
    p_lm_settings.add_argument("--json", action="store_true")
    p_lm_settings.set_defaults(func=_cmd_lm_settings)

    p_lm_corpus = sub.add_parser("lm-corpus", help="how much text a run would train on")
    p_lm_corpus.add_argument("--json", action="store_true")
    p_lm_corpus.set_defaults(func=_cmd_lm_corpus)

    p_lm_train = sub.add_parser("lm-train", help="train or continue the project's own model")
    p_lm_train.add_argument("--steps", type=int, default=None)
    p_lm_train.add_argument("--preset", default="")
    p_lm_train.add_argument("--tokenizer-kind", default="", choices=["", "byte", "bpe"])
    p_lm_train.add_argument("--repeats", type=int, default=None)
    p_lm_train.add_argument("--seed", type=int, default=None)
    p_lm_train.add_argument("--weight-decay", type=float, default=None)
    p_lm_train.add_argument("--lr-schedule", default="", choices=["", "cosine", "linear", "constant"])
    p_lm_train.add_argument("--accum-steps", type=int, default=None)
    p_lm_train.add_argument("--kb-max-chunks", type=int, default=None)
    p_lm_train.add_argument("--shards", type=int, default=None)
    p_lm_train.add_argument("--sessions", action="store_true", help="train on the local mouths' own chats")
    p_lm_train.add_argument("--include-sessions", action="store_true", default=None)
    p_lm_train.add_argument("--include-characters", action="store_true", default=None)
    p_lm_train.add_argument("--include-character-corpus", action="store_true", default=None)
    p_lm_train.add_argument("--no-runtime", action="store_true")
    p_lm_train.add_argument("--fresh", action="store_true", help="start from step zero (the opt-out of resuming)")
    p_lm_train.add_argument("--approve", action="store_true", help="confirm the training run")
    p_lm_train.add_argument("--json", action="store_true")
    p_lm_train.set_defaults(func=_cmd_lm_train)

    p_lm_eval = sub.add_parser("lm-evaluate", help="measure the checkpoint on held-out data")
    p_lm_eval.add_argument("--json", action="store_true")
    p_lm_eval.set_defaults(func=_cmd_lm_evaluate)

    p_lm_gen = sub.add_parser("lm-generate", help="sample a continuation from the model")
    p_lm_gen.add_argument("--prompt", required=True)
    p_lm_gen.add_argument("--max-new-tokens", type=int, default=None)
    p_lm_gen.add_argument("--temperature", type=float, default=None)
    p_lm_gen.add_argument("--seed", type=int, default=None)
    p_lm_gen.add_argument("--json", action="store_true")
    p_lm_gen.set_defaults(func=_cmd_lm_generate)

    p_lm_release = sub.add_parser("lm-release", help="train every learning layer for a release")
    p_lm_release.add_argument("--sessions", action="store_true", help="include the local mouths' own chats")
    p_lm_release.add_argument("--approve", action="store_true", help="confirm the release run")
    p_lm_release.add_argument("--json", action="store_true")
    p_lm_release.set_defaults(func=_cmd_lm_release)

    p_lm_weights = sub.add_parser("lm-weights", help="ready-made weights: install, import, fetch, build")
    p_lm_weights.add_argument("--install", default="", help="install a committed weights-pack entry by key")
    p_lm_weights.add_argument("--import", dest="import_path", default="", metavar="FILE", help="install a .npz file")
    p_lm_weights.add_argument("--fetch", default="", metavar="URL", help="fetch weights over HTTPS (needs --approve)")
    p_lm_weights.add_argument("--build", default="", metavar="TIER", help="train a learned capacity tier")
    p_lm_weights.add_argument("--list", action="store_true", help="list the learned tiers")
    p_lm_weights.add_argument("--force", action="store_true", help="replace an existing file (a backup is kept)")
    p_lm_weights.add_argument("--approve", action="store_true")
    p_lm_weights.add_argument("--json", action="store_true")
    p_lm_weights.set_defaults(func=_cmd_lm_weights)

    p_lm_plan = sub.add_parser("lm-plan", help="the headless command equal to the settings in force")
    p_lm_plan.add_argument("--json", action="store_true")
    p_lm_plan.set_defaults(func=_cmd_lm_plan)

    return parser


# -- commands -------------------------------------------------------------


def _cmd_dashboard(args: argparse.Namespace) -> int:
    snapshot = _service(args.root).dashboard()
    if args.json:
        print(json.dumps(snapshot.to_primitive(), indent=2, ensure_ascii=False))
        return EXIT_OK
    print(snapshot.describe())
    for chart in snapshot.charts:
        print(f"  {chart.describe()}")
    if snapshot.recent_history:
        print("\nRecent activity:")
        for line in snapshot.recent_history:
            print(f"  {line}")
    return EXIT_OK


def _cmd_catalog(args: argparse.Namespace) -> int:
    from core.training_center.catalog import CAPABILITIES

    if args.json:
        payload = [
            {
                "cli": cap.cli,
                "governed": cap.governed,
                "key": cap.key,
                "outcome": cap.outcome,
                "section": cap.section,
                "summary": cap.summary,
                "title": cap.title,
                "title_ar": cap.title_ar,
                "writes": cap.writes,
            }
            for cap in CAPABILITIES
        ]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return EXIT_OK
    print(_service(args.root).catalog_text())
    return EXIT_OK


def _cmd_settings(args: argparse.Namespace) -> int:
    service = _service(args.root)
    for pair in args.set:
        if "=" not in pair:
            return _fail(f"invalid: expected KEY=VALUE, got {pair!r}", EXIT_USAGE)
        key, _, value = pair.partition("=")
        parsed: object = value
        if value.lower() in ("true", "false"):
            parsed = value.lower() == "true"
        elif value.isdigit():
            parsed = int(value)
        try:
            service.set_setting(key.strip(), parsed)
        except KeyError as error:
            return _fail(f"invalid: {error}", EXIT_USAGE)
    if args.json:
        print(json.dumps(service.settings(), indent=2, ensure_ascii=False))
        return EXIT_OK
    print(service.settings_text())
    return EXIT_OK


def _cmd_web(args: argparse.Namespace) -> int:
    service = _service(args.root)
    if args.on and args.off:
        return _fail("invalid: --on and --off are mutually exclusive", EXIT_USAGE)
    if args.on:
        service.set_web(True)
    elif args.off:
        service.set_web(False)
    if args.json:
        print(json.dumps({"web": service.web_enabled(), "status": service.web_status()}, indent=2, ensure_ascii=False))
        return EXIT_OK
    print(service.web_status())
    return EXIT_OK


def _cmd_providers(args: argparse.Namespace) -> int:
    service = _service(args.root)
    choices = service.provider_choices()
    if args.json:
        print(json.dumps([{"value": v, "label": label} for v, label in choices], indent=2, ensure_ascii=False))
        return EXIT_OK
    print("Providers (value — label):")
    for value, label in choices:
        print(f"  {value} — {label}")
    print("\n" + service.web_status())
    return EXIT_OK


def _cmd_import(args: argparse.Namespace) -> int:
    service = _service(args.root)
    reports = []
    for path in args.paths:
        target = Path(path)
        if target.is_dir():
            report = service.import_folder(target)
        else:
            report = service.import_files([target])
        reports.append(report)
    if args.json:
        print(json.dumps([r.as_dict() for r in reports], indent=2, ensure_ascii=False))
        return EXIT_OK
    for report in reports:
        print(report.summary())
    return EXIT_OK if all(report.ok for report in reports) else EXIT_FAILURE


def _cmd_sources(args: argparse.Namespace) -> int:
    service = _service(args.root)
    if not args.url:
        return _fail("invalid: supply at least one --url", EXIT_USAGE)
    if not args.approve:
        return _fail(
            "refused: importing from the internet needs --approve (the web path is "
            "denied by default).",
            EXIT_GATE,
        )
    if not service.web_enabled():
        return _fail(
            "refused: the Training Center's internet switch is off; run "
            "`training-center web --on` first.",
            EXIT_GATE,
        )
    outcomes = service.import_urls(args.url, approved=True)
    if args.json:
        print(json.dumps([o.__dict__ for o in outcomes], indent=2, ensure_ascii=False))
        return EXIT_OK
    for outcome in outcomes:
        print(f"{'ok' if outcome.ok else 'failed'}  {outcome.source}: {outcome.message}")
    return EXIT_OK if all(o.ok for o in outcomes) else EXIT_FAILURE


def _cmd_knowledge(args: argparse.Namespace) -> int:
    service = _service(args.root)
    if args.reindex:
        summary = service.reindex()
        print(json.dumps(summary, indent=2, ensure_ascii=False) if args.json else f"Index: {summary}")
        return EXIT_OK
    if args.export:
        print(service.export_knowledge(args.export))
        return EXIT_OK
    if args.query:
        passages = service.knowledge_search(args.query)
        if args.json:
            print(json.dumps([{"citation": p.citation, "text": p.text} for p in passages], indent=2, ensure_ascii=False))
            return EXIT_OK
        if not passages:
            print("No passage matched.")
            return EXIT_OK
        for passage in passages:
            print(f"[{passage.citation}]\n{passage.text}\n")
        return EXIT_OK
    stats = service.knowledge_stats()
    if args.json:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        print("Knowledge base")
        for key, value in sorted(stats.items()):
            print(f"  {key}: {value}")
    return EXIT_OK


def _cmd_ask(args: argparse.Namespace) -> int:
    service = _service(args.root)
    if args.topic:
        questions = service.generate_questions(args.topic, count=args.count, provider=args.provider)
        if not questions:
            return _fail(
                "unavailable: no provider could generate questions (set a provider's key, "
                "or approve the cloud path).",
                EXIT_UNAVAILABLE,
            )
        examples = service.ask_many(questions, provider=args.provider, topic=args.topic)
    elif args.question:
        examples = (service.ask(args.question, provider=args.provider),)
    else:
        return _fail("invalid: supply --question or --topic", EXIT_USAGE)
    if args.json:
        print(json.dumps([e.to_primitive() for e in examples], indent=2, ensure_ascii=False))
        return EXIT_OK
    for example in examples:
        print(example.describe())
    print("\nThese are pending; approve them with `review --approve <id>` before export.")
    return EXIT_OK


def _cmd_review(args: argparse.Namespace) -> int:
    service = _service(args.root)
    try:
        if args.approve:
            example = service.approve(args.approve)
        elif args.reject:
            example = service.reject(args.reject)
        elif args.correct:
            if not args.answer:
                return _fail("invalid: --correct needs --answer", EXIT_USAGE)
            example = service.correct(args.correct, args.answer)
        else:
            status = "pending" if args.pending else ""
            examples = service.examples(status=status)
            if args.json:
                print(json.dumps([e.to_primitive() for e in examples], indent=2, ensure_ascii=False))
                return EXIT_OK
            if not examples:
                print("No learning examples.")
                return EXIT_OK
            for item in examples:
                print(item.describe())
            print("\n" + service.review_summary().describe())
            return EXIT_OK
    except KeyError as error:
        return _fail(f"unknown: {error}", EXIT_UNAVAILABLE)
    print(example.describe())
    return EXIT_OK


def _cmd_export(args: argparse.Namespace) -> int:
    service = _service(args.root)
    try:
        row = service.build_dataset_from_approved(args.dataset)
    except ValueError as error:
        return _fail(f"refused: {error}", EXIT_GATE)
    if args.json:
        print(json.dumps(row.__dict__, indent=2, ensure_ascii=False))
        return EXIT_OK
    print(row.describe())
    return EXIT_OK


def _cmd_datasets(args: argparse.Namespace) -> int:
    service = _service(args.root)
    if args.validate:
        print(service.validate_dataset(args.validate))
        return EXIT_OK
    if args.export:
        dataset_id, destination = args.export
        report = service.export_dataset(dataset_id, destination)
        print(report.describe())
        return EXIT_OK if report.ok else EXIT_FAILURE
    rows = service.dataset_rows()
    if args.json:
        print(json.dumps([row.__dict__ for row in rows], indent=2, ensure_ascii=False))
        return EXIT_OK
    if not rows:
        print("No datasets yet. Build one with `export --dataset <id>` after approving examples.")
        return EXIT_OK
    for row in rows:
        print(row.describe())
    return EXIT_OK


def _cmd_jobs(args: argparse.Namespace) -> int:
    service = _service(args.root)
    if args.cancel:
        cancelled = service.cancel_job(args.cancel)
        print("Cancelled." if cancelled else "No running job with that id.")
        return EXIT_OK if cancelled else EXIT_UNAVAILABLE
    rows = service.job_rows()
    if args.json:
        print(json.dumps([row.__dict__ for row in rows], indent=2, ensure_ascii=False))
        return EXIT_OK
    if not rows:
        print("No training jobs yet.")
        return EXIT_OK
    for row in rows:
        print(row.describe())
    return EXIT_OK


def _cmd_evaluate(args: argparse.Namespace) -> int:
    service = _service(args.root)
    try:
        report = service.evaluate_run(args.run_id)
    except KeyError as error:
        return _fail(f"unknown: {error}", EXIT_UNAVAILABLE)
    if args.json:
        print(json.dumps(report.to_primitive(), indent=2, ensure_ascii=False))
        return EXIT_OK
    print(report.describe())
    return EXIT_OK


def _cmd_pipeline(args: argparse.Namespace) -> int:
    service = _service(args.root)
    if args.create:
        pipeline = service.create_pipeline(args.create)
        print(pipeline.describe())
        return EXIT_OK
    if args.run:
        try:
            report = service.run_pipeline(args.run, approved=args.approve)
        except KeyError as error:
            return _fail(f"unknown: {error}", EXIT_UNAVAILABLE)
        if args.json:
            print(json.dumps(
                {
                    "ok": report.ok,
                    "pipeline_id": report.pipeline_id,
                    "stopped_at": report.stopped_at,
                    "outcomes": [o.__dict__ for o in report.outcomes],
                },
                indent=2, ensure_ascii=False,
            ))
            return EXIT_OK
        print(report.describe())
        return EXIT_OK if report.ok else EXIT_GATE
    pipelines = service.pipeline_list()
    if args.json:
        print(json.dumps([p.to_primitive() for p in pipelines], indent=2, ensure_ascii=False))
        return EXIT_OK
    if not pipelines:
        print("No saved pipelines. Create one with `pipeline --create <id>`.")
        return EXIT_OK
    for pipeline in pipelines:
        print(pipeline.describe())
    return EXIT_OK


def _cmd_history(args: argparse.Namespace) -> int:
    service = _service(args.root)
    lines = service.history_lines(kind=args.kind, query=args.query, limit=args.limit)
    if args.json:
        print(json.dumps([line.__dict__ for line in lines], indent=2, ensure_ascii=False))
        return EXIT_OK
    if not lines:
        print("No training-center history yet.")
        return EXIT_OK
    for line in lines:
        print(line.describe())
    return EXIT_OK


def _cmd_resources(args: argparse.Namespace) -> int:
    snapshot = _service(args.root).resources()
    if args.json:
        print(json.dumps(snapshot.to_primitive(), indent=2, ensure_ascii=False))
        return EXIT_OK
    print(snapshot.describe())
    return EXIT_OK


def _cmd_gate(args: argparse.Namespace) -> int:
    from core.training_center.gate import describe, run_judgements, training_center_accuracy

    if args.json:
        payload = {
            "accuracy": training_center_accuracy(),
            "judgements": [{"name": j.name, "passed": j.passed, "detail": j.detail} for j in run_judgements()],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return EXIT_OK
    print(describe())
    return EXIT_OK


# -- the language model ---------------------------------------------------


def _coerce(value: str) -> object:
    """Read a ``--set`` value as the type it looks like, so a switch really flips."""
    lowered = value.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _cmd_lm_status(args: argparse.Namespace) -> int:
    service = _service(args.root)
    if args.json:
        payload = {
            "checkpoint": service.lm_checkpoint().to_primitive(),
            "corpus": service.language_model.corpus_summary(),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return EXIT_OK
    print(service.lm_status())
    return EXIT_OK


def _cmd_lm_settings(args: argparse.Namespace) -> int:
    service = _service(args.root)
    if args.reset:
        service.lm_reset_settings()
    for pair in args.set:
        if "=" not in pair:
            return _fail(f"invalid: expected KEY=VALUE, got {pair!r}", EXIT_USAGE)
        key, _, value = pair.partition("=")
        try:
            service.lm_set_setting(key.strip(), _coerce(value))
        except KeyError as error:
            return _fail(f"invalid: {error}", EXIT_USAGE)
    if args.json:
        print(json.dumps(service.lm_settings().to_primitive(), indent=2, ensure_ascii=False))
        return EXIT_OK
    print(service.lm_settings_text())
    return EXIT_OK


def _cmd_lm_corpus(args: argparse.Namespace) -> int:
    service = _service(args.root)
    if args.json:
        print(json.dumps(service.language_model.corpus_summary(), indent=2, ensure_ascii=False))
        return EXIT_OK
    print(service.lm_corpus())
    return EXIT_OK


def _cmd_lm_train(args: argparse.Namespace) -> int:
    service = _service(args.root)
    settings = service.lm_settings()
    overrides: dict[str, object] = {}
    for field, value in (
        ("steps", args.steps),
        ("preset", args.preset or None),
        ("tokenizer_kind", args.tokenizer_kind or None),
        ("repeats", args.repeats),
        ("seed", args.seed),
        ("weight_decay", args.weight_decay),
        ("lr_schedule", args.lr_schedule or None),
        ("accum_steps", args.accum_steps),
        ("kb_max_chunks", args.kb_max_chunks),
    ):
        if value is not None:
            overrides[field] = value
    if args.include_sessions or args.sessions:
        overrides["include_sessions"] = True
    if args.include_characters:
        overrides["include_characters"] = True
    if args.include_character_corpus:
        overrides["include_character_corpus"] = True
    if args.no_runtime:
        overrides["include_runtime"] = False
    if overrides:
        settings = settings.with_values(**overrides)
    if not args.approve:
        print("Dry run. This would train with:")
        print(settings.describe())
        print("\nRe-run with --approve to train.")
        return EXIT_OK

    def on_line(line: str) -> None:
        # The engine captures stdout while the trainer runs, so a live line must
        # go to the *real* stream to be seen now rather than swallowed.
        print(line, file=_STDOUT, flush=True)

    outcome = service.lm_train(settings=settings, fresh=args.fresh, shards=args.shards, on_line=on_line)
    if args.json:
        print(json.dumps(outcome.to_primitive(), indent=2, ensure_ascii=False))
        return EXIT_OK if outcome.ok else EXIT_FAILURE
    print()
    print(outcome.describe())
    return EXIT_OK if outcome.ok else EXIT_FAILURE


def _cmd_lm_evaluate(args: argparse.Namespace) -> int:
    report = _service(args.root).lm_evaluate()
    if args.json:
        print(json.dumps(report.to_primitive(), indent=2, ensure_ascii=False))
        return EXIT_OK if report.ok else EXIT_UNAVAILABLE
    print(report.describe())
    return EXIT_OK if report.ok else EXIT_UNAVAILABLE


def _cmd_lm_generate(args: argparse.Namespace) -> int:
    service = _service(args.root)
    try:
        text = service.lm_generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
    except FileNotFoundError as error:
        return _fail(f"not available: {error}", EXIT_UNAVAILABLE)
    if args.json:
        print(json.dumps({"prompt": args.prompt, "continuation": text}, indent=2, ensure_ascii=False))
        return EXIT_OK
    print(text)
    return EXIT_OK


def _cmd_lm_release(args: argparse.Namespace) -> int:
    service = _service(args.root)
    if getattr(args, "sessions", False):
        service.lm_set_setting("include_sessions", True)
    if not args.approve:
        print("Dry run. This would train every learning layer and write the release manifest.")
        print(service.lm_settings_text())
        print("\nRe-run with --approve to run it.")
        return EXIT_OK

    def on_line(line: str) -> None:
        # The engine captures stdout while the trainer runs, so a live line must
        # go to the *real* stream to be seen now rather than swallowed.
        print(line, file=_STDOUT, flush=True)

    outcome = service.lm_release(on_line=on_line)
    if args.json:
        print(json.dumps(outcome.to_primitive(), indent=2, ensure_ascii=False))
        return EXIT_OK if outcome.ok else EXIT_FAILURE
    print()
    print(outcome.describe())
    return EXIT_OK if outcome.ok else EXIT_FAILURE


def _cmd_lm_weights(args: argparse.Namespace) -> int:
    service = _service(args.root)
    if args.list:
        text = service.lm_learned_text()
        if args.json:
            print(json.dumps({"learned": list(service.language_model.list_learned())}, indent=2, ensure_ascii=False))
            return EXIT_OK
        print(text)
        return EXIT_OK
    if args.install:
        report = service.lm_install_pack(args.install, force=args.force)
    elif args.import_path:
        report = service.lm_import_weights(args.import_path, force=args.force)
    elif args.fetch:
        report = service.lm_fetch_weights(args.fetch, approved=args.approve)
    elif args.build:
        report = service.lm_build_learned(args.build, force=args.force)
    else:
        return _fail(
            "invalid: name an action — --install KEY, --import FILE, --fetch URL, "
            "--build TIER, or --list",
            EXIT_USAGE,
        )
    if args.json:
        print(json.dumps(report.to_primitive(), indent=2, ensure_ascii=False))
        return EXIT_OK if report.ok else EXIT_GATE
    print(report.describe())
    return EXIT_OK if report.ok else EXIT_GATE


def _cmd_lm_plan(args: argparse.Namespace) -> int:
    plan = _service(args.root).language_model.plan()
    if args.json:
        print(json.dumps({"command": plan}, indent=2, ensure_ascii=False))
        return EXIT_OK
    print(plan)
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(args, "func", None) is None:
        return _cmd_dashboard(argparse.Namespace(root="", json=False))
    try:
        return int(args.func(args))
    except ValueError as error:
        return _fail(f"invalid: {error}", EXIT_USAGE)
    except FileNotFoundError as error:
        return _fail(f"not available: {error}", EXIT_UNAVAILABLE)
    except Exception as error:  # pragma: no cover - defensive
        return _fail(f"unexpected failure: {type(error).__name__}: {error}", EXIT_FAILURE)


if __name__ == "__main__":
    raise SystemExit(main())
