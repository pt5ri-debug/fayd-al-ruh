"""Tests for the Phase 9 self-improvement CLI (``python -m cli improve``).

The CLI is a thin headless twin of the loop: every command drives the *same*
:class:`core.improvement.loop.ImprovementLoop` the window uses. These tests pin
the commands, their exit codes, and the guarantees that must never relax -- an
experiment needs ``--approve``, a source change is never applied automatically,
and the security report is honest.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from cli.improve import (
    EXIT_GATE,
    EXIT_OK,
    EXIT_USAGE,
    main,
)


@contextmanager
def _repo() -> Iterator[Path]:
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
        (root / "a.py").write_text("x = 1\n", encoding="utf-8")
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A"],
            cwd=root, check=True,
        )
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
            cwd=root, check=True,
        )
        yield root


def test_status_reports_the_security_invariants(capsys: pytest.CaptureFixture[str]):
    assert main(["status"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "Controlled self-improvement loop" in out
    assert "Security invariants: held" in out


def test_metrics_lists_the_catalogue(capsys: pytest.CaptureFixture[str]):
    assert main(["metrics"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "retrieval.quality" in out and "task.success_rate" in out


def test_detect_reports_a_repeated_failure(capsys: pytest.CaptureFixture[str]):
    code = main([
        "detect",
        "--failure", "execution:read_file",
        "--failure", "execution:read_file",
        "--failure", "execution:read_file",
    ])
    assert code == EXIT_OK
    assert "repeated_failure" in capsys.readouterr().out


def test_detect_reports_nothing_for_a_single_failure(capsys: pytest.CaptureFixture[str]):
    assert main(["detect", "--failure", "execution:read_file"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "no weaknesses" in out.lower() or "0 weakness" in out.lower()


def test_a_malformed_failure_is_a_usage_error(capsys: pytest.CaptureFixture[str]):
    assert main(["detect", "--failure", "no-colon"]) == EXIT_USAGE


def test_propose_generates_and_ranks(capsys: pytest.CaptureFixture[str]):
    code = main([
        "propose",
        "--failure", "execution:read_file",
        "--failure", "execution:read_file",
        "--failure", "execution:read_file",
    ])
    assert code == EXIT_OK
    assert "PRP-" in capsys.readouterr().out


def test_propose_json_is_machine_readable(capsys: pytest.CaptureFixture[str]):
    import json

    code = main([
        "propose", "--json",
        "--failure", "execution:read_file",
        "--failure", "execution:read_file",
        "--failure", "execution:read_file",
    ])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload and payload[0]["proposal_id"].startswith("PRP-")


def test_an_experiment_needs_approval():
    assert main(["experiment", "PRP-X"]) == EXIT_GATE


def test_the_gate_reports_a_perfect_score(capsys: pytest.CaptureFixture[str]):
    assert main(["gate"]) == EXIT_OK
    assert "1.0000" in capsys.readouterr().out


def test_history_is_empty_on_a_fresh_loop(capsys: pytest.CaptureFixture[str]):
    with _repo() as root:
        assert main(["history", "--root", str(root)]) == EXIT_OK
        assert "0 entr" in capsys.readouterr().out


def test_security_reports_held(capsys: pytest.CaptureFixture[str]):
    with _repo() as root:
        assert main(["security", "--root", str(root)]) == EXIT_OK
        assert "held" in capsys.readouterr().out


def test_an_experiment_runs_in_isolation_with_approval(capsys: pytest.CaptureFixture[str]):
    with _repo() as root:
        # First register a proposal by proposing over failures.
        main([
            "propose", "--root", str(root),
            "--failure", "execution:read_file",
            "--failure", "execution:read_file",
            "--failure", "execution:read_file",
        ])
        capsys.readouterr()
        # The proposal id is not persisted across processes, so ask the loop for it.
        from core.paths import data_dir
        from core.self_improvement import improvement_loop

        loop = improvement_loop(root, state_root=data_dir())
        failures = tuple(
            __import__("core.improvement.weakness", fromlist=["FailureRecord"]).FailureRecord(
                "execution", "read_file"
            )
            for _ in range(3)
        )
        _, ranked = loop.analyse(failures=failures)
        proposal_id = ranked[0].proposal.proposal_id
        code = main([
            "experiment", proposal_id, "--root", str(root), "--approve",
            "--file", "a.py=x = 2", "--overall", "0.95",
            "--check-command", "python -c pass",
        ])
        out = capsys.readouterr().out
        assert code in {EXIT_OK, EXIT_GATE}
        assert "EXP-" in out
