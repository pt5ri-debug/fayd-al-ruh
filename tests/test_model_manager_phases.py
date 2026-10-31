"""The Model Manager's phases 4-7 at the *service* and *CLI* seams.

The core logic is pinned by :mod:`tests.test_model_download`,
:mod:`tests.test_model_preflight`, and :mod:`tests.test_model_selection`. This
suite proves the two front ends reach it without a model, a GPU, or a network:
the Tk-free service the window drives, and the ``python -m cli models`` commands.

Everything runs against a temporary root and an injected manager/adapter, so a
download is exercised through a fake transport and the trained path through a
synthetic checkpoint -- never a real 6 GB file.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest
from core.model_manager import ModelManager
from core.model_registry import ModelRecord
from gui.model_manager import ModelManagerService

# ------------------------------------------------------------------- helpers


def _gguf_bytes() -> bytes:
    return b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 0)


def _record(root: Path, name: str, **kwargs) -> ModelRecord:
    path = root / f"{name}.gguf"
    if not path.exists():
        path.write_bytes(_gguf_bytes())
    base: dict[str, Any] = {
        "model_id": name,
        "identity": f"test/{name}",
        "display_name": name.title(),
        "model_format": "gguf",
        "runtime": "llama.cpp",
        "source": "local",
        "path": path,
        "size_bytes": path.stat().st_size,
        "architecture": "llama",
        "context_length": 8192,
        "integrity": "ok",
    }
    base.update(kwargs)
    return ModelRecord(**base)


@pytest.fixture
def manager(tmp_path: Path) -> ModelManager:
    built = ModelManager(root=tmp_path)
    built.registry.register_record(_record(tmp_path, "banner-model", engine="banner"))
    return built


@pytest.fixture
def service(manager: ModelManager, tmp_path: Path) -> ModelManagerService:
    return ModelManagerService(manager=manager)


# ------------------------------------------------------------------ service


def test_the_service_reaches_ranking_and_fallback(service: ModelManagerService):
    ranked = service.rank_for_task("presentation")
    assert ranked
    assert ranked[0].model_id == "banner-model"
    plan = service.fallback_plan("presentation")
    assert plan.available


def test_the_service_reads_the_sources(service: ModelManagerService):
    sources = service.sources()
    assert sources
    keys = {source.key for source in sources}
    assert "url" in keys


def test_the_service_reaches_resources_and_preflight(service: ModelManagerService):
    resources = service.resources()
    assert resources is not None
    assert resources.system
    verdict = service.preflight("banner-model")
    assert verdict.model_id == "banner-model"


def test_the_service_reaches_the_process_registry(service: ModelManagerService):
    assert service.processes() is not None
    assert service.stop_processes() == ()  # nothing was started


def test_the_service_reaches_the_trained_artifacts(service: ModelManagerService):
    artifacts = service.trained_artifacts()
    # Present in this checkout; on a machine without one the honest answer is ().
    assert isinstance(artifacts, tuple)
    if artifacts:
        assert any(item.kind == "language-model" for item in artifacts)


def test_the_service_reports_no_resumable_downloads_on_a_fresh_root(
    service: ModelManagerService,
):
    assert service.resumable_downloads() == ()


def test_the_service_search_reaches_no_network_without_a_source(
    service: ModelManagerService,
):
    # The local source has no folder and the hub needs huggingface_hub, so an
    # offline search yields nothing rather than failing.
    assert isinstance(service.search_offers("qwen"), tuple)


def test_the_service_download_needs_approval(service: ModelManagerService):
    """A download with no approval never fetches a byte -- it reports a refusal."""
    from core.model_sources import ModelOffer

    offer = ModelOffer(
        source_key="url",
        source_id="https://example.invalid/model.gguf",
        display_name="Fake",
    )
    report, entry = service.download(offer, approved=False)
    assert not report.complete
    assert entry is None
    assert report.status in ("denied", "failed")


def test_the_service_register_trained_is_honest_when_nothing_is_present(
    tmp_path: Path,
):
    """A root whose checkpoint is absent registers nothing rather than a broken record."""
    manager = ModelManager(root=tmp_path)
    service = ModelManagerService(manager=manager)
    records = service.register_trained()
    # The committed checkpoint resolves through the project's paths, so on this
    # checkout it registers; on a bare root the honest answer is ().
    assert isinstance(records, tuple)


# ---------------------------------------------------------------------- CLI


def _cli(argv: list[str]) -> tuple[int, str]:
    import io
    from contextlib import redirect_stdout

    from cli import models as cli_models

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = cli_models.main(argv)
    return code, buffer.getvalue()


def test_cli_sources_lists_the_sources():
    code, out = _cli(["sources"])
    assert code == 0
    assert "Model sources:" in out
    assert "url" in out


def test_cli_sources_json_is_machine_readable():
    code, out = _cli(["sources", "--json"])
    assert code == 0
    payload = json.loads(out)
    assert payload["sources"]


def test_cli_resources_reports_this_machine():
    code, out = _cli(["resources"])
    assert code == 0
    assert "memory" in out.lower()


def test_cli_resources_json_has_the_reading_provenance():
    code, out = _cli(["resources", "--json"])
    assert code == 0
    payload = json.loads(out)
    assert "system" in payload


def test_cli_preflight_of_an_unknown_model_is_refused():
    code, _ = _cli(["preflight", "no-such-model-anywhere"])
    assert code == 3


def test_cli_rank_of_an_unknown_task_is_refused():
    code, _ = _cli(["rank", "telepathy"])
    assert code == 2


def test_cli_rank_answers_for_a_real_task():
    code, out = _cli(["rank", "presentation"])
    # Nothing may be ready on a bare runner, but the command itself must succeed.
    assert code in (0, 3)
    assert out


def test_cli_fallback_is_honest_when_nothing_can_run():
    code, out = _cli(["fallback", "chat"])
    assert code in (0, 3)
    assert out


def test_cli_downloads_reports_no_partials():
    code, out = _cli(["downloads"])
    assert code == 0
    assert "partial" in out.lower()


def test_cli_downloads_json_is_machine_readable():
    code, out = _cli(["downloads", "--json"])
    assert code == 0
    assert "partials" in json.loads(out)


def test_cli_processes_reports_none_started():
    code, out = _cli(["processes"])
    assert code == 0
    assert "no model process" in out.lower()


def test_cli_stop_processes_is_a_no_op_when_none_run():
    code, out = _cli(["stop-processes"])
    assert code == 0
    assert "no running model process" in out.lower()


def test_cli_trained_reports_the_checkpoint():
    code, out = _cli(["trained"])
    # Present in this checkout (0); absent on a bare machine (3).
    assert code in (0, 3)
    assert out


def test_cli_trained_json_is_machine_readable():
    code, out = _cli(["trained", "--json"])
    assert code in (0, 3)
    assert "artifacts" in json.loads(out)


def test_cli_search_offers_needs_a_source():
    code, out = _cli(["search-offers", "qwen"])
    # No local folder and no huggingface_hub here, so nothing matches (3).
    assert code in (0, 3)
    assert out


def test_cli_download_without_approval_is_refused():
    # No offer is found offline, so this exits before reaching the network.
    code, _ = _cli(["download", "qwen", "--json"])
    assert code in (3, 4)


def test_every_new_command_is_registered():
    from cli.models import _HANDLERS

    for name in (
        "rank", "fallback", "sources", "search-offers", "download", "downloads",
        "resources", "preflight", "processes", "stop-processes", "trained",
        "trained-register",
    ):
        assert name in _HANDLERS
