"""The offline guarantee (Technical Gap G-8 / finding F-6).

The project is fully local by design: every network egress is gated and off by
default, and no optional engine imports at module load. What was missing was an
*enforcement* -- a test that proves a base-install path opens no socket, and a
switch a user can set to make any network attempt fail. This module is both:

* :func:`test_guard_blocks_a_socket` / :func:`test_guard_blocks_urlopen` prove
  the ``FAYD_OFFLINE=1`` guard really raises.
* the ``_offline_guard`` fixture makes the answer path, a memory write, a
  training smoke step, and a personality exchange run with every network entry
  point blocked, and asserts each still succeeds.

The guard patches the *low-level* entry points (``socket.socket``,
``socket.create_connection``, ``socket.getaddrinfo``, ``urllib.request.urlopen``),
so it covers ``requests``/``urllib3`` and ``urllib`` alike, and it is restored
afterwards so no other test is affected.
"""

from __future__ import annotations

import socket
import urllib.request

import pytest
from core.offline import (
    NetworkDisabled,
    describe,
    ensure_offline,
    guard_active,
    install_guard,
    is_loopback_host,
    offline_requested,
    uninstall_guard,
)


@pytest.fixture
def offline_guard():
    """Guard every external network entry point for one test, then restore."""
    install_guard()
    try:
        yield
    finally:
        uninstall_guard()


# --------------------------------------------------------------- the switch

def test_offline_requested_reads_the_environment():
    assert offline_requested({"FAYD_OFFLINE": "1"}) is True
    assert offline_requested({"FAYD_OFFLINE": "true"}) is True
    assert offline_requested({"FAYD_OFFLINE": "YES"}) is True
    assert offline_requested({"FAYD_OFFLINE": "0"}) is False
    assert offline_requested({"FAYD_OFFLINE": ""}) is False
    assert offline_requested({}) is False


def test_loopback_is_not_an_external_egress():
    assert is_loopback_host("localhost") is True
    assert is_loopback_host("127.0.0.1") is True
    assert is_loopback_host("127.0.0.2") is True
    assert is_loopback_host("::1") is True
    assert is_loopback_host("[::1]") is True
    assert is_loopback_host("example.com") is False
    assert is_loopback_host("8.8.8.8") is False
    assert is_loopback_host(None) is False


def test_ensure_offline_installs_only_when_asked():
    uninstall_guard()
    assert ensure_offline({}) is False
    assert guard_active() is False
    try:
        assert ensure_offline({"FAYD_OFFLINE": "1"}) is True
        assert guard_active() is True
    finally:
        uninstall_guard()
    assert guard_active() is False


def test_install_is_idempotent_and_uninstall_restores():
    uninstall_guard()
    assert install_guard() is True
    try:
        assert install_guard() is False  # already installed
        assert uninstall_guard() is True
        assert uninstall_guard() is False  # already restored
        # the originals are back: a getaddrinfo call for a literal IP does no I/O
        assert socket.getaddrinfo("127.0.0.1", 0)
    finally:
        uninstall_guard()


def test_describe_names_the_state():
    uninstall_guard()
    assert "not enforced" in describe()
    install_guard()
    try:
        assert "enforced" in describe()
    finally:
        uninstall_guard()


# ------------------------------------------------------------- the blocking

def test_guard_blocks_a_remote_socket(offline_guard):
    with pytest.raises(NetworkDisabled):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(("8.8.8.8", 53))
        finally:
            sock.close()


def test_guard_allows_a_loopback_socket(offline_guard):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(server.getsockname())  # loopback is local, allowed
        client.close()
    finally:
        server.close()


def test_guard_blocks_create_connection(offline_guard):
    with pytest.raises(NetworkDisabled):
        socket.create_connection(("example.com", 80))


def test_guard_blocks_getaddrinfo(offline_guard):
    with pytest.raises(NetworkDisabled):
        socket.getaddrinfo("example.com", 443)


def test_guard_allows_getaddrinfo_for_loopback(offline_guard):
    assert socket.getaddrinfo("127.0.0.1", 0)


def test_guard_blocks_urlopen(offline_guard):
    with pytest.raises(NetworkDisabled):
        urllib.request.urlopen("https://example.com")


def test_guard_allows_urlopen_for_loopback(offline_guard):
    # No server is listening, but the *guard* must let it through: the failure is
    # a connection refusal, never a NetworkDisabled.
    with pytest.raises(Exception) as info:
        urllib.request.urlopen("http://127.0.0.1:1/", timeout=0.5)
    assert not isinstance(info.value, NetworkDisabled)


def test_guard_blocks_requests(offline_guard):
    """``requests`` must go through the patched socket, so it is blocked too."""
    requests = pytest.importorskip("requests")
    with pytest.raises(Exception) as info:
        requests.get("https://example.com", timeout=1)
    # The failure is the guard, not a real connection error.
    assert isinstance(info.value, NetworkDisabled) or isinstance(
        getattr(info.value, "__cause__", None), NetworkDisabled
    ) or "offline mode" in str(info.value)


# ---------------------------------------------------- the real local paths

def test_answer_path_runs_offline(offline_guard, tmp_path):
    from core.assistant import build_default_assistant

    assistant = build_default_assistant()
    answer = assistant.ask("what is porosity?")
    assert answer.text.strip()
    assert answer.citations


def test_memory_write_runs_offline(offline_guard, tmp_path):
    from core.long_term_memory import LongTermMemory

    memory = LongTermMemory("audit_offline", str(tmp_path / "memory.db"))
    record = memory.capture("porosity measures the void space in a rock")
    assert memory.count() >= 1
    assert memory.get(record.record_id) is not None


def test_training_smoke_step_runs_offline(offline_guard):
    import numpy as np
    from ml.llm.transformer import TransformerConfig, TransformerLM

    model = TransformerLM(TransformerConfig.preset("small"), seed=0)
    logits = model.forward(np.zeros((1, 4), dtype=int))
    assert logits.shape[0] == 1


def test_personality_exchange_runs_offline(offline_guard, tmp_path):
    from ai.characters import wrap_character
    from core.character_learning import CharacterLearning

    class _Echo:
        def ask(self, question: str) -> str:
            return f"answered: {question}"

    mouth = wrap_character("ultron", _Echo(), root=tmp_path)
    answer = mouth.ask("what is porosity?")
    assert answer

    learning = CharacterLearning("ultron", root=tmp_path)
    assert learning.exchanges()
