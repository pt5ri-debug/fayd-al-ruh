"""The one place that enforces fully-local operation (``FAYD_OFFLINE=1``).

Every network egress in the project is already gated and off by default (web
search, model downloads, the bulk build, the update check, the Cloud mouth), but
nothing *enforced* that -- a future change could start reaching the network and
no test would catch it. This module closes that: when ``FAYD_OFFLINE`` is set to
a truthy value, :func:`install_guard` blocks every connection to an **external**
host, so an unintended egress fails loudly instead of silently succeeding.

**Loopback stays open.** A local model server -- Ollama, the audio.cpp resident
server -- is part of *local* operation, so ``localhost``/``127.0.0.0/8``/``::1``
are permitted; only a connection that would leave the machine is refused. This
is the honest definition of "offline": no traffic to another machine.

Design points, matching the rest of the project:

* **Dependency-free and import-safe.** Only the standard library is imported at
  module load, and nothing here reaches the network itself.
* **One choke point.** Every external egress ends in a socket ``connect`` --
  ``requests``/``urllib3``, ``urllib.request``, and ``http.client`` all go
  through :class:`socket.socket` -- so a guarded socket subclass plus the
  ``getaddrinfo``/``create_connection``/``urlopen`` helpers cover them all.
* **Reversible.** :func:`uninstall_guard` restores the originals, so a test can
  prove both the block and the restore without leaving the process patched.
* **Honest.** :func:`describe` names whether the guard is in force, and the
  guard is installed only when the environment asks for it -- the base install is
  untouched.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import urllib.parse
import urllib.request

#: The environment variable that turns the hard-off switch on.
OFFLINE_ENV = "FAYD_OFFLINE"

#: Values that count as "yes" (case-insensitive) for :data:`OFFLINE_ENV`.
OFFLINE_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Host names that always mean the local machine.
LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost"})


class NetworkDisabled(RuntimeError):
    """Raised when an external network attempt is made under the offline guard."""


def offline_requested(env: dict[str, str] | None = None) -> bool:
    """True when the environment asks for offline operation."""
    source = os.environ if env is None else env
    return source.get(OFFLINE_ENV, "").strip().lower() in OFFLINE_TRUTHY


def is_loopback_host(host: object) -> bool:
    """True when ``host`` names this machine (and so is not an external egress)."""
    if host is None:
        return False
    if isinstance(host, bytes):
        host = host.decode("utf-8", "ignore")
    if not isinstance(host, str):
        return False
    name = host.strip().strip("[]").lower()
    if name in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def _is_loopback_address(address: object) -> bool:
    """True for a ``(host, port)`` pair whose host is loopback."""
    if isinstance(address, tuple) and address:
        return is_loopback_host(address[0])
    return is_loopback_host(address)


def _refuse(host: object) -> None:
    raise NetworkDisabled(
        f"offline mode ({OFFLINE_ENV}) forbids a connection to {host!r}; "
        "unset it to allow a governed network action"
    )


class _GuardedSocket(socket.socket):
    """A socket that refuses to connect anywhere but this machine."""

    def connect(self, address):  # type: ignore[override]
        if not _is_loopback_address(address):
            _refuse(address[0] if isinstance(address, tuple) and address else address)
        return super().connect(address)

    def connect_ex(self, address):  # type: ignore[override]
        if not _is_loopback_address(address):
            _refuse(address[0] if isinstance(address, tuple) and address else address)
        return super().connect_ex(address)

    def sendto(self, *args):  # type: ignore[override]
        address = args[-1] if args else None
        if not _is_loopback_address(address):
            _refuse(address)
        return super().sendto(*args)


def _guard_getaddrinfo(host, *args, **kwargs):
    if not is_loopback_host(host):
        _refuse(host)
    return _original_getaddrinfo(host, *args, **kwargs)


def _guard_create_connection(address, *args, **kwargs):
    if not _is_loopback_address(address):
        _refuse(address[0] if isinstance(address, tuple) and address else address)
    return _original_create_connection(address, *args, **kwargs)


def _guard_urlopen(url, *args, **kwargs):
    target = url.full_url if isinstance(url, urllib.request.Request) else url
    host = urllib.parse.urlsplit(str(target)).hostname
    if host is not None and not is_loopback_host(host):
        _refuse(host)
    return _original_urlopen(url, *args, **kwargs)


#: The originals, captured at import so the guard is exactly reversible.
_original_socket = socket.socket
_original_getaddrinfo = socket.getaddrinfo
_original_create_connection = socket.create_connection
_original_urlopen = urllib.request.urlopen

_installed = False


def install_guard() -> bool:
    """Replace the network entry points with guards. Idempotent.

    Returns ``True`` the first time it installs the guard and ``False`` if it was
    already installed, so a caller can tell whether it did the work.
    """
    global _installed
    if _installed:
        return False
    socket.socket = _GuardedSocket  # type: ignore[assignment,misc]
    socket.getaddrinfo = _guard_getaddrinfo  # type: ignore[assignment]
    socket.create_connection = _guard_create_connection  # type: ignore[assignment]
    urllib.request.urlopen = _guard_urlopen  # type: ignore[assignment]
    _installed = True
    return True


def uninstall_guard() -> bool:
    """Restore the original network entry points. Returns whether it restored."""
    global _installed
    if not _installed:
        return False
    socket.socket = _original_socket  # type: ignore[assignment,misc]
    socket.getaddrinfo = _original_getaddrinfo  # type: ignore[assignment]
    socket.create_connection = _original_create_connection  # type: ignore[assignment]
    urllib.request.urlopen = _original_urlopen  # type: ignore[assignment]
    _installed = False
    return True


def guard_active() -> bool:
    """True when the network entry points are currently guarded."""
    return _installed


def ensure_offline(env: dict[str, str] | None = None) -> bool:
    """Install the guard when the environment asks for it; report the result.

    A program entry point calls this once, first thing, so ``FAYD_OFFLINE=1``
    makes the whole run local. A run without the variable is untouched.
    """
    if offline_requested(env):
        install_guard()
    return guard_active()


def describe() -> str:
    """An honest one-line status, for a diagnostic or a status command."""
    requested = offline_requested()
    if guard_active():
        return f"offline: enforced ({OFFLINE_ENV} is set; external network access is blocked)."
    if requested:
        return f"offline: requested ({OFFLINE_ENV} is set; guard not yet installed)."
    return f"offline: not enforced (set {OFFLINE_ENV}=1 to block external network access)."


__all__ = [
    "LOOPBACK_NAMES",
    "OFFLINE_ENV",
    "OFFLINE_TRUTHY",
    "NetworkDisabled",
    "describe",
    "ensure_offline",
    "guard_active",
    "install_guard",
    "is_loopback_host",
    "offline_requested",
    "uninstall_guard",
]
