"""The training model/adapter registry, the promotion manager, and rollback.

The Model Manager already owns the project's **base** models
(:mod:`core.model_manager`, :mod:`core.model_registry`). This module does **not**
compete with it: it records only the Phase-6 additions -- the **adapters** a run
produced, the **evaluation** each carries, and its **promotion status** -- and it
registers a promoted adapter as an ordinary ``ModelRecord`` in the existing
registry so the Model Manager can list and load it like any other model.

Three objects:

* :class:`AdapterRecord` -- one adapter's provenance, compatibility metadata, and
  promotion status, stored through the project's ``JsonStore``.
* :class:`TrainingRegistry` -- the durable log of adapters (and the base models
  they adapt), with the active-adapter pointer.
* :class:`PromotionManager` -- evaluates a candidate and, only when the policy
  allows it, promotes it; and :class:`RollbackManager` returns to a previous
  accepted adapter or to the base, **without deleting training history**.

Rollback never deletes a run or an adapter: it changes *which* adapter is active,
and the history stays readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.paths import training_registry_dir
from storage.json_store import JsonStore

from ml.training.evaluation import EvaluationReport

NAMESPACE = "training_registry"
KEY_ADAPTERS = "adapters"
KEY_ACTIVE = "active"

STATUS_CANDIDATE = "candidate"
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"
STATUS_SUPERSEDED = "superseded"

PROMOTION_STATES: tuple[str, ...] = (
    STATUS_CANDIDATE,
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    STATUS_SUPERSEDED,
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class AdapterRecord:
    """One trained adapter: what it is, what it came from, and whether it is promoted."""

    adapter_id: str
    run_id: str
    base_model_id: str
    base_model_version: str = ""
    dataset_id: str = ""
    dataset_version: int = 0
    method: str = ""
    owner: str = ""
    path: str = ""
    created_at: str = ""
    #: The evaluation that decided this adapter's fate, if one was run.
    evaluation: dict[str, Any] = field(default_factory=dict)
    status: str = STATUS_CANDIDATE
    compatibility: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.status == STATUS_ACCEPTED

    @property
    def available(self) -> bool:
        """True when the adapter directory is present right now."""
        return bool(self.path) and Path(self.path).is_dir()

    def describe(self) -> str:
        where = self.path or "(no directory)"
        return (
            f"{self.adapter_id} [{self.status}] {self.method} on {self.base_model_id} · "
            f"{self.dataset_id}@v{self.dataset_version} · {where}"
        )

    def to_primitive(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "base_model_id": self.base_model_id,
            "base_model_version": self.base_model_version,
            "compatibility": dict(self.compatibility),
            "created_at": self.created_at,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "evaluation": self.evaluation,
            "method": self.method,
            "notes": list(self.notes),
            "owner": self.owner,
            "path": self.path,
            "run_id": self.run_id,
            "status": self.status,
        }

    @classmethod
    def from_primitive(cls, data: object) -> AdapterRecord:
        if not isinstance(data, dict):
            raise ValueError("An adapter record must be a mapping.")
        adapter_id = str(data.get("adapter_id", "")).strip()
        if not adapter_id:
            raise ValueError("An adapter record needs an adapter_id.")
        return cls(
            adapter_id=adapter_id,
            run_id=str(data.get("run_id", "")),
            base_model_id=str(data.get("base_model_id", "")),
            base_model_version=str(data.get("base_model_version", "")),
            dataset_id=str(data.get("dataset_id", "")),
            dataset_version=int(data.get("dataset_version", 0) or 0),
            method=str(data.get("method", "")),
            owner=str(data.get("owner", "")),
            path=str(data.get("path", "")),
            created_at=str(data.get("created_at", "")),
            evaluation=dict(data.get("evaluation", {}) or {}),
            status=str(data.get("status", STATUS_CANDIDATE)),
            compatibility=dict(data.get("compatibility", {}) or {}),
            notes=tuple(str(n) for n in (data.get("notes") or ())),
        )


class TrainingRegistry:
    """The durable adapter log and the active-adapter pointer."""

    def __init__(self, root: Path | str | None = None, store: JsonStore | None = None) -> None:
        base = Path(root) if root is not None else training_registry_dir()
        self.store = store or JsonStore(base)

    # -- reading -----------------------------------------------------------
    def adapters(self) -> tuple[AdapterRecord, ...]:
        data = self.store.load(NAMESPACE, KEY_ADAPTERS)
        if not isinstance(data, list):
            return ()
        out: list[AdapterRecord] = []
        for item in data:
            try:
                out.append(AdapterRecord.from_primitive(item))
            except ValueError:
                continue
        return tuple(out)

    def get(self, adapter_id: str) -> AdapterRecord | None:
        return next((a for a in self.adapters() if a.adapter_id == adapter_id), None)

    def accepted(self) -> tuple[AdapterRecord, ...]:
        return tuple(a for a in self.adapters() if a.accepted)

    def active_id(self) -> str:
        data = self.store.load(NAMESPACE, KEY_ACTIVE)
        if isinstance(data, dict):
            return str(data.get("adapter_id", ""))
        return ""

    def active(self) -> AdapterRecord | None:
        active = self.active_id()
        return self.get(active) if active else None

    # -- writing -----------------------------------------------------------
    def add(self, record: AdapterRecord) -> AdapterRecord:
        existing = [a.to_primitive() for a in self.adapters()]
        if any(item.get("adapter_id") == record.adapter_id for item in existing):
            raise ValueError(f"The adapter {record.adapter_id} is already registered.")
        existing.append(record.to_primitive())
        self.store.save(NAMESPACE, KEY_ADAPTERS, existing)
        return record

    def update_status(self, adapter_id: str, status: str, *, evaluation: dict[str, Any] | None = None) -> AdapterRecord:
        """Change one adapter's status (and optionally its evaluation), keeping the log."""
        if status not in PROMOTION_STATES:
            raise ValueError(f"Unknown adapter status {status!r}.")
        records = list(self.adapters())
        target: AdapterRecord | None = None
        updated: list[dict[str, Any]] = []
        for record in records:
            if record.adapter_id == adapter_id:
                target = AdapterRecord(
                    **{
                        **record.to_primitive(),
                        "notes": list(record.notes),
                        "status": status,
                        "evaluation": evaluation if evaluation is not None else record.evaluation,
                    }
                )
                updated.append(target.to_primitive())
            else:
                updated.append(record.to_primitive())
        if target is None:
            raise KeyError(f"No adapter {adapter_id!r} is registered.")
        self.store.save(NAMESPACE, KEY_ADAPTERS, updated)
        return target

    def set_active(self, adapter_id: str) -> None:
        if adapter_id and self.get(adapter_id) is None:
            raise KeyError(f"No adapter {adapter_id!r} is registered.")
        self.store.save(NAMESPACE, KEY_ACTIVE, {"adapter_id": adapter_id})

    def clear_active(self) -> None:
        self.store.save(NAMESPACE, KEY_ACTIVE, {"adapter_id": ""})

    def report(self) -> str:
        adapters = self.adapters()
        if not adapters:
            return "No training adapters yet; the base models are in use."
        active = self.active_id()
        lines = [f"Active adapter: {active or '(base model)'}"]
        for record in adapters:
            mark = "*" if record.adapter_id == active else " "
            lines.append(f"{mark} {record.describe()}")
        return "\n".join(lines)


# ------------------------------------------------------------ promotion


@dataclass(frozen=True)
class PromotionDecision:
    """The promotion manager's verdict on one candidate."""

    adapter_id: str
    promoted: bool
    reason: str = ""
    requires_approval: bool = True
    approved: bool = False

    def describe(self) -> str:
        state = "promoted" if self.promoted else "not promoted"
        text = f"{self.adapter_id}: {state} -- {self.reason}"
        if self.requires_approval and not self.approved and not self.promoted:
            text += " (explicit approval is required)"
        return text


class PromotionManager:
    """Decide whether a candidate adapter may become the active one.

    Two rules, both structural:

    * an evaluation must exist and must have **accepted** the candidate;
    * an explicit **approval** is required -- a candidate is never promoted
      automatically, and the promotion policy's own threshold cannot be raised
      by the candidate itself.
    """

    def __init__(self, registry: TrainingRegistry, *, require_approval: bool = True) -> None:
        self.registry = registry
        self.require_approval = require_approval

    def evaluate_candidate(
        self,
        adapter_id: str,
        report: EvaluationReport,
        *,
        approved: bool = False,
    ) -> PromotionDecision:
        """Record a candidate's evaluation and promote it only if the policy allows."""
        record = self.registry.get(adapter_id)
        if record is None:
            raise KeyError(f"No adapter {adapter_id!r} is registered.")
        self.registry.update_status(
            adapter_id,
            STATUS_ACCEPTED if report.accepted else STATUS_REJECTED,
            evaluation=report.to_primitive(),
        )
        if not report.accepted:
            return PromotionDecision(
                adapter_id, False, f"the evaluation rejected it: {report.verdict}"
            )
        if self.require_approval and not approved:
            return PromotionDecision(
                adapter_id,
                False,
                "the evaluation accepted it, but promotion needs explicit approval",
                requires_approval=True,
                approved=False,
            )
        self._supersede_previous(adapter_id)
        self.registry.set_active(adapter_id)
        return PromotionDecision(adapter_id, True, "promoted to the active adapter", approved=True)

    def _supersede_previous(self, adapter_id: str) -> None:
        current = self.registry.active_id()
        if current and current != adapter_id:
            self.registry.update_status(current, STATUS_SUPERSEDED)


class RollbackManager:
    """Return model selection to a previous accepted adapter, or to the base.

    Rollback **never** deletes training history: it changes the active pointer and
    leaves every run, adapter, and evaluation in place, so the history stays
    readable and the rollback itself is auditable.
    """

    def __init__(self, registry: TrainingRegistry) -> None:
        self.registry = registry

    def rollback(self) -> AdapterRecord | None:
        """Deactivate the current adapter and return the previous accepted one."""
        current_id = self.registry.active_id()
        accepted = [a for a in self.registry.accepted() if a.adapter_id != current_id]
        if not accepted:
            self.registry.clear_active()
            return None
        previous = accepted[-1]
        self.registry.set_active(previous.adapter_id)
        return previous

    def to_base(self) -> None:
        """Deactivate any adapter, returning the base models to use."""
        self.registry.clear_active()

    def history(self) -> tuple[AdapterRecord, ...]:
        """Every adapter ever registered, newest last -- nothing is dropped."""
        return self.registry.adapters()


def register_adapter_as_model(record: AdapterRecord, *, root: Path | str | None = None) -> Any:
    """Register a promoted adapter as an ordinary ``ModelRecord`` in the Model Manager.

    This is the integration point: the Model Manager's registry already lists the
    base models, so a promoted adapter appears beside them rather than in a second
    catalog. The adapter's directory is the record's path, and its capabilities
    and tags record that it is an adapter for its base.
    """
    from core.model_registry import (
        FORMAT_UNKNOWN,
        SOURCE_TRAINED,
        ModelRecord,
        ModelRegistry,
        model_id_for,
    )

    identity = f"adapter:{record.adapter_id}"
    model_record = ModelRecord(
        model_id=model_id_for(identity),
        identity=identity,
        display_name=f"LoRA adapter {record.adapter_id}",
        model_format=FORMAT_UNKNOWN,
        runtime="fayd-inprocess",
        path=Path(record.path) if record.path else None,
        source=SOURCE_TRAINED,
        version=record.base_model_version,
        tags=("adapter", "lora", record.method),
        capabilities=("fine-tune-adapter",),
        notes=f"Adapter for {record.base_model_id}; dataset {record.dataset_id}@v{record.dataset_version}",
        extra={
            "base_model": record.base_model_id,
            "adapter_id": record.adapter_id,
            "dataset": f"{record.dataset_id}@v{record.dataset_version}",
            "status": record.status,
        },
        engine="training",
    )
    registry = ModelRegistry(root) if root is not None else ModelRegistry()
    registry.register_record(model_record)
    return model_record


__all__ = [
    "KEY_ACTIVE",
    "KEY_ADAPTERS",
    "NAMESPACE",
    "PROMOTION_STATES",
    "STATUS_ACCEPTED",
    "STATUS_CANDIDATE",
    "STATUS_REJECTED",
    "STATUS_SUPERSEDED",
    "AdapterRecord",
    "PromotionDecision",
    "PromotionManager",
    "RollbackManager",
    "TrainingRegistry",
    "register_adapter_as_model",
]
