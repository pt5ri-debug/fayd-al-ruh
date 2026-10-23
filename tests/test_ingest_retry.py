"""Bounded per-item retries for a transient read failure.

A flaky local read (a file briefly locked, a network share hiccup) should not
fail a whole run. The pipeline retries the *read* a bounded number of times and
rolls back its counters so a retry never double-counts. These tests drive the
real pipeline through a connector whose first attempts fail.
"""

from __future__ import annotations

from collections.abc import Iterator

from core.knowledge_base import KnowledgeBaseStore
from ingest.dedup import DedupLedger
from ingest.pipeline import IngestionPipeline, ledger_path
from ingest.records import ConnectorInfo, IngestItem, SourceRecord, SourceSpec


class _FlakyConnector:
    """Yields one record, but fails on the first ``fail_times`` reads."""

    info = ConnectorInfo(name="flaky", description="test", suffixes=frozenset({".txt"}))

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.attempts = 0

    def plan(self, spec: SourceSpec) -> Iterator[IngestItem]:
        yield IngestItem(source_key="one", uri="mem://one", title="one")

    def stream_item(self, item: IngestItem) -> Iterator[SourceRecord]:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise OSError("transient read failure")
        yield SourceRecord(title="one", text="Recovered content.", source="file", source_key="one")


def _pipeline(root, connector, *, retries):
    store = KnowledgeBaseStore(root / "kb")
    ledger = DedupLedger(ledger_path(store.root))
    pipeline = IngestionPipeline(store, ledger, connectors={"flaky": connector}, retries=retries)
    return store, pipeline


def test_transient_failure_is_retried_then_succeeds(tmp_path):
    connector = _FlakyConnector(fail_times=2)
    store, pipeline = _pipeline(tmp_path, connector, retries=3)

    report = pipeline.ingest(SourceSpec("mem://one"), connector=connector)

    assert connector.attempts == 3  # two failures, then success
    assert report.failed == 0
    assert report.chunks == 1
    assert pipeline.errors.ok
    assert store.count() == 1


def test_retries_are_bounded_and_the_failure_is_reported(tmp_path):
    connector = _FlakyConnector(fail_times=99)
    store, pipeline = _pipeline(tmp_path, connector, retries=2)

    report = pipeline.ingest(SourceSpec("mem://one"), connector=connector)

    assert connector.attempts == 3  # 1 initial + 2 retries
    assert report.failed == 1
    assert len(pipeline.errors) == 1
    assert pipeline.errors.errors[0].kind == "OSError"
    assert store.count() == 0


def test_zero_retries_fails_on_first_attempt(tmp_path):
    connector = _FlakyConnector(fail_times=1)
    _store, pipeline = _pipeline(tmp_path, connector, retries=0)

    report = pipeline.ingest(SourceSpec("mem://one"), connector=connector)

    assert connector.attempts == 1
    assert report.failed == 1


def test_retry_does_not_double_count_partial_output(tmp_path):
    """A connector that writes one record then fails must not be counted twice."""

    class _HalfThenSucceed:
        info = ConnectorInfo(name="half", description="test", suffixes=frozenset({".txt"}))

        def __init__(self) -> None:
            self.attempts = 0

        def plan(self, spec: SourceSpec) -> Iterator[IngestItem]:
            yield IngestItem(source_key="two", uri="mem://two", title="two")

        def stream_item(self, item: IngestItem) -> Iterator[SourceRecord]:
            self.attempts += 1
            yield SourceRecord(title="two", text="First part.", source="file", source_key="two")
            if self.attempts == 1:
                raise OSError("failed after the first record")
            yield SourceRecord(title="two", text="Second part.", source="file", source_key="two")

    connector = _HalfThenSucceed()
    store, pipeline = _pipeline(tmp_path, connector, retries=1)

    report = pipeline.ingest(SourceSpec("mem://two"), connector=connector)

    assert report.failed == 0
    # The failed attempt's counters were rolled back. On the retry, "First part."
    # is already in the ledger (dedup) and "Second part." is new, so only one
    # chunk is newly counted, while the store holds both.
    assert report.records == 2
    assert report.chunks == 1
    assert store.count() == 2


def test_negative_retries_are_rejected(tmp_path):
    import pytest

    store = KnowledgeBaseStore(tmp_path / "kb")
    ledger = DedupLedger(ledger_path(store.root))
    with pytest.raises(ValueError):
        IngestionPipeline(store, ledger, retries=-1)


def test_a_failed_item_is_reported_and_the_job_still_advances(tmp_path):
    """Recovery contract: one bad item never blocks the rest, and the job moves on.

    The failure is reported for the caller to act on; the position advances so a
    resumed run continues with the remaining items instead of looping on the bad
    one. The bad item is skipped on resume (fix the file and re-run without the
    job, or without ``--job``, to retry it).
    """
    from ingest.jobs import JobState, JobStore
    from ingest.pipeline import ingest_jobs_dir

    connector = _FlakyConnector(fail_times=99)
    store = KnowledgeBaseStore(tmp_path / "kb")
    ledger = DedupLedger(ledger_path(store.root))
    jobs = JobStore(ingest_jobs_dir(store.root))
    job = JobState(job_id="run", source="mem://one")
    pipeline = IngestionPipeline(store, ledger, connectors={"flaky": connector}, retries=1, job=job, jobs=jobs)

    report = pipeline.ingest(SourceSpec("mem://one"), connector=connector)

    assert report.failed == 1
    saved = jobs.load("run")
    assert saved is not None and saved.position == 1

