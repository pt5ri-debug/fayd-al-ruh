"""Model sources: what a source may claim, and what a download may be.

Everything here runs against a fake transport and a fake hub, so no test reaches
the network, downloads a real model, or needs an account. The point is the
*safety* rules, so that is what is pinned: a resume is refused unless it is
provably safe, a partial is never promoted, a path cannot escape its folder, and
a digest is only worth what its provenance says.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from core.model_download import (
    META_SUFFIX,
    PART_SUFFIX,
    DownloadCancelled,
    DownloadDenied,
    DownloadError,
    DownloadProgress,
    DownloadRequest,
    InsufficientSpace,
    ModelDownloader,
    Response,
    check_space,
    clear_partial_state,
    hash_file,
    load_partial_state,
    meta_path,
    normalize_digest,
    parse_pinned_digests,
    partial_path,
    resumable_files,
    safe_destination,
    save_partial_state,
)
from core.model_manager import ModelManager
from core.model_registry import SOURCE_DOWNLOAD
from core.model_sources import (
    CLAIM_CLAIMED,
    CLAIM_UNKNOWN,
    CLAIM_VERIFIED,
    DIGEST_NONE,
    DIGEST_SOURCE,
    DIGEST_TRUSTED,
    DirectUrlSource,
    HuggingFaceSource,
    LocalFolderSource,
    ModelFile,
    ModelOffer,
    SourceError,
    UnknownSource,
    claim_rank,
    default_sources,
    describe_claim,
    describe_sources,
    source_for,
    source_keys,
    validate_download_url,
)

# ---------------------------------------------------------------- fake transport


class FakeTransport:
    """A deterministic HTTP server: bytes in memory, no socket.

    Models the behaviours the downloader must survive: ranges, an ignored range,
    a changed ETag, a short body, a mid-stream failure, and a 404.
    """

    def __init__(
        self,
        body: bytes = b"",
        *,
        accepts_ranges: bool = True,
        etag: str = '"v1"',
        last_modified: str = "Mon, 01 Jan 2024 00:00:00 GMT",
        ignore_range: bool = False,
        fail_after: int | None = None,
        status: int = 200,
        omit_length: bool = False,
    ) -> None:
        self.body = body
        self.accepts_ranges = accepts_ranges
        self.etag = etag
        self.last_modified = last_modified
        self.ignore_range = ignore_range
        self.fail_after = fail_after
        self.status = status
        self.omit_length = omit_length
        self.open_calls: list[dict[str, str]] = []

    def head(self, url: str, *, timeout: float) -> Response:
        if self.status >= 400:
            return Response(status=self.status, headers={}, url=url)
        headers = {
            "Content-Length": str(len(self.body)),
            "ETag": self.etag,
            "Last-Modified": self.last_modified,
        }
        if self.accepts_ranges:
            headers["Accept-Ranges"] = "bytes"
        if self.omit_length:
            headers.pop("Content-Length")
        return Response(status=200, headers=headers, url=url)

    def open(self, url: str, *, headers: dict[str, str], timeout: float) -> Response:
        self.open_calls.append(dict(headers))
        if self.status >= 400:
            return Response(status=self.status, headers={}, url=url)
        start = 0
        range_header = headers.get("Range", "")
        if range_header.startswith("bytes="):
            start = int(range_header[len("bytes="):].split("-", 1)[0])
        if self.ignore_range:
            start = 0
            response_status = 200
        else:
            response_status = 206 if range_header else 200
        payload = self.body[start:]
        if self.fail_after is not None:
            payload = payload[: self.fail_after]
        response_headers = {
            "Content-Length": str(len(payload)),
            "ETag": self.etag,
            "Last-Modified": self.last_modified,
        }
        if self.omit_length:
            response_headers.pop("Content-Length")
        if self.accepts_ranges:
            response_headers["Accept-Ranges"] = "bytes"
        if response_status == 206:
            response_headers["Content-Range"] = (
                f"bytes {start}-{len(self.body) - 1}/{len(self.body)}"
            )
        return Response(
            status=response_status,
            headers=response_headers,
            body=payload,
            url=url,
        )


def _downloader(transport: FakeTransport) -> ModelDownloader:
    return ModelDownloader(transport=transport, approved=True)


def _file(name: str, body: bytes, **kwargs) -> ModelFile:
    return ModelFile(
        filename=name,
        size_bytes=kwargs.pop("size_bytes", len(body)),
        url=kwargs.pop("url", f"https://example.org/{name}"),
        **kwargs,
    )


# ------------------------------------------------------------------- sources


def test_sources_are_adapter_based_and_ordered():
    keys = source_keys()
    assert keys == ("local", "url", "huggingface")
    for source in default_sources():
        assert hasattr(source, "search") and hasattr(source, "resolve")
        assert hasattr(source, "available") and hasattr(source, "describe")


def test_unknown_source_is_refused_not_defaulted():
    with pytest.raises(UnknownSource):
        source_for("bit-torrent")


def test_local_folder_source_lists_only_present_files(tmp_path):
    present = tmp_path / "present.gguf"
    present.write_bytes(_gguf(b"hello"))
    source = LocalFolderSource(tmp_path)
    ok, why = source.available()
    assert ok and not why
    offers = source.search("")
    assert len(offers) == 1
    assert offers[0].claim == CLAIM_VERIFIED
    assert offers[0].files[0].filename == "present.gguf"


def test_local_folder_source_reports_absence_honestly(tmp_path):
    source = LocalFolderSource(tmp_path / "nope")
    ok, why = source.available()
    assert not ok and "exists" in why
    assert source.search("") == ()


def test_local_folder_source_with_no_root_says_so():
    ok, why = LocalFolderSource(None).available()
    assert not ok and "No local folder" in why


def test_url_source_refuses_non_https_and_credentials():
    source = DirectUrlSource()
    for bad in ("http://example.org/m.gguf", "file:///etc/passwd", "ftp://x/y"):
        with pytest.raises(SourceError):
            source.resolve(bad)
    with pytest.raises(SourceError):
        source.resolve("https://user:token@example.org/m.gguf")


def test_url_source_accepts_a_plain_https_url():
    offer = DirectUrlSource().resolve("https://example.org/model.gguf")
    assert offer.files[0].url.endswith("model.gguf")
    assert offer.claim == CLAIM_CLAIMED
    assert offer.files[0].digest_strength == DIGEST_NONE


def test_validate_download_url_is_a_pure_rule():
    assert validate_download_url("https://example.org/a.gguf") == ""
    assert "https" in validate_download_url("http://example.org/a.gguf")
    assert "credentials" in validate_download_url("https://t@example.org/a.gguf")
    assert validate_download_url("") != ""


def test_huggingface_source_without_the_client_is_actionable():
    source = HuggingFaceSource(client=None)
    ok, why = source.available()
    assert not ok
    assert "huggingface_hub" in why


def test_huggingface_source_uses_an_injected_client():
    class FakeHub:
        def list_models(self, search: str, limit: int):
            class Item:
                id = "owner/repo"

            return [Item()]

        def model_info(self, repo: str, revision=None):
            class Sibling:
                rfilename = "model.gguf"
                size = 42

            class Info:
                sha = "abc123"
                siblings = (Sibling(),)
                gated = False
                # A class attribute, assigned after the class so no mutable default
                # sits in the class body (which ruff flags for good reason).
                cardData: Any = None

            Info.cardData = {
                "license": "apache-2.0",
                "language": ["en"],
                "pipeline_tag": "text-generation",
            }
            return Info()

    source = HuggingFaceSource(client=FakeHub())
    assert source.available()[0]
    results = source.search("qwen")
    assert results and results[0].source_id == "owner/repo"
    offer = source.resolve("owner/repo")
    assert offer.license == "apache-2.0"
    assert offer.revision == "abc123"
    assert offer.languages == ("en",)
    assert offer.tasks == ("text-generation",)


def test_huggingface_source_requires_owner_repo():
    source = HuggingFaceSource(client=object())
    with pytest.raises(SourceError):
        source.resolve("justaname")


def test_claim_ranking_and_labels():
    from core.model_sources import CLAIM_TRUSTED

    assert claim_rank(CLAIM_UNKNOWN) < claim_rank(CLAIM_CLAIMED)
    assert claim_rank(CLAIM_CLAIMED) < claim_rank(CLAIM_VERIFIED)
    assert claim_rank(CLAIM_VERIFIED) < claim_rank(CLAIM_TRUSTED)
    assert claim_rank("nonsense") == 0
    assert describe_claim(CLAIM_CLAIMED) != ""
    assert describe_claim("nonsense") != ""


def test_describe_sources_lists_every_source():
    text = describe_sources()
    for key in source_keys():
        assert key in text


def test_offer_describe_and_primitive_are_honest():
    offer = ModelOffer(
        source_key="url",
        source_id="https://example.org/a.gguf",
        display_name="a.gguf",
        files=(ModelFile(filename="a.gguf", size_bytes=1000, url="https://example.org/a.gguf"),),
    )
    assert offer.total_bytes == 1000
    assert "unknown" in offer.describe()  # no license stated
    primitive = offer.to_primitive()
    assert primitive["total_bytes"] == 1000
    assert json.dumps(primitive)  # JSON-safe


def test_offer_total_is_unknown_when_a_file_size_is():
    offer = ModelOffer(
        source_key="url",
        source_id="x",
        files=(ModelFile(filename="a", size_bytes=None),),
    )
    assert offer.total_bytes is None
    assert "size unknown" in offer.describe()


# --------------------------------------------------------------- url safety


def test_safe_destination_refuses_traversal(tmp_path):
    for bad in ("../escape.gguf", "a/../../b.gguf", "/etc/passwd", "C:/windows/x"):
        with pytest.raises(DownloadError):
            safe_destination(tmp_path, bad)


def test_safe_destination_allows_a_plain_name(tmp_path):
    target = safe_destination(tmp_path, "model.gguf")
    assert target == (tmp_path / "model.gguf").resolve()


def test_safe_destination_allows_a_subfolder(tmp_path):
    target = safe_destination(tmp_path, "sub/model.gguf")
    assert target.parent.name == "sub"


# ------------------------------------------------------------------ digests


def test_hash_file_is_the_standard_sha256(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert hash_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_normalize_digest_reads_an_algo_prefix():
    assert normalize_digest("sha256:abc") == ("sha256", "abc")
    assert normalize_digest("abc") == ("sha256", "abc")
    assert normalize_digest("md5:abc") == ("md5", "abc")


def test_parse_pinned_digests_refuses_a_malformed_pair():
    assert parse_pinned_digests(["m.gguf=sha256:abc"]) == {"m.gguf": "sha256:abc"}
    with pytest.raises(DownloadError):
        parse_pinned_digests(["no-equals-sign"])


# ------------------------------------------------------------- disk space


def test_check_space_refuses_a_known_size_that_does_not_fit(tmp_path):
    with pytest.raises(InsufficientSpace):
        check_space(tmp_path, 10**18)


def test_check_space_allows_an_unknown_size(tmp_path):
    check_space(tmp_path, None)  # no error


def test_check_space_allows_a_small_file(tmp_path):
    check_space(tmp_path, 10)  # no error


# ----------------------------------------------------------- happy path


def test_download_promotes_and_hashes(tmp_path):
    body = b"model-bytes"
    downloader = _downloader(FakeTransport(body))
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path,
            files=(_file("model.gguf", body),),
            source_id="https://example.org/model.gguf",
        )
    )
    assert report.complete and report.status == "complete"
    target = tmp_path / "model.gguf"
    assert target.read_bytes() == body
    assert not (tmp_path / f"model.gguf{PART_SUFFIX}").exists()
    assert not (tmp_path / f"model.gguf{META_SUFFIX}").exists()
    assert report.files[0].digest == hashlib.sha256(body).hexdigest()
    assert report.files[0].digest_strength == DIGEST_NONE


def test_download_reports_measurable_progress(tmp_path):
    body = b"x" * 5000
    seen: list[DownloadProgress] = []
    downloader = _downloader(FakeTransport(body))
    downloader.download(
        DownloadRequest(
            destination=tmp_path, files=(_file("m.gguf", body),), source_id="u",
        ),
        on_progress=seen.append,
    )
    assert seen and seen[-1].measurable
    assert seen[-1].bytes_done == 5000
    assert seen[-1].fraction == 1.0


def test_download_reports_indeterminate_progress_when_total_unknown(tmp_path):
    body = b"y" * 100
    seen: list[DownloadProgress] = []
    downloader = _downloader(FakeTransport(body, omit_length=True))
    downloader.download(
        DownloadRequest(
            destination=tmp_path,
            files=(ModelFile(filename="m.gguf", url="https://example.org/m.gguf"),),
            source_id="u",
        ),
        on_progress=seen.append,
    )
    assert seen
    assert any(not item.measurable for item in seen)
    assert "unknown" in seen[0].describe()


def test_empty_file_downloads_and_is_registered(tmp_path):
    downloader = _downloader(FakeTransport(b""))
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path,
            files=(_file("empty.gguf", b"", size_bytes=0),),
            source_id="u",
        )
    )
    assert report.complete
    assert (tmp_path / "empty.gguf").exists()


def test_multiple_files_all_promoted(tmp_path):
    downloader = _downloader(FakeTransport(b"data"))
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path,
            files=(_file("a.gguf", b"data"), _file("b.gguf", b"data")),
            source_id="u",
        )
    )
    assert report.complete
    assert (tmp_path / "a.gguf").exists() and (tmp_path / "b.gguf").exists()


# ---------------------------------------------------------------- failures


def test_missing_file_is_a_failed_report_not_an_exception(tmp_path):
    downloader = _downloader(FakeTransport(b"", status=404))
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path, files=(_file("m.gguf", b"x"),), source_id="u",
        )
    )
    assert not report.complete
    assert report.status == "failed"
    assert not (tmp_path / "m.gguf").exists()


def test_size_mismatch_is_refused_and_not_promoted(tmp_path):
    body = b"short"
    downloader = _downloader(FakeTransport(body))
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path,
            files=(ModelFile(filename="m.gguf", size_bytes=9999, url="https://example.org/m.gguf"),),
            source_id="u",
        )
    )
    assert not report.complete
    assert not (tmp_path / "m.gguf").exists()


def test_network_failure_mid_stream_is_retried(tmp_path):
    body = b"z" * 100
    transport = FakeTransport(body, fail_after=10)
    downloader = ModelDownloader(transport=transport, approved=True)
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path,
            files=(_file("m.gguf", body),),
            source_id="u",
            max_attempts=2,
        )
    )
    # The fake always truncates, so this must end in a failed report, not a hang.
    assert not report.complete
    assert len(transport.open_calls) >= 2


def test_timeout_like_failure_is_reported_actionably(tmp_path):
    class Exploding(FakeTransport):
        def open(self, url, *, headers, timeout):
            raise DownloadError("The transfer failed: timed out")

    downloader = ModelDownloader(transport=Exploding(b"x"), approved=True)
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path, files=(_file("m.gguf", b"x"),), source_id="u",
            max_attempts=1,
        )
    )
    assert not report.complete
    assert "attempt" in report.detail or "timed out" in report.detail


def test_cancellation_keeps_a_resumable_partial(tmp_path):
    body = b"q" * 100000
    state = {"stop": False}

    def should_cancel() -> bool:
        return state["stop"]

    def on_progress(progress: DownloadProgress) -> None:
        if progress.bytes_done > 0:
            state["stop"] = True

    downloader = ModelDownloader(
        transport=FakeTransport(body), approved=True
    )
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path,
            files=(_file("m.gguf", body),),
            source_id="u",
            chunk_bytes=1000,
        ),
        should_cancel=should_cancel,
        on_progress=on_progress,
    )
    assert report.status == "cancelled"
    assert not report.complete
    assert not (tmp_path / "m.gguf").exists()
    assert report.resumable
    assert (tmp_path / f"m.gguf{PART_SUFFIX}").exists()


def test_cancelled_before_start_leaves_nothing(tmp_path):
    downloader = ModelDownloader(transport=FakeTransport(b"abc"), approved=True)
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path, files=(_file("m.gguf", b"abc"),), source_id="u",
        ),
        should_cancel=lambda: True,
    )
    assert report.status == "cancelled"
    assert not report.complete


# ----------------------------------------------------------------- resumption


def test_resume_continues_from_the_partial(tmp_path):
    body = b"abcdefghij" * 10
    first = FakeTransport(body)
    downloader = ModelDownloader(transport=first, approved=True)
    part = partial_path(tmp_path, "m.gguf")
    tmp_path.mkdir(exist_ok=True)
    part.write_bytes(body[:30])
    save_partial_state(
        tmp_path,
        _state(body, bytes_written=30, etag=first.etag),
    )
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path, files=(_file("m.gguf", body),), source_id="u",
        )
    )
    assert report.complete
    assert report.files[0].resumed
    assert (tmp_path / "m.gguf").read_bytes() == body


def test_resume_is_refused_when_the_etag_changed(tmp_path):
    body = b"abcdefghij" * 10
    transport = FakeTransport(body, etag='"v2"')
    downloader = ModelDownloader(transport=transport, approved=True)
    part = partial_path(tmp_path, "m.gguf")
    tmp_path.mkdir(exist_ok=True)
    part.write_bytes(body[:30])
    save_partial_state(tmp_path, _state(body, bytes_written=30, etag='"v1"'))
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path, files=(_file("m.gguf", body),), source_id="u",
        )
    )
    assert report.complete
    assert not report.files[0].resumed  # restarted, not appended
    assert (tmp_path / "m.gguf").read_bytes() == body


def test_resume_is_refused_when_the_server_ignores_ranges(tmp_path):
    body = b"abcdefghij" * 10
    transport = FakeTransport(body, ignore_range=True)
    downloader = ModelDownloader(transport=transport, approved=True)
    part = partial_path(tmp_path, "m.gguf")
    tmp_path.mkdir(exist_ok=True)
    part.write_bytes(body[:30])
    save_partial_state(tmp_path, _state(body, bytes_written=30, etag='"v1"'))
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path, files=(_file("m.gguf", body),), source_id="u",
        )
    )
    assert report.complete
    assert not report.files[0].resumed


def test_resume_is_refused_when_ranges_are_unsupported(tmp_path):
    body = b"abcdefghij" * 10
    transport = FakeTransport(body, accepts_ranges=False)
    downloader = ModelDownloader(transport=transport, approved=True)
    part = partial_path(tmp_path, "m.gguf")
    tmp_path.mkdir(exist_ok=True)
    part.write_bytes(body[:30])
    save_partial_state(tmp_path, _state(body, bytes_written=30, etag='"v1"'))
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path, files=(_file("m.gguf", body),), source_id="u",
        )
    )
    assert report.complete
    assert not report.files[0].resumed


def test_resume_is_refused_when_the_partial_is_longer_than_the_source(tmp_path):
    body = b"abcdefghij" * 10
    transport = FakeTransport(body)
    downloader = ModelDownloader(transport=transport, approved=True)
    part = partial_path(tmp_path, "m.gguf")
    tmp_path.mkdir(exist_ok=True)
    part.write_bytes(body + b"extra")
    save_partial_state(tmp_path, _state(body, bytes_written=len(body) + 5, etag='"v1"'))
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path, files=(_file("m.gguf", body),), source_id="u",
        )
    )
    assert report.complete
    assert not report.files[0].resumed
    assert (tmp_path / "m.gguf").read_bytes() == body


def test_partial_without_a_sidecar_is_not_trusted(tmp_path):
    body = b"abcdefghij" * 10
    downloader = ModelDownloader(transport=FakeTransport(body), approved=True)
    part = partial_path(tmp_path, "m.gguf")
    tmp_path.mkdir(exist_ok=True)
    part.write_bytes(body[:30])
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path, files=(_file("m.gguf", body),), source_id="u",
        )
    )
    assert report.complete
    assert not report.files[0].resumed


def test_restart_discards_the_partial(tmp_path):
    body = b"abcdefghij" * 10
    downloader = ModelDownloader(transport=FakeTransport(body), approved=True)
    part = partial_path(tmp_path, "m.gguf")
    tmp_path.mkdir(exist_ok=True)
    part.write_bytes(body[:30])
    save_partial_state(tmp_path, _state(body, bytes_written=30, etag='"v1"'))
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path, files=(_file("m.gguf", body),), source_id="u",
            restart=True,
        )
    )
    assert report.complete
    assert not report.files[0].resumed


def test_resumable_files_lists_partials(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / f"a.gguf{PART_SUFFIX}").write_bytes(b"x")
    (tmp_path / "b.gguf").write_bytes(b"y")
    found = resumable_files(tmp_path)
    assert len(found) == 1 and found[0].name == f"a.gguf{PART_SUFFIX}"


def test_partial_state_round_trips(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    save_partial_state(tmp_path, _state(b"x", bytes_written=5, etag='"e"'))
    loaded = load_partial_state(tmp_path, "m.gguf")
    assert loaded is not None and loaded.bytes_written == 5 and loaded.etag == '"e"'
    clear_partial_state(tmp_path, "m.gguf")
    assert load_partial_state(tmp_path, "m.gguf") is None


def test_a_corrupt_sidecar_reads_as_absent(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    meta_path(tmp_path, "m.gguf").write_text("{not json")
    assert load_partial_state(tmp_path, "m.gguf") is None


# ------------------------------------------------------------------ digests


def test_a_pinned_digest_that_matches_is_trusted(tmp_path):
    body = b"trusted-bytes"
    digest = hashlib.sha256(body).hexdigest()
    downloader = _downloader(FakeTransport(body))
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path,
            files=(_file("m.gguf", body),),
            source_id="u",
            pinned_digests={"m.gguf": f"sha256:{digest}"},
        )
    )
    assert report.complete
    assert report.files[0].digest_strength == DIGEST_TRUSTED
    assert report.files[0].verified


def test_a_pinned_digest_that_differs_blocks_the_promotion(tmp_path):
    body = b"real-bytes"
    downloader = _downloader(FakeTransport(body))
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path,
            files=(_file("m.gguf", body),),
            source_id="u",
            pinned_digests={"m.gguf": "sha256:" + "0" * 64},
        )
    )
    assert not report.complete
    assert not (tmp_path / "m.gguf").exists()


def test_a_source_published_digest_is_only_the_sources_word(tmp_path):
    body = b"source-bytes"
    digest = hashlib.sha256(body).hexdigest()
    downloader = _downloader(FakeTransport(body))
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path,
            files=(
                ModelFile(
                    filename="m.gguf",
                    size_bytes=len(body),
                    url="https://example.org/m.gguf",
                    expected_digest=digest,
                    digest_algorithm="sha256",
                    digest_strength=DIGEST_SOURCE,
                ),
            ),
            source_id="u",
        )
    )
    assert report.complete
    assert report.files[0].digest_strength == DIGEST_SOURCE
    assert report.files[0].verified


def test_a_mismatched_source_digest_blocks_the_promotion(tmp_path):
    body = b"source-bytes"
    downloader = _downloader(FakeTransport(body))
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path,
            files=(
                ModelFile(
                    filename="m.gguf",
                    size_bytes=len(body),
                    url="https://example.org/m.gguf",
                    expected_digest="0" * 64,
                    digest_algorithm="sha256",
                ),
            ),
            source_id="u",
        )
    )
    assert not report.complete
    assert not (tmp_path / "m.gguf").exists()


# -------------------------------------------------------------------- gates


def test_download_is_denied_without_approval(tmp_path):
    downloader = ModelDownloader(transport=FakeTransport(b"x"), approved=False)
    with pytest.raises(DownloadDenied):
        downloader.download(
            DownloadRequest(
                destination=tmp_path, files=(_file("m.gguf", b"x"),), source_id="u",
            )
        )
    assert not (tmp_path / "m.gguf").exists()


def test_a_download_with_no_url_is_refused(tmp_path):
    downloader = _downloader(FakeTransport(b"x"))
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path,
            files=(ModelFile(filename="m.gguf"),),
            source_id="u",
        )
    )
    assert not report.complete
    assert "No URL" in report.detail or "URL" in report.detail


# --------------------------------------------------------------- manager glue


def _manager(tmp_path) -> ModelManager:
    return ModelManager(root=tmp_path)


def test_manager_search_offers_uses_a_fake_hub(tmp_path):
    class FakeHub:
        def list_models(self, search: str, limit: int):
            class Item:
                id = "owner/repo"

            return [Item()]

        def model_info(self, repo, revision=None):
            raise AssertionError("not used")

    manager = _manager(tmp_path)
    offers = manager.search_offers("q", source="huggingface", hub_client=FakeHub())
    assert offers and offers[0].source_id == "owner/repo"


def test_manager_download_registers_a_completed_model(tmp_path):
    body = _gguf(b"weights")
    manager = _manager(tmp_path)
    offer = ModelOffer(
        source_key="url",
        source_id="https://example.org/model.gguf",
        display_name="model.gguf",
        model_format="gguf",
        files=(
            ModelFile(
                filename="model.gguf", size_bytes=len(body),
                url="https://example.org/model.gguf",
            ),
        ),
    )
    report, entry = manager.download(
        offer, destination=tmp_path / "dl", approved=True,
        transport=FakeTransport(body),
    )
    assert report.complete
    assert entry is not None
    assert entry.record.source == SOURCE_DOWNLOAD
    assert entry.record.version == ""
    assert entry.record.extra["download_source"] == "url"


def test_manager_does_not_register_an_incomplete_download(tmp_path):
    manager = _manager(tmp_path)
    offer = ModelOffer(
        source_key="url",
        source_id="https://example.org/model.gguf",
        display_name="model.gguf",
        model_format="gguf",
        files=(
            ModelFile(
                filename="model.gguf", size_bytes=100,
                url="https://example.org/model.gguf",
            ),
        ),
    )
    report, entry = manager.download(
        offer, destination=tmp_path / "dl", approved=True,
        transport=FakeTransport(b"x", status=500),
    )
    assert not report.complete
    assert entry is None
    assert manager.registry.count() == 0


def test_manager_download_without_approval_is_recorded(tmp_path):
    manager = _manager(tmp_path)
    offer = ModelOffer(
        source_key="url",
        source_id="https://example.org/model.gguf",
        files=(ModelFile(filename="m.gguf", url="https://example.org/m.gguf"),),
    )
    report, entry = manager.download(
        offer, destination=tmp_path / "dl", approved=False,
        transport=FakeTransport(b"x"),
    )
    assert not report.complete and entry is None
    assert any(not event.ok for event in manager.log_tail())


def test_manager_resumable_downloads(tmp_path):
    manager = _manager(tmp_path)
    target = tmp_path / "dl"
    target.mkdir()
    (target / f"m.gguf{PART_SUFFIX}").write_bytes(b"x")
    assert len(manager.resumable_downloads(destination=target)) == 1


def test_manager_sources_and_source_by_key(tmp_path):
    manager = _manager(tmp_path)
    assert {s.key for s in manager.sources()} == {"local", "url", "huggingface"}
    assert manager.source("url").key == "url"
    with pytest.raises(UnknownSource):
        manager.source("nope")


def test_download_report_primitive_is_json_safe(tmp_path):
    body = b"x"
    downloader = _downloader(FakeTransport(body))
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path, files=(_file("m.gguf", body),), source_id="u",
        )
    )
    assert json.dumps(report.to_primitive())
    assert "downloaded" in report.describe()


def test_report_never_claims_complete_on_a_partial(tmp_path):
    downloader = _downloader(FakeTransport(b"x", status=500))
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path, files=(_file("m.gguf", b"x"),), source_id="u",
        )
    )
    assert not report.ok
    assert "not a model" in report.describe()


def test_cancel_from_a_progress_callback_becomes_a_report(tmp_path):
    """A cancel is always a report at the ``download`` boundary, never a raise."""
    downloader = ModelDownloader(transport=FakeTransport(b"x" * 5000), approved=True)
    state = {"n": 0}

    def on_progress(progress: DownloadProgress) -> None:
        state["n"] += 1
        if state["n"] > 1:
            raise DownloadCancelled("stop")

    report = downloader.download(
        DownloadRequest(
            destination=tmp_path,
            files=(_file("m.gguf", b"x" * 5000),),
            source_id="u",
            chunk_bytes=100,
            max_attempts=1,
        ),
        on_progress=on_progress,
    )
    assert report.status == "cancelled"
    assert not report.complete
    assert not (tmp_path / "m.gguf").exists()


def test_ignored_range_restarts_cleanly(tmp_path):
    """A server that answers 200 to a range is restarted, not appended to."""
    body = b"abcdefghij" * 10
    transport = FakeTransport(body, ignore_range=True)
    downloader = ModelDownloader(transport=transport, approved=True)
    part = partial_path(tmp_path, "m.gguf")
    tmp_path.mkdir(exist_ok=True)
    part.write_bytes(body[:30])
    save_partial_state(tmp_path, _state(body, bytes_written=30, etag='"v1"'))
    report = downloader.download(
        DownloadRequest(
            destination=tmp_path, files=(_file("m.gguf", body),), source_id="u",
        )
    )
    assert report.complete
    assert not report.files[0].resumed
    assert (tmp_path / "m.gguf").read_bytes() == body


# ------------------------------------------------------------------ fixtures


def _gguf(payload: bytes) -> bytes:
    """A minimal but real GGUF header, so ``probe_file`` accepts the file."""
    import struct

    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
    return header + payload


def _state(body: bytes, *, bytes_written: int, etag: str):
    from core.model_download import PartialState

    return PartialState(
        url="https://example.org/m.gguf",
        filename="m.gguf",
        size_bytes=len(body),
        etag=etag,
        bytes_written=bytes_written,
    )
