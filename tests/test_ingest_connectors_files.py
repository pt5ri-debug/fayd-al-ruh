"""Tests for the file connector: formats, streaming, and directory walking."""

from __future__ import annotations

import json

import pytest

from ingest.connectors.files import ALL_SUFFIXES, FileConnector
from ingest.errors import ConnectorError
from ingest.records import SourceSpec


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_suffixes_cover_the_required_formats():
    # Stage 1 supports TXT, Markdown, CSV, and JSON; Stage 2A adds digital PDF;
    # Stage 2B adds Word (.docx/.docm). OCR remains out of scope.
    for suffix in (".txt", ".md", ".markdown", ".csv", ".json", ".jsonl", ".pdf", ".docx", ".docm"):
        assert suffix in ALL_SUFFIXES


def test_stream_reads_a_txt_file(tmp_path):
    _write(tmp_path / "note.txt", "Water makes the soil wet.\nLight lights the candle.\n")
    records = list(FileConnector().stream(SourceSpec(str(tmp_path / "note.txt"))))
    assert len(records) == 1
    assert "Water makes the soil wet." in records[0].text
    assert records[0].source == "file"


def test_stream_reads_markdown(tmp_path):
    _write(tmp_path / "readme.md", "# Heading\n\nA seed needs soil and water.\n")
    records = list(FileConnector().stream(SourceSpec(str(tmp_path / "readme.md"))))
    assert records and "seed" in records[0].text


def test_csv_rows_become_key_value_lines(tmp_path):
    _write(tmp_path / "data.csv", "name,desc\nporosity,voids in rock\npermeability,flow capacity\n")
    records = list(FileConnector().stream(SourceSpec(str(tmp_path / "data.csv"))))
    text = records[0].text
    assert "name: porosity" in text
    assert "desc: flow capacity" in text


def test_csv_groups_rows_into_parts(tmp_path):
    rows = ["n,v"] + [f"{i},{i * 2}" for i in range(100)]
    _write(tmp_path / "big.csv", "\n".join(rows))
    spec = SourceSpec(str(tmp_path / "big.csv"), options={"row_group": 10})
    records = list(FileConnector().stream(spec))
    assert len(records) == 10  # 100 rows / 10 per group
    assert all(len(r.text.splitlines()) == 10 for r in records)


def test_jsonl_streams_one_record_per_row(tmp_path):
    lines = [json.dumps({"id": i, "label": f"item-{i}"}) for i in range(5)]
    _write(tmp_path / "items.jsonl", "\n".join(lines))
    records = list(FileConnector().stream(SourceSpec(str(tmp_path / "items.jsonl"))))
    assert len(records) == 1  # grouped into one record by default row_group
    assert "id: 0" in records[0].text


def test_json_list_becomes_records(tmp_path):
    _write(tmp_path / "list.json", json.dumps([{"a": 1}, {"a": 2}, {"a": 3}]))
    spec = SourceSpec(str(tmp_path / "list.json"), options={"row_group": 1})
    records = list(FileConnector().stream(spec))
    assert len(records) == 3
    assert "a: 1" in records[0].text


def test_invalid_json_reports_a_connector_error(tmp_path):
    _write(tmp_path / "bad.json", "{not json}")
    with pytest.raises(ConnectorError):
        list(FileConnector().stream(SourceSpec(str(tmp_path / "bad.json"))))


def test_directory_walk_is_sorted_and_skips_hidden(tmp_path):
    _write(tmp_path / "b.txt", "second")
    _write(tmp_path / "a.txt", "first")
    _write(tmp_path / ".hidden.txt", "secret")
    _write(tmp_path / "sub" / "c.txt", "third")
    records = list(FileConnector().stream(SourceSpec(str(tmp_path))))
    titles = [r.title for r in records]
    assert titles == ["a", "b", "c"]  # sorted, hidden skipped


def test_non_recursive_walk(tmp_path):
    _write(tmp_path / "a.txt", "top")
    _write(tmp_path / "sub" / "b.txt", "nested")
    spec = SourceSpec(str(tmp_path), options={"recursive": False})
    records = list(FileConnector().stream(spec))
    assert [r.title for r in records] == ["a"]


def test_plan_reports_fingerprints_without_reading(tmp_path):
    _write(tmp_path / "a.txt", "content")
    items = list(FileConnector().plan(SourceSpec(str(tmp_path))))
    assert len(items) == 1
    assert ":" in items[0].fingerprint  # size:mtime
    assert items[0].source_key


def test_missing_path_reports_a_connector_error(tmp_path):
    with pytest.raises(ConnectorError):
        list(FileConnector().stream(SourceSpec(str(tmp_path / "nope.txt"))))


def test_root_confinement_refuses_escape(tmp_path):
    outside = tmp_path / "outside.txt"
    _write(outside, "outside")
    root = tmp_path / "root"
    root.mkdir()
    spec = SourceSpec("../outside.txt", options={"root": str(root)})
    with pytest.raises(ConnectorError):
        list(FileConnector().stream(spec))


def test_exclude_roots_skips_a_subtree(tmp_path):
    _write(tmp_path / "keep" / "a.txt", "keep")
    _write(tmp_path / "kb" / "b.txt", "skip")
    spec = SourceSpec(str(tmp_path), options={"exclude_roots": [str(tmp_path / "kb")]})
    records = list(FileConnector().stream(spec))
    assert [r.title for r in records] == ["a"]


def test_docx_is_now_supported_by_the_file_connector(tmp_path):
    # Stage 2B adds Word. A malformed .docx is still refused (as a bad package),
    # not silently ingested; a valid one is read.
    import zipfile

    bad = tmp_path / "doc.docx"
    bad.write_bytes(b"PK\x03\x04 not really a docx")
    with pytest.raises(ConnectorError):
        list(FileConnector().stream(SourceSpec(str(bad))))

    good = tmp_path / "good.docx"
    xml = (
        '<?xml version="1.0"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>Reservoir porosity.</w:t></w:r></w:p></w:body></w:document>"
    )
    with zipfile.ZipFile(good, "w") as archive:
        archive.writestr("word/document.xml", xml)
    records = list(FileConnector().stream(SourceSpec(str(good))))
    assert records and records[0].metadata["kind"] == "docx"
