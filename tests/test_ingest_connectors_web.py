"""Tests for the governed web connector: allow-list, extraction, and no leaks."""

from __future__ import annotations

import pytest

from ingest.connectors.web import WebConnector, _html_to_text
from ingest.errors import ConnectorError
from ingest.records import SourceSpec


class _FakeResponse:
    def __init__(self, body: bytes, charset: str = "utf-8"):
        self._body = body
        self.headers = _Headers(charset)

    def read(self, size):
        return self._body[:size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Headers:
    def __init__(self, charset):
        self._charset = charset

    def get_content_charset(self):
        return self._charset


def _opener(body: bytes):
    calls = {}

    def open_url(request, timeout):
        calls["url"] = request.full_url
        calls["headers"] = dict(request.header_items())
        calls["timeout"] = timeout
        return _FakeResponse(body)

    return open_url, calls


def test_html_to_text_drops_tags_and_scripts():
    html = "<html><head><style>x{}</style></head><body><h1>Title</h1><script>bad()</script><p>Real text.</p></body></html>"
    text = _html_to_text(html)
    assert "Title" in text
    assert "Real text." in text
    assert "bad()" not in text
    assert "x{}" not in text


def test_plan_refuses_a_host_not_on_the_allow_list():
    connector = WebConnector()
    with pytest.raises(ConnectorError):
        list(connector.plan(SourceSpec("https://example.com/page")))


def test_plan_allows_an_explicit_host():
    connector = WebConnector()
    spec = SourceSpec("https://example.com/page", options={"allowed_hosts": ["example.com"]})
    items = list(connector.plan(spec))
    assert items[0].uri == "https://example.com/page"


def test_plan_allows_authorized_opt_in():
    connector = WebConnector()
    items = list(connector.plan(SourceSpec("https://example.com/page", options={"authorized": True})))
    assert items and items[0].metadata["host"] == "example.com"


def test_stream_item_fetches_and_extracts_text():
    opener, calls = _opener(b"<html><body><p>Oil flows through pores.</p></body></html>")
    connector = WebConnector(opener=opener)
    spec = SourceSpec("https://example.com/a", options={"authorized": True})
    records = list(connector.stream(spec))
    assert "Oil flows through pores." in records[0].text
    assert records[0].source == "web"
    assert records[0].url == "https://example.com/a"
    assert records[0].metadata["credibility"] > 0
    assert calls["url"] == "https://example.com/a"


def test_token_is_sent_but_never_kept_on_the_record():
    opener, calls = _opener(b"<html><body>text</body></html>")
    connector = WebConnector(opener=opener)
    spec = SourceSpec(
        "https://example.com/a",
        options={"authorized": True, "token": "super-secret"},
    )
    records = list(connector.stream(spec))
    assert calls["headers"].get("Authorization") == "Bearer super-secret"
    # The secret must not appear anywhere on the resulting record.
    assert "super-secret" not in records[0].text
    assert all("super-secret" not in str(value) for value in records[0].metadata.values())


def test_http_error_becomes_a_connector_error():
    from urllib.error import HTTPError

    def open_url(request, timeout):
        raise HTTPError(request.full_url, 404, "Not Found", {}, None)

    connector = WebConnector(opener=open_url)
    spec = SourceSpec("https://example.com/a", options={"authorized": True})
    with pytest.raises(ConnectorError) as error:
        list(connector.stream(spec))
    assert "HTTP 404" in str(error.value)


def test_timeout_must_be_positive():
    with pytest.raises(ValueError):
        WebConnector(timeout=0)


def test_scheme_resolution_includes_http_and_https():
    assert {"http", "https"} <= WebConnector.info.schemes
    assert WebConnector.info.remote is True
