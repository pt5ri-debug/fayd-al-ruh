"""The acceptance policy for newly learned knowledge.

Every statement that the project learns from the internet -- or from any other
external source -- is judged by this one module. It exists so the criteria are
*one thing in one place*: the same thresholds that the Continuous Internet
Learner applied when it reviewed the Petroleum Disciplines branch are the
thresholds that govern every future statement. There is no second, softer path
for data that arrives later.

The policy is deliberately conservative and fully local: it looks only at the
result and at the project's own seeded knowledge, never at the network and never
at already-learned text (a learned statement must not be able to vouch for the
next one).

A result is accepted only when it clears every gate, in order:

1. **Trusted source** -- the host must be on the explicit trusted list or a
   government/academic host (:func:`is_trusted`).
2. **Credible enough** -- the source credibility must meet ``MIN_CREDIBILITY``.
3. **Non-empty** -- there must be a snippet to store.
4. **Not a formula fragment** -- symbolic markup (``\\displaystyle`` etc.) is not
   a statement and is rejected.
5. **About the topic** -- the article's own title must genuinely be about the
   topic; a shared head-noun ("reservoir" -> a water reservoir) is not enough
   (:func:`is_relevant_to_topic`).
6. **Anchored to seeded knowledge** -- the statement must share terms with the
   project's seeded corpus, checked by ``KnowledgeUpdater`` against the project's
   knowledge only (:func:`core.knowledge_update`).

The numeric thresholds are named constants so a reviewer can see and change them
in one place.
"""

from __future__ import annotations

import html
import re
from urllib.parse import urlparse

# --- thresholds -----------------------------------------------------------

MIN_CREDIBILITY = 0.6
"""Credibility floor a search result must reach to be considered."""

TITLE_COVERAGE = 0.5
"""Fraction of the topic's words, and of the title's words, that must be shared
before a result is judged to be about the topic. Applied in both directions, so
an article that is mostly about something else is rejected even when it shares a
word with the topic."""

# --- trust ----------------------------------------------------------------

TRUSTED_HOSTS = (
    "spe.org",
    "onepetro.org",
    "api.org",
    "iso.org",
    "astm.org",
    "wikipedia.org",
)
"""Hosts treated as authoritative for petroleum engineering and general facts."""

TRUSTED_SUFFIXES = (".gov", ".edu", ".mil", ".ac.uk", ".gov.uk")
"""Top-level suffixes that mark a source as government or academic."""

# --- text cleaning --------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+")
_FORMULA_MARKERS_RE = re.compile(r"displaystyle|\\[a-zA-Z]+|\\\(")
_TITLE_RE = re.compile(r"[a-z0-9]+")

REJECT_UNTRUSTED = "domain not trusted"
REJECT_LOW_CREDIBILITY = "credibility below floor"
REJECT_EMPTY = "empty snippet"
REJECT_FORMULA = "formula fragment"


def title_words(text: str) -> set[str]:
    """The lowercase word set of a title, for topic/title comparison."""
    return set(_TITLE_RE.findall(text.casefold()))


def clean_statement(text: str) -> str:
    """Normalise a snippet into a stored statement.

    Search results arrive with HTML entities and citation URLs mixed into the
    prose. Left in, a URL's host leaks words like ``www`` into the store and
    creates spurious matches with unrelated text. This unescapes entities, drops
    tags and URLs, and collapses whitespace, so only the statement survives.
    """
    unescaped = html.unescape(text)
    without_tags = _TAG_RE.sub(" ", unescaped)
    without_urls = _URL_RE.sub(" ", without_tags)
    return " ".join(without_urls.split())


def is_formula_fragment(text: str) -> bool:
    """Whether text is symbolic markup rather than a quotable statement."""
    return bool(_FORMULA_MARKERS_RE.search(text))


def is_relevant_to_topic(topic: str, title: str) -> bool:
    """Whether a result's article is genuinely about the topic.

    A search for an ambiguous term returns results that merely share a word
    ("reservoir" -> a drinking-water reservoir, "screen" -> "Blue screen of
    death"). Those statements would pass the corpus gate yet teach nothing
    petroleum-specific, so the article's own title must be *about* the topic:
    the topic's words and the title's words must overlap substantially in both
    directions. A shared head-noun is not enough -- the article must not be
    mostly about something else.
    """
    topic_set = title_words(topic)
    title_set = title_words(title)
    if not topic_set or not title_set:
        return False
    shared = topic_set & title_set
    return (
        len(shared) / len(topic_set) >= TITLE_COVERAGE
        and len(shared) / len(title_set) >= TITLE_COVERAGE
    )


def source_rejection_reason(
    *,
    url: str,
    snippet: str,
    credibility: float,
    trusted_only: bool = True,
    min_credibility: float = MIN_CREDIBILITY,
) -> str | None:
    """Why a source's *content* is refused, independent of any topic.

    These gates apply to every learning path -- batch topic searches and
    single-question web answers alike -- because they judge the source itself,
    not its fit to a query. The topic-specific gate lives in
    :func:`rejection_reason`.
    """
    if trusted_only and not is_trusted(url):
        return REJECT_UNTRUSTED
    if credibility < min_credibility:
        return f"credibility {credibility:.2f} below {min_credibility:.2f}"
    if not snippet.strip():
        return REJECT_EMPTY
    if is_formula_fragment(snippet):
        return REJECT_FORMULA
    return None


def rejection_reason(
    *,
    url: str,
    title: str,
    topic: str,
    snippet: str,
    credibility: float,
    trusted_only: bool = True,
    min_credibility: float = MIN_CREDIBILITY,
) -> str | None:
    """Why a topic-search result is refused, or ``None`` when it passes.

    Returns a short machine-readable reason so callers can log or count refusals
    without re-deriving the decision. The source-level gates run first, then the
    topic-relevance gate.
    """
    source_reason = source_rejection_reason(
        url=url,
        snippet=snippet,
        credibility=credibility,
        trusted_only=trusted_only,
        min_credibility=min_credibility,
    )
    if source_reason is not None:
        return source_reason
    if not is_relevant_to_topic(topic, title):
        return f"not about '{topic}' (article is '{title}')"
    return None


def is_trusted(url: str) -> bool:
    """Whether a URL's host is on the trusted list or a government/academic host."""
    host = urlparse(url).netloc.casefold()
    if not host:
        return False
    if host.endswith(TRUSTED_SUFFIXES):
        return True
    return any(host == trusted or host.endswith(f".{trusted}") for trusted in TRUSTED_HOSTS)


__all__ = [
    "MIN_CREDIBILITY",
    "REJECT_EMPTY",
    "REJECT_FORMULA",
    "REJECT_LOW_CREDIBILITY",
    "REJECT_UNTRUSTED",
    "TITLE_COVERAGE",
    "TRUSTED_HOSTS",
    "TRUSTED_SUFFIXES",
    "clean_statement",
    "is_formula_fragment",
    "is_relevant_to_topic",
    "is_trusted",
    "rejection_reason",
    "source_rejection_reason",
    "title_words",
]
