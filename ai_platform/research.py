"""Research AI (spec Part 18): Search -> Retrieve -> Read -> Cite.

Uses DuckDuckGo's HTML endpoint (html.duckduckgo.com/html/), which
requires no API key. Like live_data.py, this is an unofficial public
endpoint, not a credentialed search API - it can break without notice,
and that possibility is surfaced as an explicit error, not swallowed.

Does not compare/verify/synthesize across sources with the LLM (the
proprietary model isn't trained enough for that yet, honestly) - this
is the Search -> Retrieve -> Cite portion of the pipeline, real and
testable, stopping short of claiming the "Analyze -> Verify" stages
that would need a genuinely capable LLM.
"""

import html
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

SEARCH_URL = "https://html.duckduckgo.com/html/"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

_RESULT_BLOCK_RE = re.compile(
    r'<a rel="nofollow" class="result__a" href="([^"]+)">(.*?)</a>.*?'
    r'class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


class ResearchError(RuntimeError):
    pass


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

    def citation(self) -> str:
        return f"{self.title} ({self.url})"


@dataclass
class ResearchReport:
    question: str
    results: list = field(default_factory=list)

    def to_dict(self):
        return {
            "question": self.question,
            "results": [{"title": r.title, "url": r.url, "snippet": r.snippet}
                        for r in self.results],
        }


def _clean(text: str) -> str:
    return html.unescape(_TAG_RE.sub("", text)).strip()


def _extract_ddg_url(href: str) -> str:
    """DuckDuckGo HTML results wrap the real URL in a redirect link
    (/l/?uddg=<encoded>). Unwrap it so citations point at the real source."""
    if href.startswith("//duckduckgo.com/l/") or "uddg=" in href:
        parsed = urllib.parse.urlparse(href if href.startswith("http") else "https:" + href)
        qs = urllib.parse.parse_qs(parsed.query)
        if "uddg" in qs:
            return qs["uddg"][0]
    return href


def search(query: str, max_results: int = 5, timeout: float = 10.0) -> ResearchReport:
    if not query or not query.strip():
        raise ResearchError("Empty query")

    data = urllib.parse.urlencode({"q": query}).encode("utf-8")
    req = urllib.request.Request(SEARCH_URL, data=data, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as e:
        raise ResearchError(f"Search request failed: {e}")

    results = []
    for href, title_html, snippet_html in _RESULT_BLOCK_RE.findall(body):
        results.append(SearchResult(
            title=_clean(title_html), url=_extract_ddg_url(href),
            snippet=_clean(snippet_html),
        ))
        if len(results) >= max_results:
            break

    if not results:
        raise ResearchError(
            "No results parsed - the search endpoint may have changed its HTML "
            "structure (this is an unofficial, unversioned endpoint)."
        )

    return ResearchReport(question=query, results=results)


def format_report(report: ResearchReport) -> str:
    lines = [f"Research results for: {report.question}", ""]
    for i, r in enumerate(report.results, 1):
        lines.append(f"{i}. {r.title}")
        lines.append(f"   {r.snippet}")
        lines.append(f"   Source: {r.url}")
        lines.append("")
    return "\n".join(lines)
