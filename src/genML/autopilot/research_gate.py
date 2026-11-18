"""The research step's fetch gate: which URLs a research session may read, and which sources count.

It runs as a Claude Code PreToolUse hook on every WebFetch of the research session::

    python research_gate.py --before 2026-11-01 [--extra www.example.org,reports.example.com]

reads the hook's JSON on stdin and exits 2 (blocked, the reason on stderr, which the session sees)
unless the URL is one of these:

* the OpenAlex API (``api.openalex.org``): a ``/works`` search whose filter ends before the cutoff
  (``to_publication_date:``), or one work whose publication date is before it,
* the Semantic Scholar API (``api.semanticscholar.org``): a paper search limited by
  ``publicationDateOrYear`` (or ``year``) to before the cutoff, or one paper dated before it,
* the arXiv API (``export.arxiv.org/api/query``): a search limited by ``submittedDate:[.. TO ..]`` to
  before the cutoff, or papers by id whose latest version is older than the cutoff,
* an arXiv paper page (``arxiv.org/abs|pdf|html/<id>``): version 1 submitted before the cutoff, or any
  version when the latest one is older than the cutoff,
* an old revision of an English Wikipedia article (``/w/index.php?oldid=N``) saved before the cutoff,
  or the revisions API asking for revisions up to a time before the cutoff,
* a site the owner listed for the problem (``--extra``): allowed, but not date-checked.

Everything else blocks, and so does any error: the gate fails closed. Claude Code lets some
documentation sites through without a permission rule, so this hook, not the allow rules, is what
holds the line. It uses only the standard library, so it cannot break on an import, and it takes no
path or problem name as an argument (the session sees the hook's command when it blocks).

The same checks verify the brief's citations afterwards (``check_citation``) and audit the session
log (``research.audit_session``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import date, datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, quote, unquote, urlparse

# Hosts that carry solutions, notebooks, discussions or other people's predictions. Never readable,
# not even when an owner lists one by mistake (``research.load_config`` refuses them too).
ANSWER_HOSTS = (
    "kaggle.com", "kaggleusercontent.com", "kaggle.net", "github.com", "githubusercontent.com",
    "github.io", "gitlab.com", "bitbucket.org", "huggingface.co", "medium.com", "towardsdatascience.com",
    "substack.com", "reddit.com", "youtube.com", "youtu.be", "x.com", "twitter.com", "linkedin.com",
    "dev.to", "hashnode.dev", "analyticsvidhya.com", "kdnuggets.com", "machinelearningmastery.com",
    "paperswithcode.com", "colab.research.google.com", "drive.google.com", "docs.google.com",
    "stackoverflow.com", "stackexchange.com", "zhihu.com", "csdn.net", "qiita.com", "zenn.dev",
)
SCHOLARLY_HOSTS = ("api.openalex.org", "api.semanticscholar.org", "export.arxiv.org", "arxiv.org",
                   "en.wikipedia.org")
USER_AGENT = "genml-research-gate/1.0"
ARXIV_ID = re.compile(r"^(?:(?P<old>[a-z][a-z\-]*(?:\.[A-Za-z]{2})?/(?P<old_num>\d{7}))|(?P<new>\d{4}\.\d{4,5}))"
                      r"(?:v(?P<version>\d+))?$")
CITATION_KINDS = ("openalex", "doi", "arxiv", "s2", "wikipedia", "url")
WIKIPEDIA_PAGE_KEYS = {"title", "oldid", "action", "printable", "useskin"}

Lookup = Callable[[str, str], Optional[date]]


def is_answer_host(host: str) -> bool:
    host = (host or "").lower().rstrip(".")
    return any(host == h or host.endswith("." + h) for h in ANSWER_HOSTS)


def _day_before(day: date) -> date:
    return date.fromordinal(day.toordinal() - 1)


def _parse_day(text: Optional[str]) -> Optional[date]:
    """The calendar day of an ISO date or timestamp (``2019-06-30`` or ``2019-06-30T12:00:00Z``)."""
    return date.fromisoformat(str(text)[:10]) if text else None


# --------------------------------------------------------------------------- #
# Dates of single records (network; tests pass their own lookup)
# --------------------------------------------------------------------------- #

def _get(url: str, timeout: float = 20.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:   # noqa: S310 (fixed hosts)
        return response.read()


def http_lookup(kind: str, ident: str) -> Optional[date]:
    """The date that decides whether a record counts, or None when it cannot be found.

    ``openalex``: a work id (W123) or ``doi:10...``; ``s2``: a Semantic Scholar paper id (also the
    ``DOI:``, ``ARXIV:`` and ``CorpusId:`` forms); ``arxiv_updated`` / ``arxiv_published``: the latest
    version's date / version 1's date; ``wikipedia_rev``: a revision's timestamp."""
    if kind == "openalex":
        data = json.loads(_get(f"https://api.openalex.org/works/{quote(ident, safe=':/')}?select=publication_date"))
        return _parse_day(data.get("publication_date"))
    if kind == "s2":
        data = json.loads(_get(f"https://api.semanticscholar.org/graph/v1/paper/{quote(ident, safe=':/')}"
                               "?fields=publicationDate,year"))
        if data.get("publicationDate"):
            return _parse_day(data["publicationDate"])
        return date(int(data["year"]), 12, 31) if data.get("year") else None      # a year alone: its last day
    if kind in ("arxiv_updated", "arxiv_published"):
        text = _get(f"https://export.arxiv.org/api/query?id_list={quote(ident, safe='/.')}&max_results=1").decode()
        tag = "updated" if kind == "arxiv_updated" else "published"
        parts = text.split("<entry>", 1)
        match = re.search(rf"<{tag}>([^<]+)</{tag}>", parts[1]) if len(parts) == 2 else None
        return _parse_day(match.group(1)) if match else None
    if kind == "wikipedia_rev":
        data = json.loads(_get("https://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=timestamp"
                               f"&format=json&revids={int(ident)}"))
        for page in (data.get("query") or {}).get("pages", {}).values():
            for rev in page.get("revisions") or []:
                return _parse_day(rev.get("timestamp"))
        return None
    raise ValueError(f"unknown lookup kind {kind}")


def _dated(kind: str, ident: str, before: date, lookup: Lookup, what: str) -> Tuple[bool, str]:
    try:
        when = lookup(kind, ident)
    except Exception as exc:  # network, rate limit, bad record: blocked; the session may try again later
        return False, (f"could not check the date of {what} ({type(exc).__name__}); try again later or use "
                       "another source")
    if when is None:
        return False, f"{what} has no date the gate can check"
    if when >= before:
        return False, f"{what} is dated {when}, not before {before}"
    return True, f"{what} dated {when}"


# --------------------------------------------------------------------------- #
# Read-time checks, one per source
# --------------------------------------------------------------------------- #

def _values(query: Dict[str, List[str]], key: str) -> List[str]:
    return query.get(key) or []


def _one(query: Dict[str, List[str]], key: str) -> str:
    values = _values(query, key)
    return values[0] if values else ""


def _openalex(path: str, query: Dict[str, List[str]], before: date, lookup: Lookup) -> Tuple[bool, str]:
    path = path.rstrip("/")
    if path == "/works":
        filters = ",".join(_values(query, "filter"))
        values = re.findall(r"to_publication_date:([^,]*)", filters)
        if not values:
            return False, (f"an OpenAlex search must be limited to works published before {before}: add "
                           f"filter=to_publication_date:{_day_before(before)}")
        for value in values:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                return False, "to_publication_date must be one plain date (no '|', '!' or ranges)"
            if date.fromisoformat(value) >= before:
                return False, f"to_publication_date must be before {before}"
        return True, "OpenAlex search limited by date"
    if path.startswith("/works/"):
        ident = unquote(path[len("/works/"):])
        if ident.lower().startswith("https://doi.org/"):
            ident = "doi:" + ident[len("https://doi.org/"):]
        if re.fullmatch(r"W\d+", ident) or (ident.lower().startswith("doi:10.") and " " not in ident):
            return _dated("openalex", ident, before, lookup, f"OpenAlex work {ident}")
    return False, "on OpenAlex only /works searches (with the date filter) and single works (W..., doi:...) may be read"


def _end_of(text: str) -> Optional[date]:
    """The last day a date limit covers (``2019-06-30``, ``2019-06`` or ``2019``); None if open."""
    text = text.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return date.fromisoformat(text)
    if re.fullmatch(r"\d{4}-\d{2}", text):
        year, month = map(int, text.split("-"))
        return _day_before(date(year + (month == 12), month % 12 + 1, 1))
    if re.fullmatch(r"\d{4}", text):
        return date(int(text), 12, 31)
    return None


def _semantic_scholar(path: str, query: Dict[str, List[str]], before: date, lookup: Lookup) -> Tuple[bool, str]:
    path = path.rstrip("/")
    if path in ("/graph/v1/paper/search", "/graph/v1/paper/search/bulk"):
        end = None
        if _one(query, "publicationDateOrYear"):
            limit = _one(query, "publicationDateOrYear")
            end = _end_of(limit.rsplit(":", 1)[-1] if ":" in limit else limit)
        elif _one(query, "year"):
            limit = _one(query, "year")
            end = _end_of(limit.rsplit("-", 1)[-1] if "-" in limit else limit)
        if end is None:
            return False, (f"a Semantic Scholar search must be limited to papers before {before}: add "
                           f"publicationDateOrYear=:{_day_before(before)}")
        if end >= before:
            return False, f"the search's date limit must end before {before}"
        return True, "Semantic Scholar search limited by date"
    match = re.fullmatch(r"/graph/v1/paper/([^/]+)", path)
    if match and match.group(1) not in ("search", "batch", "autocomplete"):
        ident = unquote(match.group(1))
        return _dated("s2", ident, before, lookup, f"Semantic Scholar paper {ident}")
    return False, ("on Semantic Scholar only paper searches (with the date limit) and single papers may be read "
                   "(no citations, references or recommendations: those reach newer work)")


def _arxiv_month(match: "re.Match[str]") -> Tuple[int, int]:
    number = match.group("new") or match.group("old_num")
    yy, month = int(number[:2]), int(number[2:4])
    return (1900 + yy if yy >= 91 else 2000 + yy), month


def _arxiv_ok(ident: str, before: date, lookup: Lookup) -> Tuple[bool, str]:
    match = ARXIV_ID.match(ident.strip())
    if not match:
        return False, f"'{ident}' is not an arXiv id"
    base = match.group("new") or match.group("old")
    month = _arxiv_month(match)
    if month > (before.year, before.month):
        return False, f"arXiv {base} was submitted after {before}"
    if match.group("version") == "1":
        if month < (before.year, before.month):
            return True, f"arXiv {base} version 1, submitted {month[0]}-{month[1]:02d}"
        return _dated("arxiv_published", base, before, lookup, f"arXiv {base} version 1")
    return _dated("arxiv_updated", base, before, lookup,
                  f"the latest version of arXiv {base} (pin version 1 as {base}v1 if a later one is too new)")


def _arxiv_api(path: str, query: Dict[str, List[str]], before: date, lookup: Lookup) -> Tuple[bool, str]:
    if path.rstrip("/") != "/api/query":
        return False, "on export.arxiv.org only /api/query may be read"
    ids = [i.strip() for i in _one(query, "id_list").split(",") if i.strip()]
    search = _one(query, "search_query")
    if search:
        ends = re.findall(r"submittedDate:\[\s*\d{8,14}\s+TO\s+(\d{8,14})\s*\]", search)
        if not ends or " OR " in f" {search.upper()} ":
            return False, (f"an arXiv search must be limited to papers submitted before {before} (and use no OR): add "
                           f"AND submittedDate:[199101010000 TO {_day_before(before):%Y%m%d}2359] to search_query")
        if min(datetime.strptime(e[:8], "%Y%m%d").date() for e in ends) >= before:
            return False, f"the search's submittedDate range must end before {before}"
        if not ids:
            return True, "arXiv search limited by date"
    if not ids:
        return False, "an arXiv query needs search_query (with submittedDate) or id_list"
    if len(ids) > 20:
        return False, "at most 20 arXiv ids per query"
    for ident in ids:
        match = ARXIV_ID.match(ident)
        if not match:
            return False, f"'{ident}' is not an arXiv id"
        ok, reason = _dated("arxiv_updated", match.group("new") or match.group("old"), before, lookup,
                            f"the latest version of arXiv {ident}")
        if not ok:
            return False, reason
    return True, "arXiv papers by id, all older than the cutoff"


def _arxiv_page(path: str, before: date, lookup: Lookup) -> Tuple[bool, str]:
    match = re.fullmatch(r"/(?:abs|pdf|html)/(.+?)(?:\.pdf)?/?", path)
    if not match:
        return False, ("on arxiv.org only paper pages (/abs/<id>, /pdf/<id>, /html/<id>) may be read; search through "
                       "export.arxiv.org/api/query with a submittedDate limit")
    return _arxiv_ok(match.group(1), before, lookup)


def _wikipedia(path: str, query: Dict[str, List[str]], before: date, lookup: Lookup) -> Tuple[bool, str]:
    if path == "/w/index.php" and _one(query, "oldid").isdigit():
        if set(query) - WIKIPEDIA_PAGE_KEYS or _one(query, "action") not in ("", "raw", "view", "render"):
            return False, "read one old revision: only title, oldid and action=raw|render (no diff, direction or curid)"
        return _dated("wikipedia_rev", _one(query, "oldid"), before, lookup,
                      f"Wikipedia revision {_one(query, 'oldid')}")
    if path == "/w/api.php":
        if (_one(query, "action") != "query" or _one(query, "prop") != "revisions" or query.get("list")
                or query.get("generator") or query.get("revids")):
            return False, ("on the Wikipedia API only action=query&prop=revisions with titles and rvstart (revisions "
                           "saved before the cutoff) may be read")
        start = _one(query, "rvstart")
        if not start or _one(query, "rvdir") not in ("", "older"):
            return False, f"ask for revisions up to a time before {before}: add rvstart={_day_before(before)}T23:59:59Z"
        try:
            when = _parse_day(start)
        except ValueError:
            return False, f"rvstart '{start}' is not a timestamp"
        if when is None or when >= before:
            return False, f"rvstart must be before {before}"
        return True, "Wikipedia revisions saved before the cutoff"
    return False, ("read Wikipedia only as an old revision: find its id with /w/api.php?action=query&prop=revisions&"
                   f"titles=<title>&rvlimit=1&rvprop=ids|timestamp&rvstart={_day_before(before)}T23:59:59Z"
                   "&format=json, "
                   "then read /w/index.php?title=<title>&oldid=<id>")


def check_url(url: str, before: date, extra_hosts: Sequence[str] = (),
              lookup: Optional[Lookup] = None) -> Tuple[bool, str]:
    """(allowed, reason) for one WebFetch of a research session."""
    lookup = lookup or http_lookup
    try:
        parsed = urlparse(str(url).strip())
        host = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme not in ("https", "http") or not host:
            return False, "only http(s) URLs"
        if parsed.username or parsed.password or parsed.port not in (None, 80, 443):
            return False, "no credentials or unusual ports in research URLs"
        if is_answer_host(host):
            return False, f"{host} hosts solutions, notebooks or discussions and is never a research source"
        if host in {h.lower() for h in extra_hosts}:
            return True, f"{host} is a source the owner listed for this problem (not date-checked)"
        query = parse_qs(parsed.query, keep_blank_values=True)
        if host == "api.openalex.org":
            return _openalex(parsed.path, query, before, lookup)
        if host == "api.semanticscholar.org":
            return _semantic_scholar(parsed.path, query, before, lookup)
        if host == "export.arxiv.org":
            return _arxiv_api(parsed.path, query, before, lookup)
        if host == "arxiv.org":
            return _arxiv_page(parsed.path, before, lookup)
        if host == "en.wikipedia.org":
            return _wikipedia(parsed.path, query, before, lookup)
        return False, (f"{host} is not a research source here (allowed: {', '.join(SCHOLARLY_HOSTS)}"
                       + (f", {', '.join(extra_hosts)}" if extra_hosts else "") + "); there is no general web search")
    except Exception as exc:  # anything unexpected blocks
        return False, f"the gate could not judge this URL ({type(exc).__name__}: {exc})"


# --------------------------------------------------------------------------- #
# Citations in the brief
# --------------------------------------------------------------------------- #

def check_citation(source_id: str, before: date, extra_hosts: Sequence[str] = (),
                   lookup: Optional[Lookup] = None) -> Tuple[bool, str]:
    """(counts, reason) for one cited source, given as ``kind:value``: ``openalex:W123``,
    ``doi:10.1145/...``, ``arxiv:1905.01234`` (or ``...v1``), ``s2:<paper id>``, ``wikipedia:<oldid>``
    or ``url:https://...`` (an owner-listed site only)."""
    lookup = lookup or http_lookup
    kind, _, value = str(source_id).partition(":")
    value = value.strip()
    if kind not in CITATION_KINDS or not value:
        return False, (f"'{source_id}' is not a citation the gate can check "
                       f"(use {', '.join(k + ':' for k in CITATION_KINDS)})")
    try:
        if kind == "openalex":
            if not re.fullmatch(r"W\d+", value):
                return False, f"'{value}' is not an OpenAlex work id (W...)"
            return _dated("openalex", value, before, lookup, f"OpenAlex work {value}")
        if kind == "doi":
            return _dated("openalex", f"doi:{value}", before, lookup, f"DOI {value}")
        if kind == "arxiv":
            return _arxiv_ok(value, before, lookup)
        if kind == "s2":
            return _dated("s2", value, before, lookup, f"Semantic Scholar paper {value}")
        if kind == "wikipedia":
            if not value.isdigit():
                return False, "cite Wikipedia by the revision id you read (wikipedia:<oldid>)"
            return _dated("wikipedia_rev", value, before, lookup, f"Wikipedia revision {value}")
        host = (urlparse(value).hostname or "").lower()
        if is_answer_host(host) or host not in {h.lower() for h in extra_hosts}:
            return False, f"url: citations are only for sites the owner listed ({host or value} is not one)"
        return True, f"{host} is a source the owner listed (date not checked)"
    except Exception as exc:
        return False, f"the citation could not be checked ({type(exc).__name__}: {exc})"


# --------------------------------------------------------------------------- #
# Hook entry point
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        parser = argparse.ArgumentParser(description="genML research fetch gate (a Claude Code PreToolUse hook)")
        parser.add_argument("--before", required=True, help="only sources older than this date (YYYY-MM-DD)")
        parser.add_argument("--extra", default="", help="owner-listed hosts, comma separated")
        args = parser.parse_args(argv)
        before = date.fromisoformat(args.before)
        extra = [h.strip().lower() for h in args.extra.split(",") if h.strip()]
        event = json.loads(sys.stdin.read() or "{}")
        tool = event.get("tool_name")
        url = (event.get("tool_input") or {}).get("url")
        if tool not in (None, "WebFetch") or not url:
            print(f"genml research gate: only WebFetch with a url is allowed here (got {tool})", file=sys.stderr)
            return 2
        ok, reason = check_url(url, before, extra)
        if ok:
            return 0
        print(f"genml research gate: blocked {url}: {reason}", file=sys.stderr)
        return 2
    except BaseException as exc:  # noqa: BLE001 (fail closed on anything, bad arguments included)
        print(f"genml research gate: blocked (gate error: {type(exc).__name__}: {exc})", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
