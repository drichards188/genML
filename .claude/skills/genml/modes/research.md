---
# Permissions for the research session (src/genML/autopilot/policy.py: check_research_policy).
# WebFetch only, through the gate hook (src/genML/autopilot/research_gate.py); no shell, no web
# search, an empty working folder.
tools: [WebFetch, Read, Write]
network: gated
credentials: hidden
sandbox: required
workdir: empty
---
# Research a domain

A team that builds machine-learning models on tabular data needs to know what carries signal in this
domain:

{{DOMAIN}}

Find what published research and practice say, and write a brief of at most {{MAX_ITEMS}} items. Each item
is one signal (a property of a single record, of an entity's history, of its context or its counterparts)
that practitioners use or that studies found predictive, with why it matters and how practitioners
measure it. Spend about 40 minutes; a shorter brief of well-supported items beats a long one.

## Sources

Only sources published before {{SOURCES_BEFORE}} count. A gate checks every fetch and refuses anything
newer and any site not listed here. There is no web search: search these indexes through their APIs
with WebFetch.

- OpenAlex (works with abstracts):
  `https://api.openalex.org/works?search=<words>&filter=to_publication_date:{{LAST_DAY}}&per-page=10&select=id,display_name,publication_date,doi,cited_by_count,abstract_inverted_index`.
  One work: `https://api.openalex.org/works/W123`.
- Semantic Scholar:
  `https://api.semanticscholar.org/graph/v1/paper/search?query=<words>&publicationDateOrYear=:{{LAST_DAY}}&fields=title,abstract,year,publicationDate,externalIds&limit=10`.
  One paper: `https://api.semanticscholar.org/graph/v1/paper/<paperId>?fields=title,abstract,year`. It limits
  requests; when it refuses, use OpenAlex.
- arXiv:
  `https://export.arxiv.org/api/query?search_query=abs:<word>+AND+abs:<word>+AND+submittedDate:[199101010000+TO+{{LAST_DAY_COMPACT}}2359]&max_results=10`
  (no OR). A paper: `https://arxiv.org/abs/<id>v1` (version 1 is always safe to read).
- English Wikipedia, as a revision saved before the cutoff. Find its id with
  `https://en.wikipedia.org/w/api.php?action=query&prop=revisions&titles=<Title>&rvlimit=1&rvprop=ids|timestamp&rvstart={{LAST_DAY}}T23:59:59Z&format=json`,
  then read `https://en.wikipedia.org/w/index.php?title=<Title>&oldid=<id>`.
- Other sites the project's owner listed: {{EXTRA_HOSTS}}.

When the gate blocks a fetch it says why: fix the URL (usually the date limit) or move on. Do not try
other hosts or ways around it.

## What belongs in the brief

- Signals of the domain and the practice around them: what analysts, risk teams and researchers in the
  field measure and why. Prefer findings several sources support, and studies on real operational data.
- Not: anything about one particular public dataset, benchmark or contest (its quirks, its known best
  features, other people's models or scores); how a dataset was generated, ordered or formatted (digit or
  rounding patterns, ids, row order); leaderboard tricks. When a source studies a named benchmark dataset,
  take only its general domain findings.
- Say how a practitioner measures each signal in general terms (for example "a value relative to the same
  entity's usual values over a recent window"); you do not know the team's columns and do not need them.

## Result

Write {{RESULT_FILE}} as JSON:

```json
{
  "items": [
    {
      "id": "B01",
      "signal": "short name of the signal",
      "why": "what it indicates in the domain and what the sources found (with numbers when they give them)",
      "practice": "how practitioners measure or use it, in general terms",
      "sources": [{"id": "openalex:W2741809807", "title": "title of the work", "published": "2017-05-01"}]
    }
  ],
  "searches": ["each search you ran"],
  "notes": "what you could not find or check"
}
```

Source ids: `openalex:W...`, `doi:10....`, `arxiv:<id>` (or `<id>v1`), `s2:<paperId>`, `wikipedia:<revision id>`,
or `url:<address>` for a site the owner listed. Cite only sources you read in this session. Trusted code
checks the date of every cited source again and drops items with no source that counts.
