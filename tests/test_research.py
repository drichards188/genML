"""The domain research step: the fetch gate, citation checks, blindness, the audit and a full run
with a fake Claude CLI (no network: every date lookup and the reviewer are faked)."""

import io
import json
import sys
import textwrap
from datetime import date
from pathlib import Path

import pytest

from src.genML.autopilot import research
from src.genML.autopilot import research_gate as gate
from src.genML.autopilot.agent import AgentSettings
from src.genML.autopilot.loop import Autopilot, AutopilotConfig, validate_explore, validate_plan

CUTOFF = date(2019, 7, 15)
DATES = {
    ("openalex", "W1"): date(2017, 5, 1), ("openalex", "W9"): date(2020, 1, 1),
    ("openalex", "doi:10.1/old"): date(2015, 1, 1),
    ("s2", "abc"): date(2018, 3, 3), ("s2", "new"): date(2021, 3, 3),
    ("arxiv_updated", "1905.01234"): date(2019, 5, 20), ("arxiv_updated", "1810.00001"): date(2020, 2, 1),
    ("arxiv_published", "1907.00002"): date(2019, 7, 2), ("arxiv_published", "1907.00003"): date(2019, 7, 20),
    ("wikipedia_rev", "123"): date(2019, 1, 1), ("wikipedia_rev", "999"): date(2023, 1, 1),
}


def lookup(kind, ident):
    if ident == "broken":
        raise TimeoutError("network")
    return DATES.get((kind, ident))


def allowed(url, extra=()):
    return gate.check_url(url, CUTOFF, extra, lookup)[0]


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("url", [
    "https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/111284",
    "https://kaggle.com/code/someone/notebook",
    "https://raw.githubusercontent.com/someone/repo/main/solution.py",
    "https://someone.github.io/post",
    "https://medium.com/@someone/how-i-won",
    "https://www.reddit.com/r/MachineLearning/",
    "https://docs.python.org/3/",                      # Claude Code pre-approves it; the gate does not
    "https://www.google.com/search?q=fraud",
    "ftp://arxiv.org/abs/1905.01234v1",
    "https://arxiv.org:8443/abs/1905.01234v1",
])
def test_gate_blocks_answer_hosts_and_anything_unlisted(url):
    assert not allowed(url)


def test_gate_openalex_needs_the_date_limit():
    base = "https://api.openalex.org/works?search=card%20fraud"
    assert not allowed(base)
    assert allowed(base + "&filter=to_publication_date:2019-07-14")
    assert allowed(base + "&filter=type:article,to_publication_date:2018-01-01&per-page=10")
    assert not allowed(base + "&filter=to_publication_date:2019-07-15")                 # the cutoff day itself
    assert not allowed(base + "&filter=to_publication_date:2018-01-01|2030-01-01")      # an OR inside the filter
    assert not allowed(base + "&filter=to_publication_date:2018-01-01,to_publication_date:2030-01-01")
    assert allowed("https://api.openalex.org/works/W1") and not allowed("https://api.openalex.org/works/W9")
    assert allowed("https://api.openalex.org/works/doi:10.1/old")
    assert not allowed("https://api.openalex.org/works/broken")                          # lookup failed: blocked
    assert not allowed("https://api.openalex.org/authors?search=x")
    assert not allowed("https://api.openalex.org/works/W1/ngrams")


def test_gate_semantic_scholar_and_arxiv():
    s2 = "https://api.semanticscholar.org/graph/v1/paper/search?query=fraud"
    assert not allowed(s2)
    assert allowed(s2 + "&publicationDateOrYear=:2019-06-30")
    assert allowed(s2 + "&publicationDateOrYear=2010:2018")
    assert not allowed(s2 + "&publicationDateOrYear=2015:2019")                         # runs to 2019-12-31
    assert not allowed(s2 + "&publicationDateOrYear=2015:")                             # open-ended
    assert allowed(s2 + "&year=2010-2018") and not allowed(s2 + "&year=2015-")
    assert allowed("https://api.semanticscholar.org/graph/v1/paper/abc?fields=title")
    assert not allowed("https://api.semanticscholar.org/graph/v1/paper/new")
    assert not allowed("https://api.semanticscholar.org/graph/v1/paper/abc/citations")   # newer work cites it
    api = "https://export.arxiv.org/api/query?search_query=abs:fraud"
    assert not allowed(api)
    assert allowed(api + "+AND+submittedDate:[199101010000+TO+201907142359]")
    assert not allowed(api + "+OR+abs:card+AND+submittedDate:[199101010000+TO+201907142359]")
    assert not allowed(api + "+AND+submittedDate:[199101010000+TO+202001012359]")
    assert allowed("https://export.arxiv.org/api/query?id_list=1905.01234")
    assert not allowed("https://export.arxiv.org/api/query?id_list=1810.00001")         # revised after the cutoff
    assert allowed("https://arxiv.org/abs/1905.01234v1")                               # month before: no lookup
    assert allowed("https://arxiv.org/pdf/1905.01234v1.pdf") and allowed("https://arxiv.org/abs/cs/0601001v1")
    assert not allowed("https://arxiv.org/abs/1908.00001v1")                           # submitted after
    assert allowed("https://arxiv.org/abs/1907.00002v1") and not allowed("https://arxiv.org/abs/1907.00003v1")
    assert allowed("https://arxiv.org/abs/1905.01234") and not allowed("https://arxiv.org/abs/1810.00001v2")
    assert not allowed("https://arxiv.org/list/cs.LG/recent")


def test_gate_wikipedia_only_old_revisions():
    assert not allowed("https://en.wikipedia.org/wiki/Credit_card_fraud")
    assert allowed("https://en.wikipedia.org/w/index.php?title=Credit_card_fraud&oldid=123")
    assert not allowed("https://en.wikipedia.org/w/index.php?title=Credit_card_fraud&oldid=999")
    assert not allowed("https://en.wikipedia.org/w/index.php?oldid=123&direction=next")   # the next revision
    assert not allowed("https://en.wikipedia.org/w/index.php?oldid=123&diff=prev")
    api = "https://en.wikipedia.org/w/api.php?action=query&prop=revisions&titles=Credit_card_fraud&rvprop=ids"
    assert allowed(api + "&rvstart=2019-07-14T23:59:59Z")
    assert not allowed(api) and not allowed(api + "&rvstart=2019-07-14T23:59:59Z&rvdir=newer")
    assert not allowed(api + "&rvstart=2020-01-01T00:00:00Z")
    assert not allowed("https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=fraud")


def test_gate_owner_hosts_are_allowed_but_never_answer_hosts():
    assert allowed("https://www.example.org/report.pdf", extra=["www.example.org"])
    assert not allowed("https://www.kaggle.com/x", extra=["www.kaggle.com"])


def test_citations_are_checked_by_date_and_kind():
    def counts(source, extra=()):
        return gate.check_citation(source, CUTOFF, extra, lookup)[0]

    assert counts("openalex:W1") and not counts("openalex:W9") and not counts("openalex:nope")
    assert counts("doi:10.1/old") and counts("s2:abc") and not counts("s2:new")
    assert counts("arxiv:1905.01234v1") and not counts("arxiv:1908.00001v1")
    assert counts("wikipedia:123") and not counts("wikipedia:999") and not counts("wikipedia:Credit_card_fraud")
    assert counts("url:https://www.example.org/r", extra=["www.example.org"]) and not counts("url:https://x.org/r")
    assert not counts("https://arxiv.org/abs/1905.01234v1") and not counts("openalex:broken")


def test_gate_main_reads_the_hook_event(monkeypatch, capsys):
    def run(event, *args):
        monkeypatch.setattr(sys, "stdin", io.StringIO(event))
        return gate.main(["--before", "2019-07-15", *args])

    fetch = lambda url: json.dumps({"tool_name": "WebFetch", "tool_input": {"url": url}})   # noqa: E731
    assert run(fetch("https://api.openalex.org/works?search=x&filter=to_publication_date:2019-01-01")) == 0
    assert run(fetch("https://www.kaggle.com/")) == 2 and "blocked" in capsys.readouterr().err
    assert run(fetch("https://www.example.org/a"), "--extra", "www.example.org") == 0
    assert run("{broken") == 2 and run(json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls"}})) == 2


# --------------------------------------------------------------------------- #
# Configuration and blindness
# --------------------------------------------------------------------------- #

DOMAIN = ("Fraud detection on payment card transactions: which properties of a transaction and of the "
          "card's recent activity indicate fraud.")


def _yaml(domain=DOMAIN, rest="sources_before: 2019-07-15\n"):
    return f"domain: {json.dumps(domain)}\n{rest}"


def _problems(tmp_path, research_yaml, name="toy-s1e1"):
    problems = tmp_path / "problems"
    (problems / name).mkdir(parents=True)
    (problems / name / "problem.yaml").write_text(
        f"name: {name}\nsource: {{type: kaggle, competition: playground-series-s1e1}}\ntarget: is_fraud\n")
    (problems / name / research.CONFIG_FILE).write_text(research_yaml)
    return problems


def test_config_is_validated(tmp_path):
    problems = _problems(tmp_path, _yaml())
    config = research.load_config("toy-s1e1", problems)
    assert config.sources_before == CUTOFF and config.hosts[0] == "api.openalex.org" and config.max_items == 20
    assert research.load_config("missing", problems) is None
    for bad in (_yaml("short"), _yaml(rest=""), _yaml(rest="sources_before: soon\n"),
                _yaml(rest="sources_before: 2019-07-15\nextra_hosts: [www.kaggle.com]\n"),
                _yaml(rest="sources_before: 2019-07-15\nextra_hosts: ['https://x.org/a']\n"),
                _yaml(rest="sources_before: 2019-07-15\nmax_items: 500\n"),
                _yaml(rest="sources_before: 2019-07-15\nsurprise: 1\n"), "domain: [\n"):
        (problems / "toy-s1e1" / research.CONFIG_FILE).write_text(bad)
        with pytest.raises(research.ResearchError):
            research.load_config("toy-s1e1", problems)


@pytest.mark.parametrize("domain, problem", [
    (DOMAIN + " As in toy-s1e1.", "toy-s1e1"),
    (DOMAIN + " From playground-series-s1e1.", "playground-series-s1e1"),
    (DOMAIN + " The Kaggle data.", "kaggle"),
    (DOMAIN + " Columns like amount_usd matter.", "amount_usd"),
    (DOMAIN + " The label is_fraud.", "is_fraud"),
    (DOMAIN + " See train.csv.", "data file"),
])
def test_a_domain_that_names_the_problem_is_refused(tmp_path, domain, problem):
    problems = _problems(tmp_path, _yaml())
    config = research.ResearchConfig(domain=domain, sources_before=CUTOFF)
    found = research.blindness_problems(config, "toy-s1e1", problems_dir=problems, columns=["id", "amount_usd"])
    assert any(problem in f for f in found), found
    clean = research.ResearchConfig(domain=DOMAIN, sources_before=CUTOFF)
    assert research.blindness_problems(clean, "toy-s1e1", problems_dir=problems, columns=["id", "amount_usd"]) == []


def test_the_fraud_problem_is_configured_blind():
    config = research.load_config("fraud-s6e11", Path(__file__).resolve().parents[1] / "problems")
    assert config is not None and config.sources_before == date(2026, 11, 1)
    assert research.blindness_problems(config, "fraud-s6e11",
                                       problems_dir=Path(__file__).resolve().parents[1] / "problems",
                                       columns=["amount_usd", "is_fraud", "txn_count_last_24h"]) == []


# --------------------------------------------------------------------------- #
# The result, the audit and the checks after the session
# --------------------------------------------------------------------------- #

def _item(n, sources):
    return {"id": f"B{n:02d}", "signal": f"signal {n}", "why": "why", "practice": "how",
            "sources": [{"id": s, "title": f"title {s}"} for s in sources]}


def test_brief_validation():
    good = {"items": [_item(1, ["openalex:W1"])], "searches": ["card fraud velocity"], "notes": ""}
    assert research.validate_brief(good) == []
    assert research.validate_brief({"items": []})
    assert research.validate_brief({"items": [_item(1, [])]})
    assert research.validate_brief({"items": [_item(1, ["W1"])]})                         # no kind:value
    assert research.validate_brief({"items": [_item(1, ["openalex:W1"]), _item(1, ["openalex:W1"])]})   # same id
    assert research.validate_brief({"items": [_item(n, ["openalex:W1"]) for n in range(1, 6)]}, max_items=3)


def _event(kind, part):
    return json.dumps({"type": kind, "message": {"content": [part]}})


def _log(path, calls):
    """A stream-json log with (tool, input, ok) calls."""
    lines = []
    for n, (tool, data, ok) in enumerate(calls):
        lines.append(_event("assistant", {"type": "tool_use", "id": f"t{n}", "name": tool, "input": data}))
        result = {"type": "tool_result", "tool_use_id": f"t{n}", "content": "x"}
        if not ok:
            result["is_error"] = True
        lines.append(_event("user", result))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_audit_proves_the_gate_held(tmp_path):
    folder = tmp_path / "work"
    folder.mkdir()
    config = research.ResearchConfig(domain=DOMAIN, sources_before=CUTOFF)
    ok_url = "https://api.openalex.org/works?search=x&filter=to_publication_date:2019-01-01"
    clean = _log(tmp_path / "clean.jsonl", [
        ("WebFetch", {"url": ok_url}, True),
        ("WebFetch", {"url": "https://www.kaggle.com/"}, False),
        ("Write", {"file_path": str(folder / "brief.json")}, True)])
    audit = research.audit_session(clean, folder, config, lookup)
    assert audit["problems"] == [] and audit["fetched"] == [ok_url] and audit["blocked"] == 1
    leaked = _log(tmp_path / "leaked.jsonl", [
        ("WebFetch", {"url": "https://github.com/x/solution"}, True),
        ("Bash", {"command": "ls"}, True),
        ("Read", {"file_path": "/etc/passwd"}, True)])
    problems = research.audit_session(leaked, folder, config, lookup)["problems"]
    assert len(problems) == 3 and "github.com" in problems[0] and "Bash" in problems[1] and "outside" in problems[2]


def test_items_without_a_counting_source_are_dropped():
    config = research.ResearchConfig(domain=DOMAIN, sources_before=CUTOFF)
    kept, dropped = research.verify_items([_item(1, ["openalex:W1", "openalex:W9"]), _item(2, ["openalex:W9"])],
                                          config, lookup)
    assert [i["id"] for i in kept] == ["B01"] and [s["counts"] for s in kept[0]["sources"]] == [True, False]
    assert dropped[0]["id"] == "B02" and "2020-01-01" in dropped[0]["reason"]


def test_retrying_lookup_caches_and_retries():
    calls = []

    def flaky(kind, ident):
        calls.append(ident)
        if len(calls) == 1:
            raise TimeoutError("once")
        return date(2018, 1, 1)

    wrapped = research.retrying(flaky, sleep=lambda s: None)
    assert wrapped("openalex", "W1") == date(2018, 1, 1) and wrapped("openalex", "W1") == date(2018, 1, 1)
    assert len(calls) == 2


# --------------------------------------------------------------------------- #
# A full run with a fake Claude CLI
# --------------------------------------------------------------------------- #

FAKE_RESEARCHER = textwrap.dedent('''\
    #!/usr/bin/env python3
    import json, os, re, sys
    from pathlib import Path
    argv = sys.argv[1:]
    prompt = sys.stdin.read()
    Path(os.environ["FAKE_SEEN"]).write_text(json.dumps({"argv": argv, "prompt": prompt, "cwd": os.getcwd()}))
    target = Path(re.search(r"(/\\S+/brief\\.json)", prompt).group(1))
    leak = os.environ.get("FAKE_LEAK") == "1"
    url = "https://github.com/x/solution" if leak else \\
        "https://api.openalex.org/works?search=fraud&filter=to_publication_date:2019-07-14"
    def event(kind, part):
        print(json.dumps({"type": kind, "message": {"content": [part]}}))
    event("assistant", {"type": "tool_use", "id": "a", "name": "WebFetch", "input": {"url": url}})
    event("user", {"type": "tool_result", "tool_use_id": "a", "content": "ok"})
    event("assistant", {"type": "tool_use", "id": "b", "name": "WebFetch", "input": {"url": "https://www.kaggle.com/"}})
    event("user", {"type": "tool_result", "tool_use_id": "b", "content": "blocked", "is_error": True})
    item = lambda n, src: {"id": f"B0{n}", "signal": f"signal {n}", "why": "w", "practice": "p",
                           "sources": [{"id": src, "title": "t"}]}
    target.write_text(json.dumps({"items": [item(1, "openalex:W1"), item(2, "openalex:W9"), item(3, "s2:abc")],
                                  "searches": ["card fraud velocity"], "notes": "n"}))
    session = argv[argv.index("--session-id") + 1] if "--session-id" in argv else argv[argv.index("--resume") + 1]
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "session_id": session,
                      "result": "done", "num_turns": 3}))
    ''')


class FakeAsk:
    def __init__(self):
        self.prompts = []

    def __call__(self, system, prompt, schema, **kwargs):
        self.prompts.append((system, prompt, kwargs))

        class Result:
            backend, model, log_file = "fake-codex", "m", "log"
            data = {"summary": "one item is a dataset quirk", "items": [
                {"id": "B01", "keep": True, "kind": "domain_signal", "reason": "fine"},
                {"id": "B03", "keep": False, "kind": "dataset_artifact", "reason": "a quirk"}]}
        return Result()


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    fake = tmp_path / "fake_claude"
    fake.write_text(FAKE_RESEARCHER)
    fake.chmod(0o755)
    monkeypatch.setenv("FAKE_SEEN", str(tmp_path / "seen.json"))
    return fake


def test_run_research_end_to_end(tmp_path, fake_cli):
    problems = _problems(tmp_path, _yaml())
    outputs = tmp_path / "outputs"
    ask = FakeAsk()
    brief = research.run_research("toy-s1e1", outputs_root=outputs, problems_dir=problems,
                                  agent=AgentSettings(claude_bin=str(fake_cli)), workspace=tmp_path / "ws",
                                  lookup=lookup, ask=ask, columns=["id", "amount_usd"])
    assert [i["id"] for i in brief["items"]] == ["B01"]
    assert {d["id"]: d["by"] for d in brief["dropped"]} == {"B02": "source check", "B03": "review"}
    assert brief["audit"]["fetched"] == 1 and brief["audit"]["blocked"] == 1 and brief["audit"]["problems"] == []
    assert research.load_brief("toy-s1e1", outputs)["config_fingerprint"] == brief["config_fingerprint"]
    seen = json.loads((tmp_path / "seen.json").read_text())
    assert "toy" not in seen["prompt"] and "toy" not in seen["cwd"] and DOMAIN in seen["prompt"]
    argv = seen["argv"]
    assert argv[argv.index("--tools") + 1] == "WebFetch,Read,Write" and "--restricted" in argv
    assert "--add-dir" not in argv
    assert "toy" not in argv[argv.index("--settings") + 1]
    session = Path(brief["session"])
    settings = json.loads((session / "research.settings.json").read_text())       # kept with the session record
    assert settings["hooks"]["PreToolUse"][0]["matcher"] == "WebFetch" and "Bash" in settings["permissions"]["deny"]
    assert "toy" not in ask.prompts[0][1] and ask.prompts[0][2]["purpose"] == "research-review"
    assert (session / "research.jsonl").exists() and (session / "brief_raw.json").exists()
    assert not any((tmp_path / "ws" / "research").iterdir())                 # the session folders are gone
    agents = research.brief_for_agents(brief)
    assert [i["id"] for i in agents["items"]] == ["B01"] and "measurement" in agents["about"]


def test_a_leaky_session_is_refused(tmp_path, fake_cli, monkeypatch):
    monkeypatch.setenv("FAKE_LEAK", "1")
    problems = _problems(tmp_path, _yaml())
    outputs = tmp_path / "outputs"
    with pytest.raises(research.ResearchError, match="audit"):
        research.run_research("toy-s1e1", outputs_root=outputs, problems_dir=problems,
                              agent=AgentSettings(claude_bin=str(fake_cli)), workspace=tmp_path / "ws",
                              lookup=lookup, ask=FakeAsk(), columns=[])
    assert research.load_brief("toy-s1e1", outputs) is None
    assert list((outputs / "toy-s1e1" / "research" / "sessions").glob("*/refused.json"))


def test_a_domain_that_is_not_blind_never_starts_a_session(tmp_path, fake_cli):
    problems = _problems(tmp_path, _yaml(DOMAIN + " Like toy-s1e1."))
    with pytest.raises(research.ResearchError, match="not blind"):
        research.run_research("toy-s1e1", outputs_root=tmp_path / "outputs", problems_dir=problems,
                              agent=AgentSettings(claude_bin=str(fake_cli)), workspace=tmp_path / "ws",
                              lookup=lookup, ask=FakeAsk(), columns=[])
    assert not (tmp_path / "seen.json").exists()


# --------------------------------------------------------------------------- #
# The loop: once per configuration, never fatal, the brief in every cycle's context
# --------------------------------------------------------------------------- #

def _pilot(tmp_path, problems):
    config = AutopilotConfig(problem="toy-s1e1", workspace_dir=str(tmp_path / "ws"))
    pilot = Autopilot(config, repo=tmp_path, outputs_root=tmp_path / "outputs", problems_dir=problems)
    pilot.state = {"problem": "toy-s1e1", "best": {"run_id": "r1"}, "cycles": []}
    return pilot


def test_loop_runs_research_once_per_configuration_and_survives_failures(tmp_path, monkeypatch):
    problems = _problems(tmp_path, _yaml())
    pilot = _pilot(tmp_path, problems)
    calls = []

    def fake_run(problem, **kwargs):
        calls.append(kwargs["config"].fingerprint())
        brief = {"config_fingerprint": kwargs["config"].fingerprint(), "items": [_item(1, ["openalex:W1"])],
                 "dropped": [], "audit": {"fetched": 1, "blocked": 0}, "review": {"status": "reviewed"},
                 "sources_before": "2019-07-15", "created_at": "now"}
        path = tmp_path / "outputs" / problem / "research" / research.BRIEF_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(brief))
        return brief

    monkeypatch.setattr(research, "run_research", fake_run)
    pilot.ensure_research()
    pilot.ensure_research()                                   # same settings: not run again
    assert len(calls) == 1
    (problems / "toy-s1e1" / research.CONFIG_FILE).write_text(_yaml(rest="sources_before: 2019-06-01\n"))
    monkeypatch.setattr(research, "run_research", lambda *a, **k: (_ for _ in ()).throw(research.ResearchError("x")))
    pilot.ensure_research()                                   # changed settings, failed run: logged, not raised
    assert "did not finish" in (tmp_path / "outputs" / "toy-s1e1" / "autopilot" / "autopilot.log").read_text()
    pilot.config.research = False
    pilot.ensure_research()


def test_cycles_get_the_brief_and_plans_cite_only_its_items(tmp_path):
    problems = _problems(tmp_path, _yaml())
    pilot = _pilot(tmp_path, problems)
    brief = {"items": [{**_item(1, []), "sources": [{"id": "openalex:W1", "title": "t", "counts": True}]}],
             "sources_before": "2019-07-15"}
    path = tmp_path / "outputs" / "toy-s1e1" / "research" / research.BRIEF_FILE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(brief))
    worktree = tmp_path / "wt"
    worktree.mkdir()
    pilot._write_context({"id": "C001", "worktree": str(worktree)})
    context = json.loads((worktree / ".autopilot" / research.CONTEXT_FILE).read_text())
    assert [i["id"] for i in context["items"]] == ["B01"] and context["sources_before"] == "2019-07-15"
    idea = {"title": "t", "kind": "settings", "hypothesis": "h", "evidence": "e", "novelty": "general",
            "predicted_gain": 0.0003, "spec_patch_yaml": "budget:\n  n_trials: 30\n", "brief_items": ["B01"]}
    plan = {"decision": "continue", "reason": "r", "idea": idea}
    assert validate_plan(plan, pilot._brief_ids()) == []
    assert validate_plan({**plan, "idea": {**idea, "brief_items": ["B07"]}}, pilot._brief_ids())
    assert validate_plan({**plan, "idea": {**idea, "brief_items": "B01"}})
    assert validate_plan({**plan, "idea": {**idea, "brief_items": ["B01"]}}, [])            # no brief: no citing
    probes = [{"script": "a.py", "question": "q", "finding": "f"}] * 2
    explore = {**plan, "idea": {**idea, "novelty": "adapted", "probes": probes}}
    assert validate_explore(explore, ["B01"]) == []
