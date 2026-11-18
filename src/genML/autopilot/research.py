"""Domain research for the autopilot: a cited brief on what carries signal in a problem's domain.

The one autopilot step with web access, built so it cannot reach a problem's answers even if it
tries:

1. **Blind.** The session gets the owner's description of the domain (``problems/<name>/research.yaml``),
   never the problem's name, its competition, its data files or its columns (``blindness_problems``
   refuses a description that names them). It runs in an empty folder with a random name, with no
   shell, so it cannot read code, data, notes, outputs or session logs.
2. **Reference sources only.** WebFetch reaches only scholarly indexes (OpenAlex, Semantic Scholar,
   arXiv), old Wikipedia revisions and sites the owner lists; there is no web search. Hosts that
   carry solutions (Kaggle, GitHub, blogs, forums) are denied outright.
3. **Older than the problem.** Searches must be limited to sources published before ``sources_before``
   (a competition's start date), and single records are read only when their date is earlier.
   Nothing published before a competition can hold its answers.

The gate hook (``research_gate.py``) enforces 2 and 3 on every fetch, because Claude Code lets some
documentation sites through without a permission rule; it fails closed. Afterwards trusted code
audits the session log (every successful fetch must pass the same gate again), checks every cited
source's date, and a second model (``second_opinion.ORDER``) drops items that are dataset artifacts or name a
competition rather than domain signals. The brief is saved as ``outputs/<problem>/research/brief.json``
and goes to every plan and explore session as ``domain_brief.json``; a plan that builds on an item
lists its id in ``idea.brief_items``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import yaml

from src.genML.autopilot import policy as policies
from src.genML.autopilot import research_gate as gate
from src.genML.autopilot.agent import AgentFailed, AgentSettings, run_agent

CONFIG_FILE = "research.yaml"
BRIEF_FILE = "brief.json"
CONTEXT_FILE = "domain_brief.json"
MODE_FILE = "research.md"
CONFIG_KEYS = {"domain", "sources_before", "extra_hosts", "max_items"}
DEFAULT_MAX_ITEMS = 20
MAX_ITEMS_LIMIT = 40
ITEM_ID = re.compile(r"^B\d{2}$")
# Words that would tell the session which problem it serves, or point it at answers.
TELLTALE_WORDS = ("kaggle", "playground", "leaderboard", "competition", "hackathon", "public notebook",
                  "synthetic", "generated dataset")
GATE_SCRIPT = Path(gate.__file__).resolve()
# Where the hook runs its copy of the gate: a checkout's own path can name the problem (autopilot
# worktrees are worktrees/<problem>/<cycle>), and the session sees the hook command.
GATE_COPY_ROOT = Path.home() / ".cache" / "genml-autopilot" / "gate"

RESEARCH_SYSTEM = """You research a domain for a team that builds machine-learning models on tabular \
data. Everything you report must come from sources you read in this session and must be useful to a \
practitioner of the domain: what carries signal, why, and how practitioners measure it. You have no \
shell and no web search; you read sources only with WebFetch, through a gate that explains every \
refusal. Write plainly and specifically, without em dashes, and with no attribution to yourself. Be \
honest: if a source does not support a point, leave the point out."""

REVIEW_KINDS = ("domain_signal", "domain_practice", "dataset_artifact", "names_a_source_to_avoid",
                "no_domain_reason", "unsupported", "other")
REVIEW_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "items"],
    "properties": {
        "summary": {"type": "string", "description": "Two to four sentences on the brief as a whole."},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "keep", "kind", "reason"],
                "properties": {
                    "id": {"type": "string"},
                    "keep": {"type": "boolean"},
                    "kind": {"type": "string", "enum": list(REVIEW_KINDS)},
                    "reason": {"type": "string", "description": "One sentence, citing the item."},
                },
            },
        },
    },
}
REVIEW_SYSTEM = """You review a research brief for genML, an automated pipeline for tabular machine \
learning. The brief lists signals that a research session found in published domain sources; later \
sessions use it to choose features and experiments. The project's rules for every idea:

{rules}

For each item decide keep or drop:
- keep (domain_signal or domain_practice): a property of the real-world process, or a practice of the \
domain's practitioners, with a reason a practitioner would accept and a source that supports it.
- drop as dataset_artifact: anything about how a particular dataset was generated, formatted or \
ordered (digit or rounding patterns, row order, ids, synthetic-data quirks, leaks).
- drop as names_a_source_to_avoid: an item that names a specific competition, a benchmark dataset's \
known solution, or other people's models or predictions.
- drop as no_domain_reason or unsupported: no reason a practitioner would accept, or the cited \
source does not support it.
You see only the material given; do not invent facts. Keep every item that is sound."""


class ResearchError(RuntimeError):
    """The research step cannot run, or its result was refused."""


@dataclass
class ResearchConfig:
    domain: str
    sources_before: date
    extra_hosts: List[str] = field(default_factory=list)
    max_items: int = DEFAULT_MAX_ITEMS

    @property
    def hosts(self) -> List[str]:
        return list(dict.fromkeys(list(gate.SCHOLARLY_HOSTS) + list(self.extra_hosts)))

    @property
    def last_day(self) -> date:
        return date.fromordinal(self.sources_before.toordinal() - 1)

    def fingerprint(self) -> str:
        text = json.dumps([self.domain, self.sources_before.isoformat(), sorted(self.extra_hosts), self.max_items])
        return hashlib.sha1(text.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Configuration and blindness
# --------------------------------------------------------------------------- #

def _problems_dir(problems_dir: Optional[Path]) -> Path:
    from src.genML.pipeline.problem import PROBLEMS_DIR

    return Path(problems_dir or PROBLEMS_DIR)


def _outputs(outputs_root: Optional[Path]) -> Path:
    from src.genML.run_context import OUTPUTS_ROOT

    return Path(outputs_root or OUTPUTS_ROOT)


def config_path(problem: str, problems_dir: Optional[Path] = None) -> Path:
    return _problems_dir(problems_dir) / problem / CONFIG_FILE


def load_config(problem: str, problems_dir: Optional[Path] = None) -> Optional[ResearchConfig]:
    """The problem's research settings, or None when it has no ``research.yaml``."""
    path = config_path(problem, problems_dir)
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ResearchError(f"{path}: not valid YAML ({exc})") from exc
    if not isinstance(data, dict):
        raise ResearchError(f"{path}: must be a mapping")
    errors = [f"unknown key '{k}'" for k in data if k not in CONFIG_KEYS]
    domain = data.get("domain")
    if not isinstance(domain, str) or not 40 <= len(domain.strip()) <= 2000:
        errors.append("domain must describe the domain in 40 to 2000 characters")
    before = data.get("sources_before")
    if isinstance(before, str):
        try:
            before = date.fromisoformat(before)
        except ValueError:
            before = None
    if isinstance(before, datetime):
        before = before.date()
    if not isinstance(before, date):
        errors.append("sources_before must be a date (YYYY-MM-DD): the problem's start, for example a competition's")
    extra = data.get("extra_hosts") or []
    if not isinstance(extra, list):
        errors.append("extra_hosts must be a list of host names")
        extra = []
    errors += policies.check_hosts([str(h).lower() for h in extra])
    max_items = data.get("max_items", DEFAULT_MAX_ITEMS)
    if not isinstance(max_items, int) or isinstance(max_items, bool) or not 3 <= max_items <= MAX_ITEMS_LIMIT:
        errors.append(f"max_items must be an integer from 3 to {MAX_ITEMS_LIMIT}")
    if errors:
        raise ResearchError(f"{path}: " + "; ".join(errors))
    return ResearchConfig(domain=" ".join(domain.split()), sources_before=before,
                          extra_hosts=[str(h).lower() for h in extra], max_items=max_items)


def problem_columns(problem: str) -> List[str]:
    """Column names of the problem's main training table (empty when the data is not here)."""
    try:
        from src.genML.autopilot.canary import main_train_file
        from src.genML.pipeline.problem import load_problem, locate_data_dir

        spec = load_problem(problem)
        with open(Path(locate_data_dir(spec)) / main_train_file(spec), newline="") as handle:
            return next(csv.reader(handle))
    except Exception:
        return []


def _identifier_like(name: str) -> bool:
    return "_" in name or any(ch.isdigit() for ch in name) or (name != name.lower() and name != name.upper())


def blindness_problems(config: ResearchConfig, problem: str, *, problems_dir: Optional[Path] = None,
                       columns: Sequence[str] = ()) -> List[str]:
    """Ways the domain description would tell the session which problem it serves (empty = blind)."""
    text = config.domain.lower()
    found = []
    names = {problem.lower(), *problem.lower().replace("_", "-").split("-")}
    spec_path = _problems_dir(problems_dir) / problem / "problem.yaml"
    if spec_path.exists():
        spec = yaml.safe_load(spec_path.read_text()) or {}
        source = spec.get("source") or {}
        for key in ("competition", "dataset", "path"):
            if source.get(key):
                names.add(str(source[key]).lower())
                names.add(Path(str(source[key])).name.lower())
        if spec.get("target"):
            names.add(str(spec["target"]).lower())
    for name in sorted(n for n in names if len(n) >= 5 and _identifier_like(n) or "-" in n or len(n) >= 12):
        if re.search(rf"(?<![a-z0-9_]){re.escape(name)}(?![a-z0-9_])", text):
            found.append(f"names the problem or its source ('{name}')")
    for word in TELLTALE_WORDS:
        if word in text:
            found.append(f"mentions '{word}'")
    for column in columns:
        lowered = str(column).lower()
        if _identifier_like(str(column)) and re.search(rf"(?<![a-z0-9_]){re.escape(lowered)}(?![a-z0-9_])", text):
            found.append(f"names the column '{column}'")
    if re.search(r"\b[\w-]+\.(csv|parquet|feather|zip)\b", text):
        found.append("names a data file")
    return found


# --------------------------------------------------------------------------- #
# The gate as the session's hook
# --------------------------------------------------------------------------- #

def gate_copy(script: Optional[Path] = None) -> Path:
    """The gate script (default: this checkout's) copied to a folder named by its content hash, so the
    hook command does not depend on where the checkout lives. The copy is read-only and rewritten when
    its bytes differ. The loop makes it for a worktree's script before each session: the folder is
    outside the worktree, where a sandboxed session cannot write."""
    source = Path(script) if script is not None else GATE_SCRIPT
    data = source.read_bytes()
    folder = GATE_COPY_ROOT / hashlib.sha256(data).hexdigest()[:16]
    target = folder / source.name
    try:
        if target.read_bytes() == data:
            return target
    except OSError:
        pass
    folder.mkdir(parents=True, exist_ok=True)
    temp = folder / f".{source.name}.{os.getpid()}.tmp"
    temp.write_bytes(data)
    temp.chmod(0o444)
    os.replace(temp, target)
    return target


def gate_command(config: ResearchConfig) -> str:
    """The hook command: the interpreter in isolated mode (its real path, not a virtualenv link inside
    the checkout; the gate needs only the standard library), the gate's copy, the cutoff and the
    owner's hosts. Nothing in it names the problem (the session sees it when a fetch is blocked)."""
    parts = [os.path.realpath(sys.executable), "-I", str(gate_copy()), "--before", config.sources_before.isoformat()]
    if config.extra_hosts:
        parts += ["--extra", ",".join(config.extra_hosts)]
    return " ".join(shlex.quote(p) for p in parts)


def gate_self_test(command: str, config: ResearchConfig) -> List[str]:
    """The hook must block an answer host and allow a dated search, run exactly as the session runs it."""
    cases = [("https://www.kaggle.com/competitions", 2),
             ("https://api.openalex.org/works?search=fraud&filter=to_publication_date:"
              f"{config.last_day.isoformat()}", 0),
             ("https://api.openalex.org/works?search=fraud", 2)]
    problems = []
    for url, expected in cases:
        event = json.dumps({"hook_event_name": "PreToolUse", "tool_name": "WebFetch", "tool_input": {"url": url}})
        try:
            done = subprocess.run(shlex.split(command), input=event, capture_output=True, text=True, timeout=30)
            code = done.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            problems.append(f"the gate hook did not run ({exc})")
            break
        if code != expected:
            problems.append(f"the gate hook returned {code} for {url} (expected {expected}): "
                            f"{done.stderr.strip()[:200]}")
    return problems


# --------------------------------------------------------------------------- #
# The session's result and its audit
# --------------------------------------------------------------------------- #

def validate_brief(data: Any, max_items: int = MAX_ITEMS_LIMIT) -> List[str]:
    """Problems with the research session's ``brief.json`` (empty list = valid)."""
    if not isinstance(data, dict):
        return ["the result must be a JSON object"]
    items = data.get("items")
    if not isinstance(items, list) or not items:
        return ["'items' must be a non-empty list"]
    errors = []
    if len(items) > max_items:
        errors.append(f"at most {max_items} items")
    seen = set()
    for n, item in enumerate(items):
        where = f"items[{n}]"
        if not isinstance(item, dict):
            errors.append(f"{where} must be an object")
            continue
        if not isinstance(item.get("id"), str) or not ITEM_ID.match(item["id"]) or item["id"] in seen:
            errors.append(f"{where}.id must be a new id like B01")
        seen.add(item.get("id"))
        for key in ("signal", "why", "practice"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                errors.append(f"{where}.{key} must be a non-empty string")
        sources = item.get("sources")
        if not isinstance(sources, list) or not sources:
            errors.append(f"{where}.sources must list at least one source")
            continue
        for k, source in enumerate(sources):
            if not isinstance(source, dict) or not isinstance(source.get("id"), str) or ":" not in source["id"] \
                    or not isinstance(source.get("title"), str):
                errors.append(f"{where}.sources[{k}] needs an id (kind:value) and a title")
    if "searches" in data and not (isinstance(data["searches"], list)
                                   and all(isinstance(s, str) for s in data["searches"])):
        errors.append("'searches' must be a list of strings")
    return errors


def session_tool_calls(log_path: Path) -> List[Dict[str, Any]]:
    """Every tool call in a session's stream-json log, with whether it succeeded."""
    calls: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for line in Path(log_path).read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = (event.get("message") or {}).get("content") if isinstance(event.get("message"), dict) else None
        for part in content if isinstance(content, list) else []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "tool_use":
                calls[part.get("id")] = {"tool": part.get("name"), "input": part.get("input") or {}, "ok": None}
                order.append(part.get("id"))
            elif part.get("type") == "tool_result" and part.get("tool_use_id") in calls:
                calls[part["tool_use_id"]]["ok"] = not bool(part.get("is_error"))
    return [calls[i] for i in order]


def _within(path: str, folder: Path) -> bool:
    try:
        Path(path).resolve().relative_to(folder.resolve())
        return True
    except (ValueError, OSError):
        return False


def audit_session(log_path: Path, folder: Path, config: ResearchConfig,
                  lookup: Optional[gate.Lookup] = None) -> Dict[str, Any]:
    """Re-check the session after the fact: only the research tools, files only in its folder, and
    every fetch that went through allowed by the gate (the hook is the guard; this proves it held)."""
    lookup = lookup or retrying(gate.http_lookup)
    calls = session_tool_calls(log_path)
    problems, fetched, blocked = [], [], 0
    for call in calls:
        tool, data = call["tool"], call["input"]
        if tool not in policies.RESEARCH_TOOLS:
            problems.append(f"the session used {tool}, which it should not have")
            continue
        if tool in ("Read", "Write") and call["ok"] is not False:
            path = str(data.get("file_path") or "")
            if path and not _within(path, folder):
                problems.append(f"the session's {tool} reached {path}, outside its folder")
        if tool == "WebFetch":
            url = str(data.get("url") or "")
            if call["ok"] is False:
                blocked += 1
                continue
            allowed, reason = gate.check_url(url, config.sources_before, config.extra_hosts, lookup)
            if not allowed:
                problems.append(f"{url} was fetched although the gate does not allow it ({reason})")
            else:
                fetched.append(url)
    return {"tool_calls": len(calls), "fetched": fetched, "blocked": blocked, "problems": problems}


def retrying(lookup: gate.Lookup, tries: int = 3, pause: float = 3.0,
             sleep: Callable[[float], None] = time.sleep) -> gate.Lookup:
    """A lookup that caches answers and retries transient failures (rate limits, timeouts)."""
    cache: Dict[Tuple[str, str], Optional[date]] = {}

    def wrapped(kind: str, ident: str) -> Optional[date]:
        key = (kind, ident)
        if key in cache:
            return cache[key]
        for attempt in range(tries):
            try:
                cache[key] = lookup(kind, ident)
                return cache[key]
            except Exception:
                if attempt == tries - 1:
                    raise
                sleep(pause * (attempt + 1))
        return None

    return wrapped


def verify_items(items: Sequence[Dict[str, Any]], config: ResearchConfig,
                 lookup: Optional[gate.Lookup] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Check every cited source's date and kind; keep items with at least one source that counts."""
    lookup = lookup or retrying(gate.http_lookup)
    kept, dropped = [], []
    for item in items:
        checked = []
        for source in item.get("sources") or []:
            counts, reason = gate.check_citation(source.get("id", ""), config.sources_before, config.extra_hosts,
                                                 lookup)
            checked.append({**source, "counts": counts, "check": reason})
        if any(s["counts"] for s in checked):
            kept.append({**item, "sources": checked})
        else:
            dropped.append({"id": item.get("id"), "signal": item.get("signal"), "by": "source check",
                            "reason": "; ".join(s["check"] for s in checked)[:600]})
    return kept, dropped


def review_items(problem: str, config: ResearchConfig, items: Sequence[Dict[str, Any]], *,
                 outputs_root: Optional[Path] = None, backends: Optional[Sequence[str]] = None,
                 ask: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """A second model's keep/drop on each item. No reviewer: every item stays, marked unreviewed."""
    from src.genML.autopilot import second_opinion

    if not items:
        return {"status": "nothing to review", "drop": {}}
    if ask is None:
        from src.genML.llm import ask
    material = {"domain": config.domain, "sources_before": config.sources_before.isoformat(),
                "items": [{"id": i["id"], "signal": i["signal"], "why": i["why"], "practice": i["practice"],
                           "sources": [{"id": s["id"], "title": s.get("title")} for s in i["sources"] if s["counts"]]}
                          for i in items]}
    prompt = ("Review each item of this research brief and say whether to keep it.\n\n## Material\n\n```json\n"
              f"{json.dumps(material, indent=1)}\n```")
    try:
        result = ask(REVIEW_SYSTEM.format(rules=second_opinion.rules_text()), prompt, REVIEW_SCHEMA,
                     purpose="research-review", problem=problem, backends=list(backends or second_opinion.ORDER),
                     outputs_root=outputs_root)
    except Exception as exc:  # no reviewer: the mechanical checks still stand
        return {"status": "unavailable", "summary": f"no review: {exc}"[:600], "drop": {}}
    verdicts = result.data["items"]
    return {"status": "reviewed", "backend": result.backend, "model": result.model,
            "summary": result.data["summary"], "verdicts": verdicts, "log_file": result.log_file,
            "drop": {v["id"]: v for v in verdicts if not v["keep"]}}


# --------------------------------------------------------------------------- #
# The step
# --------------------------------------------------------------------------- #

def _default_workspace() -> Path:
    from src.genML.autopilot.loop import REPO_ROOT, _default_workspace as loop_workspace

    return loop_workspace(REPO_ROOT)


def render_prompt(template: str, config: ResearchConfig, result_file: Path) -> str:
    values = {
        "DOMAIN": config.domain, "SOURCES_BEFORE": config.sources_before.isoformat(),
        "LAST_DAY": config.last_day.isoformat(), "LAST_DAY_COMPACT": config.last_day.strftime("%Y%m%d"),
        "RESULT_FILE": str(result_file), "MAX_ITEMS": str(config.max_items),
        "EXTRA_HOSTS": ", ".join(config.extra_hosts) if config.extra_hosts else "none",
    }
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def run_research(problem: str, *, config: Optional[ResearchConfig] = None, outputs_root: Optional[Path] = None,
                 problems_dir: Optional[Path] = None, modes_dir: Optional[Path] = None,
                 agent: Optional[AgentSettings] = None, workspace: Optional[Path] = None,
                 backends: Optional[Sequence[str]] = None, allow_unsandboxed: bool = False, timeout: int = 3600,
                 lookup: Optional[gate.Lookup] = None, ask: Optional[Callable[..., Any]] = None,
                 columns: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Run the research session, check its brief and save it. Returns the saved brief.

    Raises ``ResearchError`` when the step cannot run or its result is refused, and lets
    ``UsageLimited`` through (the caller decides whether to wait)."""
    from src.genML.autopilot.loop import MODES_DIR

    root = _outputs(outputs_root)
    config = config or load_config(problem, problems_dir)
    if config is None:
        raise ResearchError(f"{problem} has no {CONFIG_FILE}; write one to enable domain research")
    blind = blindness_problems(config, problem, problems_dir=problems_dir,
                               columns=problem_columns(problem) if columns is None else columns)
    if blind:
        raise ResearchError(f"{config_path(problem, problems_dir)}: the domain description is not blind: "
                            + "; ".join(blind))
    header, template = policies.load_research_mode(Path(modes_dir or MODES_DIR) / MODE_FILE)
    command = gate_command(config)
    broken = gate_self_test(command, config)
    if broken:
        raise ResearchError("the research gate does not work: " + "; ".join(broken))
    settings = policies.research_session_settings(header, hosts=config.hosts, gate_command=command,
                                                  allow_unsandboxed=allow_unsandboxed)
    base = Path(workspace or _default_workspace()) / "research"
    token = uuid.uuid4().hex[:12]
    folder, private = base / token, base / f"{token}-log"       # nothing in either name points at the problem
    folder.mkdir(parents=True)
    private.mkdir(parents=True)
    result_file = folder / BRIEF_FILE
    log_path = private / "research.jsonl"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = root / problem / "research" / "sessions" / stamp
    lookup = lookup or retrying(gate.http_lookup)
    try:
        prompt, session, raw = render_prompt(template, config, result_file), None, None
        for attempt in range(2):
            try:
                result = run_agent(prompt, cwd=folder, log_path=log_path, settings=agent or AgentSettings(),
                                   system_append=RESEARCH_SYSTEM, policy=header, settings_override=settings,
                                   session_id=session, resume=session is not None, timeout=timeout)
            except AgentFailed as exc:
                raise ResearchError(f"the research session failed: {exc}") from exc
            session = result.session_id
            try:
                raw = json.loads(result_file.read_text()) if result_file.exists() else None
                errors = validate_brief(raw, config.max_items) if raw is not None else ["the file does not exist"]
            except json.JSONDecodeError as exc:
                errors = [f"not valid JSON: {exc}"]
            if not errors:
                break
            if attempt == 1:
                raise ResearchError(f"no valid brief ({'; '.join(errors)}). Last message: {result.text[-400:]}")
            prompt = (f"The result file {result_file} is missing or invalid: {'; '.join(errors)}. "
                      "Write it now, as specified.")
        audit = audit_session(log_path, folder, config, lookup)
    finally:
        record.mkdir(parents=True, exist_ok=True)
        for path in (log_path, log_path.with_suffix(".settings.json"), result_file):
            if path.exists():
                shutil.copy(path, record / ("brief_raw.json" if path == result_file else path.name))
        shutil.rmtree(folder, ignore_errors=True)
        shutil.rmtree(private, ignore_errors=True)
    if audit["problems"]:
        (record / "refused.json").write_text(json.dumps(audit, indent=2))
        raise ResearchError("the session's log failed the audit, so the brief is refused: "
                            + "; ".join(audit["problems"][:5]))
    kept, dropped = verify_items(raw["items"], config, lookup)
    review = review_items(problem, config, kept, outputs_root=root, backends=backends, ask=ask)
    final = []
    for item in kept:
        verdict = review["drop"].get(item["id"])
        if verdict:
            dropped.append({"id": item["id"], "signal": item["signal"], "by": "review",
                            "reason": f"{verdict['kind']}: {verdict['reason']}"})
        else:
            final.append(item)
    brief = {
        "created_at": datetime.now().isoformat(timespec="seconds"), "problem": problem,
        "domain": config.domain, "sources_before": config.sources_before.isoformat(), "hosts": config.hosts,
        "config_fingerprint": config.fingerprint(), "status": "ok" if final else "empty",
        "items": final, "dropped": dropped, "searches": raw.get("searches") or [], "notes": raw.get("notes") or "",
        "review": {k: v for k, v in review.items() if k != "drop"},
        "audit": {k: (len(v) if k == "fetched" else v) for k, v in audit.items()},
        "session": str(record),
    }
    out = root / problem / "research" / BRIEF_FILE
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(brief, indent=2, default=str))
    tmp.replace(out)
    return brief


# --------------------------------------------------------------------------- #
# Reading the brief
# --------------------------------------------------------------------------- #

def load_brief(problem: str, outputs_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    path = _outputs(outputs_root) / problem / "research" / BRIEF_FILE
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except json.JSONDecodeError:
        return None


def brief_for_agents(brief: Dict[str, Any]) -> Dict[str, Any]:
    """What plan and explore sessions read: the kept items and the sources that passed the checks."""
    return {
        "about": ("Signals found in published domain research, all from sources dated before "
                  f"{brief['sources_before']} (before this problem existed), checked by trusted code and reviewed by a "
                  "second model. A brief item is a reason to test an idea, not evidence that it helps here: only a "
                  "measurement on this problem is. Cite the ids you build on in idea.brief_items."),
        "sources_before": brief["sources_before"],
        "items": [{"id": i["id"], "signal": i["signal"], "why": i["why"], "practice": i["practice"],
                   "sources": [{"id": s["id"], "title": s.get("title"), "check": s.get("check")}
                               for s in i["sources"] if s.get("counts")]}
                  for i in brief.get("items") or []],
    }


def item_ids(brief: Optional[Dict[str, Any]]) -> List[str]:
    return [i["id"] for i in (brief or {}).get("items") or []]


def format_brief(brief: Optional[Dict[str, Any]]) -> str:
    if not brief:
        return "No domain brief yet."
    audit, review = brief.get("audit") or {}, brief.get("review") or {}
    lines = [f"Domain brief for {brief.get('problem')} ({brief['status']}), {brief['created_at']}",
             f"  sources before {brief['sources_before']}; fetches {audit.get('fetched', 0)}, blocked by the gate "
             f"{audit.get('blocked', 0)}; review: {review.get('status')}"
             + (f" ({review.get('backend')})" if review.get("backend") else ""), ""]
    for item in brief.get("items") or []:
        cited = ", ".join(s["id"] for s in item["sources"] if s.get("counts"))
        lines.append(f"  {item['id']}  {item['signal']}")
        lines.append(f"       why: {item['why'][:300]}")
        lines.append(f"       sources: {cited}")
    if brief.get("dropped"):
        lines += ["", f"  Dropped ({len(brief['dropped'])}):"]
        lines += [f"  {d['id']}  [{d['by']}] {d.get('signal') or ''}: {d['reason'][:200]}" for d in brief["dropped"]]
    lines += ["", f"  Session log: {brief.get('session')}"]
    return "\n".join(lines)


__all__ = ["BRIEF_FILE", "CONFIG_FILE", "CONTEXT_FILE", "ResearchConfig", "ResearchError", "audit_session",
           "blindness_problems", "brief_for_agents", "config_path", "format_brief", "gate_command", "gate_self_test",
           "item_ids", "load_brief", "load_config", "problem_columns", "render_prompt", "retrying", "review_items",
           "run_research", "session_tool_calls", "validate_brief", "verify_items"]
