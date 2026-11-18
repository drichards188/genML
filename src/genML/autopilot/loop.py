"""The autopilot loop: plan, build, verify, run, judge, integrate; resumable at every step.

State lives in ``outputs/<problem>/autopilot/``:

* ``state.json``   anchors, the current best, every cycle and the step it reached
* ``journal.md``   what each cycle tried and what was measured (read by the next plan)
* ``cycles/C###/`` prompts, agent event logs, result files, test output, verdicts
* ``STOP``         create this file to stop after the current cycle

Kill the process at any point and start it again: each cycle continues from the
step it was on (agent sessions are resumed, a queued run is waited for, not relaunched).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import yaml

from src.genML.autopilot import judge as judging
from src.genML.autopilot import policy as policies
from src.genML.autopilot import probe as probes
from src.genML.autopilot import research
from src.genML.autopilot import second_opinion
from src.genML.autopilot import workspace as ws
from src.genML.autopilot.agent import AgentFailed, AgentSettings, UsageLimited, run_agent

logger = logging.getLogger("genml.autopilot")

REPO_ROOT = Path(__file__).resolve().parents[3]
MODES_DIR = REPO_ROOT / ".claude" / "skills" / "genml" / "modes"
STATE_FILE, JOURNAL_FILE, STOP_FILE, INBOX = "state.json", "journal.md", "STOP", "inbox"
SECTIONS = ("models", "budget", "feature_engineering", "ensemble")
MEASURED = ("accepted", "rejected", "suspicious")
LOOP_MODES = ("plan", "explore", "revise", "build", "fix", "conclude")   # each has a permission header
NOVELTY = ("general", "adapted", "new")  # never ideas from published solutions


class StepFailed(RuntimeError):
    """A cycle step could not finish; the cycle is recorded as failed."""


@dataclass
class AutopilotConfig:
    problem: str
    cycles: int = 1                       # cycles to run in this invocation
    until: Optional[str] = None           # ISO time; no new cycle starts after it
    explore_every: int = 3                # every Nth cycle must propose a new idea (0: never)
    max_fix_attempts: int = 2
    max_consecutive_failures: int = 3
    suspicious_ratio: float = judging.SUSPICIOUS_RATIO   # accepted gains this many times the noise bar wait
    suspicious_gain: Optional[float] = None  # optional absolute limit as well (metric units)
    canary: bool = True                   # rerun accepted code cycles on shuffled labels (leak test)
    plan_timeout: int = 4500                # plans and explores screen ideas with probes first
    build_timeout: int = 7200
    fix_timeout: int = 2700
    conclude_timeout: int = 1800
    verify_timeout: int = 2400
    run_timeout: int = 5 * 3600
    verify_command: Optional[List[str]] = None
    limit_wait_default: int = 1800        # usage limit without a stated reset time
    limit_wait_max_total: int = 10 * 3600
    workspace_dir: Optional[str] = None   # worktrees and downloaded data (default ~/.cache/genml-autopilot/...)
    directive: Optional[str] = None       # the next plan works on this
    plan_only: bool = False               # stop each cycle after the plan (dry run)
    allow_unsandboxed: bool = False       # owner's override: run sessions and runs without bubblewrap
    second_opinion: bool = True           # ask another model about each plan and each suspicious gain
    second_opinion_backends: Optional[List[str]] = None   # default: second_opinion.ORDER
    research: bool = True                 # run the domain research step once when research.yaml asks for it
    research_timeout: int = 3600
    agent: AgentSettings = field(default_factory=AgentSettings)


class JobRunner:
    """Runs experiments through the normal job queue, with the cycle's code."""

    def __init__(self, outputs_root: Optional[Path] = None, poll_seconds: float = 20.0):
        self.outputs_root = outputs_root
        self.poll_seconds = poll_seconds

    def launch(self, problem: str, options: Dict[str, Any], notes: str) -> Dict[str, Any]:
        from src.genML import jobs

        job = jobs.enqueue(problem, "run", options, notes, outputs_root=self.outputs_root)
        jobs.ensure_worker(self.outputs_root)
        return job

    def wait(self, job_id: int, timeout: float) -> Dict[str, Any]:
        from src.genML import jobs

        deadline = time.time() + timeout
        while True:
            job = jobs.get_job(job_id, self.outputs_root)
            if job["status"] not in ("queued", "running"):
                return job
            if time.time() > deadline:
                jobs.cancel_job(job_id, self.outputs_root)
                return {**jobs.get_job(job_id, self.outputs_root), "error": f"timed out after {timeout:.0f}s"}
            jobs.ensure_worker(self.outputs_root)
            time.sleep(self.poll_seconds)

    def log_tail(self, job_id: int, chars: int = 4000) -> str:
        from src.genML import jobs

        root = Path(self.outputs_root or jobs._default_root())
        path = root / "jobs" / f"job_{job_id}.log"
        return path.read_text(errors="replace")[-chars:] if path.exists() else ""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError, ValueError, TypeError):
        return False


def _default_workspace(repo: Path) -> Path:
    tag = hashlib.sha1(str(repo).encode()).hexdigest()[:8]
    return Path.home() / ".cache" / "genml-autopilot" / f"{repo.name}-{tag}"


def check_patch_structure(patch: Any) -> List[str]:
    """What any experiment may change: run settings only, never folds, target or metric."""
    if not isinstance(patch, dict) or not patch:
        return ["spec_patch_yaml must be a non-empty mapping"]
    errors = [f"'{key}' cannot be changed (allowed: {', '.join(SECTIONS)})" for key in patch if key not in SECTIONS]
    if "models" in patch and not (isinstance(patch["models"], list) and patch["models"]):
        errors.append("models must be a non-empty list")
    errors += [f"{key} must be a mapping" for key in ("budget", "feature_engineering", "ensemble")
               if key in patch and not isinstance(patch[key], dict)]
    return errors


def _missing(data: Any, required: Dict[str, Any]) -> List[str]:
    if not isinstance(data, dict):
        return ["the result must be a JSON object"]
    errors = []
    for key, kind in required.items():
        if key not in data:
            errors.append(f"missing '{key}'")
        elif not isinstance(data[key], kind):
            errors.append(f"'{key}' has the wrong type")
    return errors


def _brief_item_errors(idea: Dict[str, Any], brief_ids: Optional[List[str]]) -> List[str]:
    cited = idea.get("brief_items")
    if cited is None:
        return []
    if not isinstance(cited, list) or not all(isinstance(i, str) for i in cited):
        return ["idea.brief_items must be a list of brief item ids"]
    if brief_ids is not None:
        unknown = [i for i in cited if i not in brief_ids]
        if unknown:
            return [f"idea.brief_items names items that are not in domain_brief.json: {unknown}"]
    return []


def _probe_id_errors(idea: Dict[str, Any], probe_ids: Optional[List[str]]) -> List[str]:
    cited = idea.get("probe_ids")
    if cited is None:
        return []
    if not isinstance(cited, list) or not all(isinstance(i, str) for i in cited):
        return ["idea.probe_ids must be a list of probe ids (P01, P02, ...)"]
    if probe_ids is not None:
        unknown = [i for i in cited if i not in probe_ids]
        if unknown:
            return [f"idea.probe_ids names probes that are not in .autopilot/probes.jsonl: {unknown}"]
    return []


def validate_plan(data: Any, brief_ids: Optional[List[str]] = None,
                  probe_ids: Optional[List[str]] = None) -> List[str]:
    errors = _missing(data, {"decision": str, "reason": str})
    if errors:
        return errors
    if data["decision"] not in ("continue", "conclude"):
        return ["decision must be 'continue' or 'conclude'"]
    if data["decision"] == "conclude":
        return []
    idea = data.get("idea")
    errors = _missing(idea, {"title": str, "kind": str, "hypothesis": str, "evidence": str, "novelty": str,
                             "predicted_gain": (int, float)})
    if errors:
        return [f"idea: {e}" for e in errors]
    if idea["kind"] not in ("settings", "code"):
        errors.append("idea.kind must be 'settings' or 'code'")
    if idea["novelty"] not in NOVELTY:
        errors.append(f"idea.novelty must be one of {list(NOVELTY)}")
    if idea["kind"] == "settings" and not str(idea.get("spec_patch_yaml") or "").strip():
        errors.append("a settings idea needs idea.spec_patch_yaml")
    if idea["kind"] == "code" and not str(idea.get("build_brief") or "").strip():
        errors.append("a code idea needs idea.build_brief")
    return errors + _brief_item_errors(idea, brief_ids) + _probe_id_errors(idea, probe_ids)


def validate_explore(data: Any, brief_ids: Optional[List[str]] = None,
                     probe_ids: Optional[List[str]] = None) -> List[str]:
    """An explore plan must rest on its own probes and must not be a known public idea."""
    errors = validate_plan(data, brief_ids, probe_ids)
    if errors or data["decision"] == "conclude":
        return errors
    idea = data["idea"]
    if idea["novelty"] == "general":
        errors.append("an explore idea must be new or adapted, not general practice")
    probes = idea.get("probes")
    if not isinstance(probes, list) or len(probes) < 2:
        errors.append("idea.probes must list at least two probes (script, question, finding)")
    return errors


def validate_build(data: Any) -> List[str]:
    errors = _missing(data, {"summary": str, "spec_patch_yaml": str, "tests": list})
    if not errors and not data["spec_patch_yaml"].strip():
        errors.append("spec_patch_yaml must enable the new code")
    return errors


class Autopilot:
    def __init__(self, config: AutopilotConfig, repo: Path = REPO_ROOT, outputs_root: Optional[Path] = None,
                 problems_dir: Optional[Path] = None, modes_dir: Path = MODES_DIR, runner=None,
                 sleep: Callable[[float], None] = time.sleep, data_dir: Optional[Path] = None):
        from src.genML.pipeline.problem import PROBLEMS_DIR
        from src.genML.run_context import OUTPUTS_ROOT

        self.config = config
        self.problem = config.problem
        self.repo = Path(repo)
        self.outputs_root = Path(outputs_root or OUTPUTS_ROOT)
        self.problems_dir = Path(problems_dir or PROBLEMS_DIR)
        self.modes_dir = Path(modes_dir)
        self.runner = runner or JobRunner(outputs_root)
        self.sleep = sleep
        self.data_dir = data_dir
        self.home = self.outputs_root / self.problem / "autopilot"
        workspace = Path(config.workspace_dir) if config.workspace_dir else _default_workspace(self.repo)
        self.workspace = workspace
        self.worktrees = workspace / "worktrees"
        self.state: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    @property
    def state_path(self) -> Path:
        return self.home / STATE_FILE

    def save(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, default=str))
        tmp.replace(self.state_path)

    def load_or_init(self) -> Dict[str, Any]:
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            return self.state
        anchors = judging.make_anchors(self.problem, self.outputs_root)
        self.state = {
            "problem": self.problem, "created_at": _now(), "repo": str(self.repo), "base_commit": None,
            "main_branch": ws.SHARED_BRANCH, "anchors": anchors, "best": anchors["best"], "cycles": [],
            "status": "idle", "consecutive_failures": 0, "pending_directive": None,
        }
        self.save()
        self.log(f"Initialised: best {anchors['best']['experiment_id']} ({anchors['best']['score']:.6f} on the "
                 f"acceptance rows), run noise {judging.noise_sigma(anchors):.6f}")
        return self.state

    def refresh_code_base(self) -> None:
        """Bring the shared code branch up to date with the owner's working tree before any cycle."""
        if self.state.get("main_branch") != ws.SHARED_BRANCH:
            self.log(f"Switching from {self.state.get('main_branch')} to the shared {ws.SHARED_BRANCH}")
            self.state["main_branch"] = ws.SHARED_BRANCH
        try:
            refreshed = ws.refresh_shared_base(self.repo, self.state["main_branch"])
        except ws.GitError as exc:
            raise RuntimeError(f"{ws.SHARED_BRANCH} has accepted autopilot code that conflicts with your working "
                               f"tree; merge it into your branch first (git merge {ws.SHARED_BRANCH}). {exc}") from exc
        self.state["base_commit"] = refreshed["commit"]
        self.save()
        self.log(f"Code base {refreshed['action']}: {ws.SHARED_BRANCH} at {refreshed['commit'][:10]}")

    def log(self, message: str) -> None:
        line = f"[{_now()}] {message}"
        print(line, flush=True)
        self.home.mkdir(parents=True, exist_ok=True)
        with open(self.home / "autopilot.log", "a") as handle:
            handle.write(line + "\n")

    def cycle_dir(self, cycle: Dict[str, Any]) -> Path:
        path = self.home / "cycles" / cycle["id"]
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------ #
    # Loop
    # ------------------------------------------------------------------ #
    def stop_reason(self, started: int) -> Optional[str]:
        if (self.home / STOP_FILE).exists():
            return "STOP file present"
        if self.config.until and datetime.now() >= datetime.fromisoformat(self.config.until):
            return f"past {self.config.until}"
        if self.state.get("consecutive_failures", 0) >= self.config.max_consecutive_failures:
            return f"{self.state['consecutive_failures']} cycles in a row failed"
        if started >= self.config.cycles:
            return f"ran the {self.config.cycles} cycle(s) asked for"
        return None

    def preflight(self) -> List[str]:
        """Why the loop cannot start safely (empty list = it can): every mode's permission header
        and the run profile must be valid, and bubblewrap must work unless the owner overrides it."""
        problems = []
        for mode in LOOP_MODES:
            try:
                policies.load_mode(self.modes_dir / f"{mode}.md")
            except (policies.PolicyError, OSError) as exc:
                problems.append(str(exc))
        try:
            policies.load_run_policy()
        except (policies.PolicyError, OSError) as exc:
            problems.append(str(exc))
        problems += self.research_problems()
        if not self.config.allow_unsandboxed:
            ok, reason = policies.sandbox_available()
            if not ok:
                problems.append(f"no sandbox for sessions and runs: {reason}. Install it, or start with "
                                "--allow-unsandboxed to accept running agent code without one.")
        return problems

    def run(self) -> Dict[str, Any]:
        problems = self.preflight()
        if problems:
            raise RuntimeError("Autopilot not started:\n- " + "\n- ".join(problems))
        self.load_or_init()
        self.refresh_code_base()
        if self.config.allow_unsandboxed:
            self.log("WARNING: running without a sandbox (--allow-unsandboxed); agent code has your permissions")
        if self.state.get("status") in ("paused", "concluded") and not self.config.directive:
            self.log(f"Not starting: status is {self.state['status']} ({self.state.get('stop_reason')}). "
                     "Check it, then start again with a directive or reset the status.")
            return self.state
        other = self.state.get("pid")
        if self.state.get("status") == "running" and other and other != os.getpid() and _alive(other):
            raise RuntimeError(f"another autopilot (pid {other}) is running for {self.problem}")
        if self.config.directive:
            self.state["pending_directive"] = self.config.directive
        self.state.update(status="running", stop_reason=None, pid=os.getpid())
        self.save()
        started = 0
        try:
            self.ensure_research()
            unfinished = [c for c in self.state["cycles"] if c.get("step") != "done"]
            if unfinished:
                self.log(f"Resuming {unfinished[-1]['id']} at step {unfinished[-1]['step']}")
                self.run_cycle(unfinished[-1])
                started += 1
            while self.state["status"] == "running":
                reason = self.stop_reason(started)
                if reason:
                    self.state.update(status="stopped", stop_reason=reason)
                    break
                self.run_cycle(self.new_cycle())
                started += 1
        finally:
            if self.state.get("status") == "running":
                self.state["status"] = "stopped"
            self.state["pid"] = None
            self.save()
        self.log(f"Autopilot {self.state['status']}: {self.state.get('stop_reason')}")
        return self.state

    # ------------------------------------------------------------------ #
    # Domain research (research.py): once per problem, before the cycles
    # ------------------------------------------------------------------ #
    def research_problems(self) -> List[str]:
        """Why the research step could not run as configured (empty when it can, or is not asked for)."""
        if not self.config.research:
            return []
        try:
            config = research.load_config(self.problem, self.problems_dir)
            if config is None:
                return []
            policies.load_research_mode(self.modes_dir / research.MODE_FILE)
        except (research.ResearchError, policies.PolicyError, OSError) as exc:
            return [str(exc)]
        problems = research.blindness_problems(config, self.problem, problems_dir=self.problems_dir,
                                               columns=research.problem_columns(self.problem))
        if problems:
            return [f"{research.config_path(self.problem, self.problems_dir)}: the domain description is not blind: "
                    + "; ".join(problems)]
        return research.gate_self_test(research.gate_command(config), config)

    def ensure_research(self) -> None:
        """Run the research step when the problem asks for it and has no brief for its current
        settings. A failure is logged and the cycles go on without a brief."""
        if not self.config.research:
            return
        config = research.load_config(self.problem, self.problems_dir)
        if config is None:
            return
        brief = research.load_brief(self.problem, self.outputs_root)
        if brief and brief.get("config_fingerprint") == config.fingerprint():
            return
        self.state["status_detail"] = "domain research"
        self.save()
        self.log(f"Domain research: sources before {config.sources_before}, hosts {', '.join(config.hosts)}")
        try:
            brief = research.run_research(
                self.problem, config=config, outputs_root=self.outputs_root, problems_dir=self.problems_dir,
                modes_dir=self.modes_dir, agent=self.config.agent, workspace=self.workspace,
                backends=self.config.second_opinion_backends, allow_unsandboxed=self.config.allow_unsandboxed,
                timeout=self.config.research_timeout)
            review = brief.get("review") or {}
            self.log(f"Domain research: {len(brief['items'])} items kept, {len(brief['dropped'])} dropped; "
                     f"{brief['audit']['fetched']} fetches, {brief['audit']['blocked']} blocked by the gate; "
                     f"review {review.get('status')}")
        except (research.ResearchError, policies.PolicyError, UsageLimited, AgentFailed) as exc:
            self.log(f"Domain research did not finish; the cycles go on without a brief: {exc}"[:1500])
        finally:
            self.state.pop("status_detail", None)
            self.save()

    def next_directive(self) -> Optional[str]:
        """The directive for the next cycle: one given at start, else the oldest file in ``inbox/``.

        Queue one with ``queue_directive`` (``--queue-directive``); it can be added while the
        loop runs. A used file moves to ``inbox/done/``.
        """
        if self.state.get("pending_directive"):
            return self.state["pending_directive"]
        inbox = self.home / INBOX
        for path in sorted(inbox.glob("*.md")) if inbox.exists() else []:
            text = path.read_text().strip()
            done = inbox / "done"
            done.mkdir(parents=True, exist_ok=True)
            path.replace(done / path.name)
            if text:
                self.log(f"Directive from {path.name}: {text[:120]}")
                return text
        return None

    def new_cycle(self) -> Dict[str, Any]:
        number = len(self.state["cycles"]) + 1
        directive = self.next_directive()
        explore = bool(self.config.explore_every and number % self.config.explore_every == 0 and not directive)
        cycle = {"id": f"C{number:03d}", "mode": "explore" if explore else "plan", "status": "running",
                 "step": "plan", "started_at": _now(), "directive": directive, "sessions": {},
                 "fix_attempts": 0, "warnings": []}
        self.state["pending_directive"] = None
        self.state["cycles"].append(cycle)
        self.save()
        return cycle

    def run_cycle(self, cycle: Dict[str, Any]) -> None:
        self.log(f"{cycle['id']} ({cycle['mode']}) at step {cycle['step']}")
        try:
            self._prepare_worktree(cycle)
            if cycle["step"] == "plan":
                self.step_plan(cycle)
            if cycle["step"] == "review":
                self.step_review(cycle)
            if cycle["step"] == "build":
                self.step_build(cycle)
            if cycle["step"] == "verify":
                self.step_verify(cycle)
            if cycle["step"] == "run":
                self.step_run(cycle)
            if cycle["step"] == "judge":
                self.step_judge(cycle)
            if cycle["step"] == "canary":
                self.step_canary(cycle)
            if cycle["step"] == "integrate":
                self.step_integrate(cycle)
            if cycle["step"] == "conclude":
                self.step_conclude(cycle)
        except StepFailed as exc:
            self._finish(cycle, "failed", error=str(exc))
        except Exception as exc:  # unexpected: record, keep the loop safe
            logger.exception("cycle %s crashed", cycle["id"])
            self._finish(cycle, "failed", error=f"{type(exc).__name__}: {exc}")

    def _finish(self, cycle: Dict[str, Any], status: str, error: Optional[str] = None) -> None:
        cycle.update(status=status, step="done", finished_at=_now())
        if error:
            cycle["error"] = error[:4000]
        if status == "failed":
            self.state["consecutive_failures"] = self.state.get("consecutive_failures", 0) + 1
        elif status in MEASURED:
            self.state["consecutive_failures"] = 0
        self._journal(cycle)
        if cycle.get("worktree"):
            ws.remove_worktree(self.repo, Path(cycle["worktree"]))
        self.save()
        self.log(f"{cycle['id']} {status}" + (f": {error[:300]}" if error else ""))

    # ------------------------------------------------------------------ #
    # Worktree and agent sessions
    # ------------------------------------------------------------------ #
    def _prepare_worktree(self, cycle: Dict[str, Any]) -> None:
        if cycle.get("worktree") and Path(cycle["worktree"]).exists():
            return
        if cycle["step"] in ("done", "integrate"):
            return
        start = ws.branch_head(self.repo, self.state["main_branch"])
        if start is None:                                   # first use of the shared branch
            self.refresh_code_base()
            start = ws.branch_head(self.repo, self.state["main_branch"])
        branch = f"autopilot/{self.problem}/{cycle['id']}"
        path = self.worktrees / self.problem / cycle["id"]
        ws.add_worktree(self.repo, path, branch, start)
        cycle.update(worktree=str(path), branch=branch, start_commit=cycle.get("start_commit") or start)
        self.save()

    def _data_dir(self) -> Optional[Path]:
        if self.data_dir is not None:
            return self.data_dir
        try:
            from src.genML.pipeline.problem import load_problem, locate_data_dir

            return Path(locate_data_dir(load_problem(self.problem)))
        except Exception:
            return None

    def _values(self, cycle: Dict[str, Any]) -> Dict[str, str]:
        wt = Path(cycle["worktree"])
        best = self.state["best"]
        cycles_done = [c for c in self.state["cycles"] if c.get("step") == "done"]
        return {
            "PROBLEM": self.problem, "CYCLE": cycle["id"], "WORKTREE": str(wt),
            "SCRATCH": str(wt / ws.SCRATCH / "scratch"), "CONTEXT_DIR": str(wt / ws.SCRATCH),
            "OUTPUTS_DIR": str(self.outputs_root / self.problem), "DATA_DIR": str(self._data_dir() or "unknown"),
            "NOTES_FILE": str(self.problems_dir / self.problem / "notes.md"),
            "BEST": f"{best['experiment_id']} (run {best['run_id']}, {self.state['anchors']['metric']} "
                    f"{best['score']:.6f})",
            "DEADLINE": self.config.until or "none set", "PROTECTED": ", ".join(ws.PROTECTED),
            "DIRECTIVE": cycle.get("directive") or "none", "CYCLES_DONE": str(len(cycles_done)),
            "SUSPICIOUS_RATIO": f"{self.config.suspicious_ratio:g}",
            "JUDGE_BAR": self._judge_bar_text(),
            **self._facts(),
        }

    def _facts(self) -> Dict[str, str]:
        """Problem and machine details the prompts quote, measured here instead of written into them."""
        if getattr(self, "_facts_cache", None):
            return self._facts_cache
        import math

        from src.genML.autopilot.canary import main_train_file
        from src.genML.pipeline.problem import load_problem

        anchors = self.state.get("anchors") or {}
        try:
            spec = load_problem(self.problem)
        except Exception:  # the prompts must still render
            spec = None
        train_file = main_train_file(spec) if spec else "train.csv"
        other = [name for name, table in (spec.tables or {}).items() if table.get("parent")] if spec else []
        scheme = (anchors.get("rows") or {}).get("scheme") or {}
        if scheme.get("kind") == "time_forward":
            n_folds, judged = judging.TIME_CHUNKS, "the latest time block only (its models saw only earlier data)"
        else:
            folds = self.outputs_root / self.problem / f"folds_{anchors.get('fold_signature')}.npy"
            n_folds = int(np.load(folds).max()) + 1 if folds.exists() else 5
            judged = "80% of the rows (the rest is held back to check the gains at the end)"
        minutes = None
        best_run = (self.state.get("best") or {}).get("run_id")
        if best_run:
            try:
                run = json.loads((self.outputs_root / self.problem / best_run / "run.json").read_text())
                minutes = (datetime.fromisoformat(run["finished_at"]) - datetime.fromisoformat(run["started_at"]))
                minutes = round(minutes.total_seconds() / 60)
            except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
                minutes = None
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        from src.genML.gpu_utils import nvidia_smi

        smi = nvidia_smi()
        gpu = subprocess.run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True) if shutil.which(smi) or Path(smi).exists() else None
        gpu_text = (f"one GPU ({gpu.stdout.strip().splitlines()[0]}; XGBoost and CatBoost use it, LightGBM runs "
                    "on the CPU)") if gpu is not None and gpu.returncode == 0 and gpu.stdout.strip() else "no GPU"
        self._facts_cache = {
            "TRAIN_FILE": train_file,
            "OTHER_TABLES": f" plus related tables ({', '.join(other)})" if other else "",
            "TRAIN_ROWS": f"{anchors['n_rows']:,}" if anchors.get("n_rows") else "an unknown number of",
            "N_FOLDS": str(n_folds),
            "FOLDS_NEEDED": str(math.ceil(0.8 * n_folds)),
            "JUDGED_ON": judged,
            "RUN_MINUTES": f"about {minutes} minutes" if minutes else "an unknown time (no timed run yet)",
            "MACHINE": f"{ram_gb:.0f} GB RAM, {os.cpu_count()} CPU cores, {gpu_text}",
            "MEMORY_LIMIT": f"{ram_gb / 2:.0f} GB",
            "OOF_KIND": ("class probabilities" if anchors.get("task") != "regression"
                         else "predictions on the model's scale (see the transform file next to the target)"),
        }
        return self._facts_cache

    def _render(self, mode: str, cycle: Dict[str, Any], extra: Optional[Dict[str, str]] = None) -> str:
        """A mode file's prompt (its permission header removed) with the placeholders filled."""
        _, text = policies.split_mode((self.modes_dir / f"{mode}.md").read_text())
        for key, value in {**self._values(cycle), **(extra or {})}.items():
            text = text.replace("{{" + key + "}}", value)
        return text

    def _policy(self, mode: str) -> Dict[str, Any]:
        return policies.load_mode(self.modes_dir / f"{mode}.md")[0]

    def _write_context(self, cycle: Dict[str, Any]) -> None:
        """Files the plan reads: dossier, ledger, journal, current settings, earlier cycles."""
        from src.genML.experiments import baseline_settings, experiment_log
        from src.genML.pipeline.problem import load_problem

        context = Path(cycle["worktree"]) / ws.SCRATCH
        context.mkdir(parents=True, exist_ok=True)
        try:
            from src.genML.autopilot.dossier import build_dossier

            dossier = build_dossier(self.problem, run_id=self.state["best"]["run_id"], outputs_root=self.outputs_root)
        except Exception as exc:
            dossier = {"error": f"dossier could not be built: {exc}"}
        bar = self._judge_bar()
        measured = dossier.get("measurements")
        if bar and isinstance(measured, dict):
            measured["judge_bar"] = bar
            if isinstance(measured.get("seed_noise"), dict) and measured["seed_noise"].get("threshold") is not None:
                measured["seed_noise"]["note"] = ("this threshold is over all OOF rows; the judge's bar on its own "
                                                  "rows is judge_bar")
        (context / "dossier.json").write_text(json.dumps(dossier, indent=2, default=str))
        self._write_probe_context(cycle, context, bar)
        (context / "ledger.json").write_text(json.dumps(experiment_log(self.problem, self.outputs_root), indent=2,
                                                       default=str))
        journal = self.home / JOURNAL_FILE
        (context / "journal.md").write_text(journal.read_text() if journal.exists() else "(no cycles yet)\n")
        try:
            settings, source = baseline_settings(self.problem, {"run_id": self.state["best"]["run_id"]},
                                                 load_problem(self.problem), self.outputs_root)
            (context / "best_settings.yaml").write_text(f"# settings of {source}\n" + yaml.safe_dump(settings))
        except Exception as exc:
            (context / "best_settings.yaml").write_text(f"# could not read: {exc}\n")
        history = [{k: c.get(k) for k in ("id", "mode", "status", "error")} | {
            "idea": (c.get("idea") or {}).get("title"), "gain": (c.get("judge") or {}).get("gain")}
            for c in self.state["cycles"] if c is not cycle]
        (context / "cycles.json").write_text(json.dumps(history, indent=2, default=str))
        brief = research.load_brief(self.problem, self.outputs_root)
        brief_file = context / research.CONTEXT_FILE
        if brief and brief.get("items"):
            brief_file.write_text(json.dumps(research.brief_for_agents(brief), indent=2, default=str))
        elif brief_file.exists():
            brief_file.unlink()

    # ------------------------------------------------------------------ #
    # The judge's bar and probes (probe.py)
    # ------------------------------------------------------------------ #
    def _judge_bar(self) -> Optional[Dict[str, Any]]:
        """The bar the judge applies on its own rows: the last judged cycle's, else a lower bound.

        The dossier's seed-noise threshold is over all OOF rows, which for time splits is far lower
        (ieee-fraud: 0.0022 against the judge's 0.0070; C001 planned against the lower one)."""
        anchors = self.state.get("anchors") or {}
        if not anchors:
            return None
        sigma = judging.noise_sigma(anchors)
        z = float(anchors.get("z") or 3.0)
        rows = anchors.get("rows") or {}
        bar = {"sigma": sigma, "z": z, "rows": (rows.get("scheme") or {}).get("kind"), "n_rows": rows.get("n_accept")}
        judged = [c for c in self.state["cycles"] if (c.get("judge") or {}).get("threshold") is not None]
        if judged:
            bar.update(threshold=float(judged[-1]["judge"]["threshold"]),
                       source=f"the bar the judge applied in {judged[-1]['id']} (it moves a little with each run's "
                              "row noise)")
        else:
            bar.update(threshold=float(z * np.sqrt(2.0) * sigma),
                       source="a lower bound from seed noise alone; the judge adds each comparison's row noise")
        return bar

    def _judge_bar_text(self) -> str:
        bar = self._judge_bar()
        if not bar or not bar.get("threshold"):
            return "unknown"
        return f"{bar['threshold']:.4f} ({bar['source']})"

    def _write_probe_context(self, cycle: Dict[str, Any], context: Path, bar: Optional[Dict[str, Any]]) -> None:
        """What the probe harness reads in a session, and earlier cycles' probes with their outcomes."""
        try:
            probes.write_context(context, problem=self.problem, outputs_root=self.outputs_root,
                                 run_id=self.state["best"]["run_id"], anchors=self.state["anchors"], judge_bar=bar)
        except Exception as exc:  # no features to probe (a pruned or old best run): sessions go on without it
            for name in (probes.CONTEXT_FILE, probes.ROWS_FILE):
                (context / name).unlink(missing_ok=True)
            self.log(f"{cycle['id']}: no probe context ({exc}); sessions cannot use the probe harness")
        (context / "probe_history.json").write_text(json.dumps(self._probe_history(cycle), indent=2, default=str))

    def _probe_records(self, cycle_id: Optional[str] = None) -> List[Dict[str, Any]]:
        records = probes.read_log(self.home / probes.LOG_FILE)
        return [r for r in records if cycle_id is None or r.get("cycle") == cycle_id]

    def _cited_probes(self, cycle: Dict[str, Any]) -> List[Dict[str, Any]]:
        by_id = {r.get("id"): r for r in self._probe_records(cycle["id"])}
        return [by_id[i] for i in (cycle.get("idea") or {}).get("probe_ids") or [] if i in by_id]

    def _probe_history(self, current: Dict[str, Any], limit: int = 200) -> List[Dict[str, Any]]:
        """Earlier probes, each with what its cycle's run measured when the plan cited it."""
        cycles = {c["id"]: c for c in self.state["cycles"]}
        history = []
        for record in self._probe_records()[-limit:]:
            cycle = cycles.get(record.get("cycle")) or {}
            cited = record.get("id") in ((cycle.get("idea") or {}).get("probe_ids") or [])
            measured = (cycle.get("judge") or {}).get("gain") if cited else None
            history.append({**{k: record.get(k) for k in ("cycle", "id", "label", "gain", "noise", "gain_vs_bar",
                                                          "eval", "added", "dropped", "weights", "variant_params")},
                            "cited_by_plan": cited, "cycle_status": cycle.get("status") if cycle is not current
                            else "this cycle", "run_measured_gain": measured})
        return history

    def _collect_probes(self, cycle: Dict[str, Any], step: str) -> None:
        """Copy a session's new probe records into the problem's probe log (outputs/<problem>/autopilot)."""
        records = probes.read_log(Path(cycle["worktree"]) / ws.SCRATCH / probes.LOG_FILE)
        done = int(cycle.get("probes_collected") or 0)
        if len(records) <= done:
            return
        self.home.mkdir(parents=True, exist_ok=True)
        with open(self.home / probes.LOG_FILE, "a") as handle:
            for record in records[done:]:
                handle.write(json.dumps({"cycle": cycle["id"], "step": step, **record}, default=str) + "\n")
        cycle["probes_collected"] = len(records)
        self.save()

    def _brief_ids(self) -> Optional[List[str]]:
        """Ids a plan may cite in ``idea.brief_items`` (None when there is no brief: nothing to check)."""
        brief = research.load_brief(self.problem, self.outputs_root)
        return research.item_ids(brief) if brief else None

    def _read_places(self) -> Dict[str, Optional[Path]]:
        """Folders outside the worktree a header may name under ``read``."""
        return {"outputs": self.outputs_root / self.problem, "data": self._data_dir(),
                "notes": self.problems_dir / self.problem}

    def _prepare_gate_copy(self, wt: Path) -> None:
        """Make the research gate's copy for this worktree's script before a session runs there.

        The copy lives outside the worktree, where the session's sandbox cannot write, and a worktree's
        script differs from the owner's in line endings (git autocrlf), so it has a copy of its own.
        Without it the gate tests fail inside the session and pass in verify (C002, 2026-11-17).
        """
        script = wt / "src" / "genML" / "autopilot" / research.GATE_SCRIPT.name
        if not script.is_file():
            return
        try:
            research.gate_copy(script)
        except OSError as exc:
            self.log(f"Could not make the research gate's copy for the session ({exc}); its gate tests will fail")

    def _agent(self, cycle: Dict[str, Any], step: str, prompt: str, timeout: int,
               result_file: Optional[Path] = None, validator: Optional[Callable[[Any], List[str]]] = None,
               resume_session: Optional[str] = None, mode: Optional[str] = None) -> Any:
        """Run (or resume) a session until it leaves a valid result file; wait out usage limits.

        ``mode`` names the mode file whose permission header the session gets (default: ``step``).
        """
        folder = self.cycle_dir(cycle)
        (folder / f"{step}_prompt.md").write_text(prompt)
        system = self._render("_shared", cycle)
        policy = self._policy(mode or step)
        wt = Path(cycle["worktree"])
        places = self._read_places()
        read_dirs = [places[name] for name in policy.get("read") or []
                     if places.get(name) is not None and Path(places[name]).exists()]
        deny = list(ws.PROTECTED) + [str(self.outputs_root), str(self.repo)] + [str(p) for p in read_dirs]
        env = {"GENML_OUTPUTS": str(self.outputs_root), "GENML_PROBLEMS": str(self.problems_dir),
               probes.CONTEXT_ENV: str(wt / ws.SCRATCH / probes.CONTEXT_FILE)}
        session = cycle["sessions"].get(step) or resume_session
        resume = bool(cycle["sessions"].get(step)) or bool(resume_session)
        if cycle["sessions"].get(step):
            prompt = (f"You were interrupted. Continue the task from where you stopped and finish by writing "
                      f"{result_file}.") if result_file else "You were interrupted. Continue and finish the task."
        self._prepare_gate_copy(wt)
        waited, corrections = 0, 0
        while True:
            try:
                result = run_agent(prompt, cwd=wt, log_path=folder / f"{step}.jsonl", settings=self.config.agent,
                                   system_append=system, policy=policy, read_dirs=read_dirs, deny_paths=deny,
                                   allow_unsandboxed=self.config.allow_unsandboxed,
                                   session_id=session, resume=resume, timeout=timeout, env=env)
            except UsageLimited as exc:
                if exc.session_id:
                    cycle["sessions"][step] = session = exc.session_id
                    resume = True
                wait = (exc.reset_at - time.time() + 120) if exc.reset_at else self.config.limit_wait_default
                wait = max(300.0, min(float(wait), 6 * 3600.0))
                if waited + wait > self.config.limit_wait_max_total:
                    raise StepFailed(f"{step}: usage limit, gave up after waiting {waited / 3600:.1f} h")
                self.state["status_detail"] = f"waiting {wait / 60:.0f} min for the usage limit ({step})"
                self.save()
                self.log(f"{cycle['id']} {step}: usage limit; sleeping {wait / 60:.0f} min")
                self.sleep(wait)
                waited += wait
                prompt = ("You were interrupted by a usage limit. Continue the task and finish by writing "
                          f"{result_file}.") if result_file else "Continue and finish the task."
                continue
            except AgentFailed as exc:
                if resume and "No conversation found" in str(exc):
                    session, resume = None, False
                    prompt = (folder / f"{step}_prompt.md").read_text()
                    continue
                raise StepFailed(f"{step}: {exc}") from exc
            self.state.pop("status_detail", None)
            self._collect_probes(cycle, step)
            cycle["sessions"][step] = session = result.session_id
            cycle.setdefault("agent_runs", []).append({"step": step, **{k: v for k, v in result.to_dict().items()
                                                                         if k != "text"}})
            self.save()
            if result_file is None:
                return result
            errors = self._result_errors(result_file, validator)
            if not errors:
                return json.loads(result_file.read_text()) if result_file.suffix == ".json" else result_file.read_text()
            if corrections >= 1:
                raise StepFailed(f"{step}: no valid result file ({'; '.join(errors)}). Last message: "
                                 f"{result.text[-500:]}")
            corrections += 1
            resume = True
            prompt = (f"The result file {result_file} is missing or invalid: {'; '.join(errors)}. "
                      "Write it now, exactly as specified.")

    @staticmethod
    def _result_errors(path: Path, validator: Optional[Callable[[Any], List[str]]]) -> List[str]:
        if not path.exists():
            return ["the file does not exist"]
        if path.suffix != ".json":
            return [] if path.read_text().strip() else ["the file is empty"]
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            return [f"not valid JSON: {exc}"]
        return validator(data) if validator else []

    # ------------------------------------------------------------------ #
    # Steps
    # ------------------------------------------------------------------ #
    def step_plan(self, cycle: Dict[str, Any]) -> None:
        self._write_context(cycle)
        result_file = Path(cycle["worktree"]) / ws.SCRATCH / "plan.json"
        prompt = self._render(cycle["mode"], cycle, {"RESULT_FILE": str(result_file)})
        check = validate_explore if cycle["mode"] == "explore" else validate_plan
        brief_ids = self._brief_ids() or []

        def validator(data: Any) -> List[str]:
            return check(data, brief_ids, [r.get("id") for r in self._probe_records(cycle["id"])])

        plan = self._agent(cycle, "plan", prompt, self.config.plan_timeout, result_file, validator,
                           mode=cycle["mode"])
        shutil.copy(result_file, self.cycle_dir(cycle) / "plan.json")
        cycle["plan"] = {k: plan.get(k) for k in ("decision", "reason", "journal")}
        if plan["decision"] == "conclude":
            cycle["step"] = "conclude"
            self.save()
            return
        cycle["idea"] = plan["idea"]
        self.log(f"{cycle['id']} plan: [{plan['idea']['kind']}, {plan['idea']['novelty']}] {plan['idea']['title']} "
                 f"(predicted {plan['idea']['predicted_gain']:+.5f})")
        if self.config.plan_only:
            self._finish(cycle, "planned")
            return
        if self.config.second_opinion:
            cycle["step"] = "review"
        else:
            cycle["step"] = "build" if plan["idea"]["kind"] == "code" else "run"
        self.save()

    # ------------------------------------------------------------------ #
    # Second opinion
    # ------------------------------------------------------------------ #
    def _ask_opinion(self, cycle: Dict[str, Any], label: str, question: str,
                     material: Dict[str, Any]) -> Dict[str, Any]:
        """Ask the second opinion and record it; a reviewer that cannot answer means 'unavailable'."""
        try:
            opinion = second_opinion.ask_second_opinion(self.problem, question, material,
                                                        outputs_root=self.outputs_root,
                                                        backends=self.config.second_opinion_backends)
        except Exception as exc:  # no CLI, usage limit, off-schema answer: advice is optional
            opinion = {"verdict": "unavailable", "summary": f"no second opinion: {exc}"[:600], "concerns": [],
                       "suggestion": ""}
        opinion = {"label": label, "at": _now(), **opinion}
        cycle.setdefault("second_opinions", []).append(opinion)
        (self.cycle_dir(cycle) / f"second_opinion_{label}.json").write_text(json.dumps(opinion, indent=2))
        self.save()
        concerns = "; ".join(f"[{c['kind']}] {c['detail']}" for c in opinion.get("concerns") or [])
        self.log(f"{cycle['id']} second opinion on the {label} ({opinion.get('backend', '-')}): "
                 f"{opinion['verdict']}" + (f": {concerns[:300]}" if concerns else ""))
        return opinion

    def _plan_material(self, cycle: Dict[str, Any]) -> Dict[str, Any]:
        from src.genML.experiments import load_ledger

        context = Path(cycle["worktree"]) / ws.SCRATCH
        try:
            dossier = json.loads((context / "dossier.json").read_text())
        except (OSError, json.JSONDecodeError):
            dossier = {}
        notes = self.problems_dir / self.problem / "notes.md"
        history = [{"id": c["id"], "mode": c.get("mode"), "status": c.get("status"),
                    "idea": (c.get("idea") or {}).get("title"), "gain": (c.get("judge") or {}).get("gain"),
                    "error": (c.get("error") or "")[:200] or None}
                   for c in self.state["cycles"] if c is not cycle]
        brief = research.load_brief(self.problem, self.outputs_root) or {}
        cited = set(cycle["idea"].get("brief_items") or [])
        brief_items = [item for item in research.brief_for_agents(brief)["items"] if item["id"] in cited] if brief else []
        idea = dict(cycle["idea"])
        cited = self._cited_probes(cycle)
        if cited:
            idea["probe_results"] = [{k: r.get(k) for k in ("id", "label", "eval", "added", "dropped", "weights",
                                                            "variant_params", "base", "variant", "gain", "noise",
                                                            "gain_vs_bar")} for r in cited]
        return second_opinion.plan_material(idea, notes=notes.read_text() if notes.exists() else None,
                                            history=history, ledger=load_ledger(self.problem, self.outputs_root),
                                            dossier=dossier, brief_items=brief_items)

    def _cycle_diff(self, cycle: Dict[str, Any]) -> str:
        if (cycle.get("idea") or {}).get("kind") != "code" or not cycle.get("commit"):
            return ""
        return ws.git(self.repo, "diff", cycle["start_commit"], cycle["commit"], check=False)

    def step_review(self, cycle: Dict[str, Any]) -> None:
        """A second opinion on the plan before a build or a run is spent on it.

        'concerns' or 'reject' give the plan session one revision; a revised plan that is rejected
        again fails the cycle. Anything else (including no reviewer) goes ahead: the judge decides."""
        opinion = self._ask_opinion(cycle, "plan", second_opinion.PLAN_QUESTION, self._plan_material(cycle))
        if opinion["verdict"] in ("concerns", "reject") and not cycle.get("revised"):
            self._revise_plan(cycle, opinion)
            if cycle["step"] == "conclude":
                return
            if opinion["verdict"] == "reject":
                opinion = self._ask_opinion(cycle, "revised_plan", second_opinion.PLAN_QUESTION,
                                            self._plan_material(cycle))
        if opinion["verdict"] == "reject" and cycle.get("revised"):
            raise StepFailed(f"the second opinion rejected the revised plan: {opinion.get('summary', '')[:600]}")
        cycle["step"] = "build" if cycle["idea"]["kind"] == "code" else "run"
        self.save()

    def _revise_plan(self, cycle: Dict[str, Any], opinion: Dict[str, Any]) -> None:
        """Resume the plan session with the second opinion; it rewrites plan.json once."""
        context = Path(cycle["worktree"]) / ws.SCRATCH
        result_file = context / "plan.json"
        if not cycle.get("revise_started"):
            if result_file.exists():
                result_file.replace(context / "plan_original.json")   # the session must write a new one
            cycle["revise_started"] = True
            self.save()
        shown = {k: opinion.get(k) for k in ("verdict", "summary", "concerns", "suggestion")}
        prompt = self._render("revise", cycle, {
            "RESULT_FILE": str(result_file), "OPINION_JSON": json.dumps(shown, indent=2),
            "REVIEWER": f"{opinion.get('backend') or 'another model'} / {opinion.get('model') or '-'}"})
        check = validate_explore if cycle["mode"] == "explore" else validate_plan
        brief_ids = self._brief_ids() or []

        def validator(data: Any) -> List[str]:
            return check(data, brief_ids, [r.get("id") for r in self._probe_records(cycle["id"])])

        plan = self._agent(cycle, "revise", prompt, self.config.plan_timeout, result_file, validator,
                           resume_session=cycle["sessions"].get("plan"), mode="revise")
        shutil.copy(result_file, self.cycle_dir(cycle) / "plan_revised.json")
        cycle["plan_before_review"] = cycle.get("idea")
        cycle["plan"] = {k: plan.get(k) for k in ("decision", "reason", "journal")}
        cycle["revised"] = True
        if plan["decision"] == "conclude":
            cycle["step"] = "conclude"
            self.save()
            return
        cycle["idea"] = plan["idea"]
        self.save()
        self.log(f"{cycle['id']} revised plan: [{plan['idea']['kind']}] {plan['idea']['title']} "
                 f"(predicted {plan['idea']['predicted_gain']:+.5f})")

    def step_build(self, cycle: Dict[str, Any]) -> None:
        result_file = Path(cycle["worktree"]) / ws.SCRATCH / "build.json"
        idea = cycle["idea"]
        prompt = self._render("build", cycle, {
            "RESULT_FILE": str(result_file), "IDEA_JSON": json.dumps(idea, indent=2)})
        build = self._agent(cycle, "build", prompt, self.config.build_timeout, result_file, validate_build)
        shutil.copy(result_file, self.cycle_dir(cycle) / "build.json")
        cycle["build"] = build
        cycle["step"] = "verify"
        self.save()

    def _fix(self, cycle: Dict[str, Any], problems: str) -> None:
        if cycle["fix_attempts"] >= self.config.max_fix_attempts:
            raise StepFailed(f"still failing after {cycle['fix_attempts']} fix attempt(s): {problems[:1500]}")
        cycle["fix_attempts"] += 1
        self.save()
        result_file = Path(cycle["worktree"]) / ws.SCRATCH / "build.json"
        prompt = self._render("fix", cycle, {"RESULT_FILE": str(result_file), "PROBLEMS": problems[-12000:],
                                             "ATTEMPT": str(cycle["fix_attempts"])})
        build = self._agent(cycle, f"fix{cycle['fix_attempts']}", prompt, self.config.fix_timeout, result_file,
                            validate_build, resume_session=cycle["sessions"].get("build"), mode="fix")
        shutil.copy(result_file, self.cycle_dir(cycle) / f"build_fix{cycle['fix_attempts']}.json")
        cycle["build"] = build
        self.save()

    def _verify_command(self, wt: Path) -> List[str]:
        if self.config.verify_command:
            return list(self.config.verify_command)
        python = wt / ".venv" / "bin" / "python"
        return [str(python if python.exists() else sys.executable), "-m", "pytest", "-m", "not slow", "-q", "-x",
                "-p", "no:cacheprovider"]

    def step_verify(self, cycle: Dict[str, Any]) -> None:
        wt = Path(cycle["worktree"])
        while True:
            info = ws.commit_cycle(wt, cycle["start_commit"], f"{cycle['id']}: {cycle['idea']['title'][:72]}")
            cycle["commit"] = ws.head(wt)
            cycle["files"] = info["files"]
            for key, label in (("reverted_protected", "protected files"), ("reverted_tests", "existing tests")):
                if info[key]:
                    cycle["warnings"].append(f"reverted changes to {label}: {', '.join(info[key])}")
            self.save()
            problems = []
            if info["reverted_protected"] or info["reverted_tests"]:
                problems.append("These changes were reverted because agents may not edit them: "
                                + ", ".join(info["reverted_protected"] + info["reverted_tests"])
                                + ". Make the change work without them (new behavior must be opt-in).")
            source = [f for f in info["files"] if f.startswith("src/")]
            if not source:
                problems.append("No source changes under src/ were committed; the idea was not built.")
            elif ws.added_test_functions(wt, cycle["start_commit"]) == 0:
                problems.append("No new test functions were added under tests/. Add unit tests, and a leak test "
                                "for anything computed from labels.")
            proc = self._run_tests(cycle, wt)
            if proc is not None:
                problems.append(f"The fast test suite failed:\n{proc}")
            if not problems:
                cycle["step"] = "run"
                self.save()
                return
            self.log(f"{cycle['id']} verify: {len(problems)} problem(s); asking the build session to fix")
            self._fix(cycle, "\n\n".join(problems))

    def _run_tests(self, cycle: Dict[str, Any], wt: Path) -> Optional[str]:
        """None when the suite passes, else the tail of its output."""
        log_path = self.cycle_dir(cycle) / f"verify_{cycle['fix_attempts']}.log"
        try:
            proc = subprocess.run(self._verify_command(wt), cwd=wt, capture_output=True, text=True,
                                  timeout=self.config.verify_timeout)
            output, code = proc.stdout + proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            output, code = f"{exc.stdout or ''}\n{exc.stderr or ''}\nTIMED OUT after {self.config.verify_timeout}s", 1
        log_path.write_text(output if isinstance(output, str) else str(output))
        cycle["verify"] = {"returncode": code, "log": str(log_path)}
        if code == 0:
            return None
        # pytest prints its warnings after the tracebacks: lead with what failed
        lines = output.splitlines()
        failed = [line for line in lines if line.startswith(("FAILED ", "ERROR "))]
        totals = [line for line in lines[-5:] if re.search(r"\b(failed|error|errors|passed)\b", line)]
        head = "\n".join(failed[:20] + totals[-1:])
        return (head + "\n\n" if head else "") + output[-6000:]

    def _experiment_settings(self, cycle: Dict[str, Any]) -> Dict[str, Any]:
        from src.genML.experiments import (_changes, baseline_settings, check_patch, deep_merge, fingerprint,
                                           leak_problems, load_ledger)
        from src.genML.pipeline.problem import load_problem

        patch_text = (cycle.get("build") or {}).get("spec_patch_yaml") or cycle["idea"].get("spec_patch_yaml") or ""
        try:
            patch = yaml.safe_load(patch_text) or {}
        except yaml.YAMLError as exc:
            raise StepFailed(f"spec_patch_yaml is not valid YAML: {exc}")
        spec = load_problem(self.problem)
        target = self.state["anchors"].get("target") or spec.target
        errors = check_patch_structure(patch)
        # Settings ideas may use only what this checkout knows; a code idea's build may add settings,
        # but the leak rules hold for both.
        errors += check_patch(patch, target=target, known_only=cycle["idea"]["kind"] != "code")
        if errors:
            raise StepFailed("; ".join(dict.fromkeys(errors)))
        base, source = baseline_settings(self.problem, {"run_id": self.state["best"]["run_id"]}, spec,
                                         self.outputs_root)
        merged = deep_merge(base, patch)
        leaks = leak_problems(merged, target)
        if leaks:
            raise StepFailed("; ".join(leaks))
        changes = _changes(base, merged)
        if not changes:
            raise StepFailed(f"the patch changes nothing compared with the settings of {source}")
        code = cycle.get("commit") or cycle["start_commit"]
        fp = fingerprint({**merged, "_code": code})
        tried = {c.get("fingerprint") for c in self.state["cycles"] if c is not cycle and c.get("status") in MEASURED}
        tried |= {e.get("fingerprint") for e in load_ledger(self.problem, self.outputs_root)
                  if e.get("status") in MEASURED}
        settings_only = cycle["idea"]["kind"] != "code"
        if fp in tried or (settings_only and fingerprint(merged) in tried):
            raise StepFailed("these exact settings (with this code) were already measured")
        return {"patch": patch, "overrides": merged, "changes": changes, "fingerprint": fp, "built_on": source}

    def step_run(self, cycle: Dict[str, Any]) -> None:
        if not cycle.get("job_id"):
            if cycle["idea"]["kind"] != "code":
                # A settings idea runs the cycle's start code. Anything a plan session wrote into the
                # worktree was never verified or guarded, so it is discarded before the run.
                discarded = ws.reset_worktree(Path(cycle["worktree"]), cycle["start_commit"])
                if discarded:
                    cycle["warnings"].append(f"discarded unverified worktree changes: {', '.join(discarded)[:300]}")
            settings = self._experiment_settings(cycle)
            cycle.update(settings)
            options = {**settings["overrides"], "code_dir": cycle["worktree"]}
            if self.config.allow_unsandboxed:
                options["unsandboxed"] = True
            job = self.runner.launch(self.problem, options, f"[{cycle['id']} autopilot] {cycle['idea']['title']}")
            cycle.update(job_id=job["id"], run_id=job["run_id"], run_started_at=_now())
            self.save()
            self.log(f"{cycle['id']} run: job {job['id']} (run {job['run_id']}); changes: {settings['changes']}")
        job = self.runner.wait(cycle["job_id"], self.config.run_timeout)
        if job["status"] == "completed":
            cycle["step"] = "judge"
            self.save()
            return
        error = job.get("error") or job["status"]
        tail = self.runner.log_tail(cycle["job_id"]) if hasattr(self.runner, "log_tail") else ""
        cycle.setdefault("failed_runs", []).append({"job_id": cycle["job_id"], "run_id": cycle["run_id"],
                                                    "error": str(error)[:1000]})
        if cycle["idea"]["kind"] != "code":
            raise StepFailed(f"run failed: {error}")
        self.log(f"{cycle['id']} run failed ({str(error)[:200]}); asking the build session to fix")
        cycle.pop("job_id", None)
        cycle.pop("run_id", None)
        self._fix(cycle, f"The full run on the real data failed: {error}\n\nEnd of the run log:\n{tail}")
        cycle["step"] = "verify"
        self.save()
        self.step_verify(cycle)
        self.step_run(cycle)

    def step_judge(self, cycle: Dict[str, Any]) -> None:
        result = judging.judge_run(self.state["anchors"], self.state["best"], self.problem, cycle["run_id"],
                                   self.outputs_root, suspicious_ratio=self.config.suspicious_ratio,
                                   suspicious_gain=self.config.suspicious_gain)
        holdout = result.pop("holdout", None)        # for the person only; agents never see it
        if holdout:
            self.state.setdefault("holdout_log", {})[cycle["id"]] = holdout
        cycle["judge"] = result
        (self.cycle_dir(cycle) / "judge.json").write_text(json.dumps(result, indent=2, default=str))
        code_gain = cycle["idea"]["kind"] == "code" and result["verdict"] in ("accepted", "suspicious")
        cycle["step"] = "canary" if code_gain and self.config.canary else "integrate"
        self.save()

    def step_canary(self, cycle: Dict[str, Any]) -> None:
        """Run the cycle's code and settings on shuffled labels; they must not beat a constant prediction.

        Only for code cycles the judge accepted or held as suspicious. A leak rejects the cycle; a canary
        that cannot finish holds the cycle for a person (the code is unverified)."""
        from src.genML.autopilot.canary import prepare_canary_data

        info = cycle.setdefault("canary", {})
        root = self.worktrees.parent / "canary" / self.problem / cycle["id"]
        if not info.get("job_id"):
            data = prepare_canary_data(self.problem, root / "data", seed=int(cycle["id"][1:]) + 1000)
            options = {**cycle["overrides"], "code_dir": cycle["worktree"],
                       "source": {"type": "csv", "path": str(data)}, "outputs_root": str(root / "outputs")}
            if self.config.allow_unsandboxed:
                options["unsandboxed"] = True
            job = self.runner.launch(self.problem, options, f"[{cycle['id']} canary] shuffled labels")
            info.update(job_id=job["id"], run_id=job["run_id"], root=str(root))
            self.save()
            self.log(f"{cycle['id']} canary: job {job['id']} (run {job['run_id']}) on shuffled labels")
        job = self.runner.wait(info["job_id"], self.config.run_timeout)
        if job["status"] == "completed":
            result = judging.canary_verdict(self.state["anchors"], root / "outputs", self.problem, info["run_id"])
        else:
            result = {"verdict": "failed", "errors": [f"canary run {job['status']}: {job.get('error') or ''}"[:500]]}
        info["result"] = result
        (self.cycle_dir(cycle) / "canary.json").write_text(json.dumps(result, indent=2, default=str))
        self._free_canary(root)
        judged = cycle["judge"]
        if result["verdict"] == "leak":
            judged.update(verdict="rejected", canary="leak")
            cycle["warnings"].append(
                f"shuffled-label test: the code beat a constant prediction on shuffled labels "
                f"({result['gain_over_constant']:+.6f} > {result['bar']:.6f}); rejected as a leak")
        elif result["verdict"] != "clean":
            judged.update(verdict="suspicious", canary="unverified")
            cycle["warnings"].append(f"shuffled-label test did not finish: {'; '.join(result.get('errors', []))}")
        else:
            judged["canary"] = "clean"
        self.log(f"{cycle['id']} canary: {result['verdict']}")
        cycle["step"] = "integrate"
        self.save()

    def _free_canary(self, root: Path) -> None:
        """The verdict is recorded: drop the shuffled data copy and the canary run's tables and matrices.
        Its reports and OOF store stay (the verdict can be recomputed from them)."""
        from src.genML.pruning import RUN_ID, prune_run

        shutil.rmtree(root / "data", ignore_errors=True)
        runs = root / "outputs" / self.problem
        for folder in sorted(runs.iterdir()) if runs.is_dir() else []:
            if RUN_ID.match(folder.name):
                try:
                    prune_run(folder)
                except Exception as exc:       # cleanup never decides a cycle
                    self.log(f"canary cleanup of {folder.name} failed: {exc}")

    def step_integrate(self, cycle: Dict[str, Any]) -> None:
        result = cycle["judge"]
        status = result["verdict"]
        if status == "accepted":
            if cycle["idea"]["kind"] == "code" and cycle.get("commit"):
                self._land_code(cycle)
            self.state["best"] = judging.best_record(self.outputs_root / self.problem,
                                                     result["candidate_experiment_id"], result["candidate_score"],
                                                     cycle["run_id"])
            held = (self.state.get("holdout_log") or {}).get(cycle["id"])
            if held:
                self.state["holdout_best"] = held["candidate"]
        self._record_ledger(cycle, status)
        triage = ""
        if status == "suspicious" and self.config.second_opinion:
            opinion = self._ask_opinion(cycle, "gain", second_opinion.GAIN_QUESTION, second_opinion.gain_material(
                cycle["idea"], result, build=cycle.get("build"), diff=self._cycle_diff(cycle),
                changes=cycle.get("changes")))
            triage = (f" Second opinion ({opinion.get('backend', '-')}): {opinion['verdict']}: "
                      f"{opinion.get('summary', '')[:400]} (cycles/{cycle['id']}/second_opinion_gain.json)")
        if status in ("suspicious", "integrity_failure"):
            self.state.update(status="paused", stop_reason=(
                f"{cycle['id']}: {status}; a person must check it "
                + ("(a gain far above the noise bar, or code the shuffled-label test could not verify: check it "
                   "for a leak before anything builds on it)"
                   if status == "suspicious" else "; ".join(result.get("errors", []))) + triage))
        if status in MEASURED:
            gain = f"{result['gain']:+.6f} vs bar {result['threshold']:.6f}, folds {result['folds_won']}"
            self.log(f"{cycle['id']} judged {status}: {gain}")
            self._finish(cycle, status)
        else:
            self._finish(cycle, "failed" if status == "failed" else status,
                         error="; ".join(result.get("errors", [])) or status)

    def _land_code(self, cycle: Dict[str, Any]) -> None:
        """Put accepted code on the shared branch: fast-forward, or merge if another loop moved it."""
        branch = self.state["main_branch"]
        try:
            ws.move_branch(self.repo, branch, cycle["commit"])
        except ws.GitError:
            try:
                ws.merge_into(self.repo, branch, cycle["commit"], f"{cycle['id']} ({self.problem}): accepted code")
            except ws.GitError as exc:
                cycle["warnings"].append(f"accepted code could not be merged into {branch}: {str(exc)[:300]}")
                self.state.update(status="paused", stop_reason=(
                    f"{cycle['id']}: accepted, but its code conflicts with {branch}; merge "
                    f"{cycle.get('branch')} by hand"))

    def step_conclude(self, cycle: Dict[str, Any]) -> None:
        report = Path(cycle["worktree"]) / ws.SCRATCH / "final_report.md"
        prompt = self._render("conclude", cycle, {"RESULT_FILE": str(report)})
        self._agent(cycle, "conclude", prompt, self.config.conclude_timeout, report)
        shutil.copy(report, self.home / "final_report.md")
        self.state.update(status="concluded", stop_reason=f"{cycle['id']} concluded: {cycle['plan'].get('reason')}")
        self._finish(cycle, "concluded")

    # ------------------------------------------------------------------ #
    # Records
    # ------------------------------------------------------------------ #
    def _record_ledger(self, cycle: Dict[str, Any], status: str) -> None:
        from src.genML.experiments import record_entry

        idea, result = cycle["idea"], cycle["judge"]
        entry = {
            "status": status if status in MEASURED else "failed",
            "created_at": cycle["started_at"], "measured_at": _now(),
            "source": {"autopilot": cycle["id"], "mode": cycle["mode"], "branch": cycle.get("branch"),
                       "commit": cycle.get("commit"), "novelty": idea.get("novelty"),
                       "brief_items": idea.get("brief_items") or []},
            "title": idea["title"], "hypothesis": idea["hypothesis"],
            "predicted_gain": f"{idea['predicted_gain']:+.5f}", "predicted_gain_value": idea["predicted_gain"],
            "changes": cycle.get("changes"), "patch": cycle.get("patch"), "fingerprint": cycle.get("fingerprint"),
            "built_on": cycle.get("built_on"), "job_id": cycle.get("job_id"), "run_id": cycle.get("run_id"),
            "baseline": {"experiment_id": result.get("best_experiment_id"), "score": result.get("best_score")},
        }
        cited = self._cited_probes(cycle)
        if cited:
            entry["probes"] = [{k: r.get(k) for k in ("id", "label", "gain", "noise", "gain_vs_bar")} for r in cited]
            entry["probe_gain"] = cited[0].get("gain")
        if status in MEASURED:
            entry.update(candidate_experiment_id=result["candidate_experiment_id"], score=result["candidate_score"],
                         measured_gain=result["gain"], threshold=result["threshold"], folds_won=result["folds_won"])
        else:
            entry["error"] = "; ".join(result.get("errors", []))
        cycle["ledger_id"] = record_entry(self.problem, entry, self.outputs_root)["id"]

    def _journal(self, cycle: Dict[str, Any]) -> None:
        idea = cycle.get("idea") or {}
        result = cycle.get("judge") or {}
        lines = [f"## {cycle['id']} [{cycle['status']}] {idea.get('title') or (cycle.get('plan') or {}).get('reason', '')}",
                 f"- mode: {cycle['mode']}; kind: {idea.get('kind')}; novelty: {idea.get('novelty')}; "
                 f"predicted {idea.get('predicted_gain')}"
                 + (f"; builds on brief items {', '.join(idea['brief_items'])}" if idea.get("brief_items") else "")]
        if result.get("gain") is not None:
            lines.append(f"- measured {result['gain']:+.6f} (bar {result['threshold']:.6f}, folds {result['folds_won']}),"
                         f" score {result['candidate_score']:.6f}; models: {result.get('models')}")
        for record in self._cited_probes(cycle)[:3]:
            lines.append(f"- probe {record.get('id')} ({record.get('label')}): gain {record.get('gain') or 0:+.5f}, "
                         f"noise {record.get('noise') or 0:.5f}")
        if cycle.get("changes"):
            lines.append(f"- settings: {cycle['changes']}")
        if cycle.get("commit") and idea.get("kind") == "code":
            lines.append(f"- code: {cycle.get('branch')} @ {cycle['commit'][:10]} ({len(cycle.get('files') or [])} files)")
        if cycle.get("error"):
            lines.append(f"- error: {cycle['error'][:600]}")
        if cycle.get("warnings"):
            lines.append(f"- warnings: {'; '.join(cycle['warnings'])[:600]}")
        journal_text = (cycle.get("plan") or {}).get("journal")
        if journal_text:
            lines.append(f"- notes: {journal_text.strip()[:2000]}")
        with open(self.home / JOURNAL_FILE, "a") as handle:
            handle.write("\n".join(lines) + "\n\n")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def load_state(problem: str, outputs_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    from src.genML.run_context import OUTPUTS_ROOT

    path = Path(outputs_root or OUTPUTS_ROOT) / problem / "autopilot" / STATE_FILE
    return json.loads(path.read_text()) if path.exists() else None


def review_cycle(problem: str, cycle_id: Optional[str] = None, outputs_root: Optional[Path] = None,
                 backends: Optional[List[str]] = None) -> Dict[str, Any]:
    """A second opinion on one cycle (default: the newest with an idea), for a person (--second-opinion).

    Saved next to the cycle's other files; changes nothing else."""
    from src.genML.run_context import OUTPUTS_ROOT

    root = Path(outputs_root or OUTPUTS_ROOT)
    state = load_state(problem, root)
    with_idea = [c for c in (state or {}).get("cycles", []) if c.get("idea")]
    if cycle_id:
        cycle = next((c for c in (state or {}).get("cycles", []) if c["id"] == cycle_id), None)
    else:
        cycle = with_idea[-1] if with_idea else None
    if cycle is None or not cycle.get("idea"):
        raise ValueError(f"No autopilot cycle with an idea for '{problem}'" + (f" ({cycle_id})" if cycle_id else ""))
    diff = ""
    if cycle["idea"].get("kind") == "code" and cycle.get("commit"):
        diff = ws.git(Path(state["repo"]), "diff", cycle["start_commit"], cycle["commit"], check=False)
    material = {
        "cycle": {k: cycle.get(k) for k in ("id", "mode", "status", "error", "warnings")},
        "plan": cycle.get("plan"),
        **second_opinion.gain_material(cycle["idea"], cycle.get("judge") or {}, build=cycle.get("build"), diff=diff,
                                       changes=cycle.get("changes")),
        "earlier_second_opinions": [{k: o.get(k) for k in ("label", "verdict", "summary")}
                                    for o in cycle.get("second_opinions") or []],
    }
    opinion = second_opinion.ask_second_opinion(problem, second_opinion.CYCLE_QUESTION, material,
                                                outputs_root=root, backends=backends)
    folder = root / problem / "autopilot" / "cycles" / cycle["id"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"second_opinion_manual_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(opinion, indent=2))
    return {**opinion, "cycle": cycle["id"], "saved_to": str(path)}


def request_stop(problem: str, outputs_root: Optional[Path] = None) -> Path:
    from src.genML.run_context import OUTPUTS_ROOT

    path = Path(outputs_root or OUTPUTS_ROOT) / problem / "autopilot" / STOP_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_now())
    return path


def queue_directive(problem: str, text: str, outputs_root: Optional[Path] = None) -> Path:
    """Add a directive to the inbox; cycles take them oldest first, one per cycle."""
    from src.genML.run_context import OUTPUTS_ROOT

    inbox = Path(outputs_root or OUTPUTS_ROOT) / problem / "autopilot" / INBOX
    inbox.mkdir(parents=True, exist_ok=True)
    number = len(list(inbox.glob("*.md"))) + len(list((inbox / "done").glob("*.md")))
    path = inbox / f"{datetime.now():%Y%m%d_%H%M%S}_{number + 1:03d}.md"
    path.write_text(text.strip() + "\n")
    return path


def clear_stop(problem: str, outputs_root: Optional[Path] = None) -> None:
    from src.genML.run_context import OUTPUTS_ROOT

    path = Path(outputs_root or OUTPUTS_ROOT) / problem / "autopilot" / STOP_FILE
    if path.exists():
        path.unlink()


def format_status(state: Optional[Dict[str, Any]], outputs_root: Optional[Path] = None) -> str:
    if not state:
        return "No autopilot state yet (start one with --autopilot PROBLEM)."
    from src.genML.run_context import OUTPUTS_ROOT

    inbox = Path(outputs_root or OUTPUTS_ROOT) / state["problem"] / "autopilot" / INBOX
    queued = sorted(inbox.glob("*.md")) if inbox.exists() else []
    best, anchors = state["best"], state["anchors"]
    status = state["status"]
    if status == "running" and not _alive(state.get("pid")):
        status = "not running (the process ended without stopping cleanly; start it again to resume)"
    lines = [f"Autopilot for {state['problem']}: {status}"
             + (f" ({state['stop_reason']})" if state.get("stop_reason") else ""),
             f"  best: {best['experiment_id']} {anchors['metric']} {best['score']:.6f}"
             f" (started from {anchors['best']['score']:.6f})",
             f"  code: branch {state['main_branch']} (base {(state.get('base_commit') or '-')[:10]})"]
    if anchors.get("rows"):
        start = anchors.get("holdout_start")
        now = state.get("holdout_best", start)
        noise = max(anchors.get("sigma_run") or 0.0, anchors.get("sigma_seed") or 0.0)
        lines.append(f"  holdout ({anchors['rows']['n_holdout']} rows that never decide): started {start:.6f}, "
                     f"best now {now:.6f} ({now - start:+.6f}); run noise {noise:.6f}; "
                     f"judged on {anchors['rows']['scheme']['kind'].replace('_', ' ')}")
    if state.get("status_detail"):
        lines.append(f"  now: {state['status_detail']}")
    brief = research.load_brief(state["problem"], outputs_root)
    if brief:
        lines.append(f"  domain brief: {len(brief.get('items') or [])} items from sources before "
                     f"{brief['sources_before']} ({brief['created_at']}; --show-research {state['problem']} to see it)")
    lines.append("")
    lines.append(f"  {'cycle':6} {'mode':8} {'kind':8} {'status':11} {'predicted':>10} {'measured':>10} {'bar':>9}  idea")
    for c in state["cycles"]:
        idea, result = c.get("idea") or {}, c.get("judge") or {}
        predicted = f"{idea['predicted_gain']:+.5f}" if isinstance(idea.get("predicted_gain"), (int, float)) else "-"
        measured = f"{result['gain']:+.5f}" if result.get("gain") is not None else "-"
        bar = f"{result['threshold']:.5f}" if result.get("threshold") is not None else "-"
        status = c["status"] if c.get("step") == "done" else f"@{c['step']}"
        title = idea.get("title") or (c.get("plan") or {}).get("reason") or ""
        lines.append(f"  {c['id']:6} {c['mode']:8} {idea.get('kind') or '-':8} {status:11} {predicted:>10} "
                     f"{measured:>10} {bar:>9}  {title[:70]}")
        if c.get("error") and c.get("step") == "done":
            lines.append(f"  {'':6} error: {c['error'][:160]}")
    if queued:
        lines += ["", f"  Queued directives ({len(queued)}, one per cycle):"]
        lines += [f"    {path.name}: {path.read_text().strip()[:100]}" for path in queued]
    accepted = [c for c in state["cycles"] if c.get("status") == "accepted" and (c.get("idea") or {}).get("kind") == "code"]
    if accepted:
        lines += ["", f"  Accepted code is on {state['main_branch']}. To see it:",
                  f"    git diff {(state.get('base_commit') or 'HEAD')[:10]} {state['main_branch']} --stat"]
    return "\n".join(lines)


def make_config(problem: str, **overrides: Any) -> AutopilotConfig:
    agent = AgentSettings(**{k: overrides.pop(k) for k in list(overrides) if k in AgentSettings.__dataclass_fields__})
    return AutopilotConfig(problem=problem, agent=agent, **{k: v for k, v in overrides.items() if v is not None})


__all__ = ["Autopilot", "AutopilotConfig", "JobRunner", "StepFailed", "check_patch_structure", "clear_stop",
           "queue_directive", "review_cycle",
           "format_status", "load_state", "make_config", "request_stop", "validate_build", "validate_explore",
           "validate_plan"]
