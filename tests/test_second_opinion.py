"""The second opinion (autopilot/second_opinion.py) and the dossier it reads; every backend is faked."""

import json

import pytest

from src.genML.autopilot import second_opinion
from src.genML.llm import client as client_mod
from src.genML.llm.backends import Backend, BackendFailed, BackendResult

OK = {"verdict": "ok", "summary": "Sound.", "concerns": [], "suggestion": ""}


class _Fake(Backend):
    billing = "subscription"
    answer = OK
    fail = None
    seen = []

    def run(self, request):
        type(self).seen.append((self.name, request))
        if self.fail:
            raise BackendFailed(self.fail)
        return BackendResult(data=self.answer, model=f"{self.name}-model", billing=self.billing)


def _register(monkeypatch, **classes):
    _Fake.seen = []
    monkeypatch.setattr(client_mod, "BACKENDS", dict(classes))


def test_rules_come_from_the_autopilot_prompt():
    rules = second_opinion.rules_text()
    assert "published solutions" in rules and "public" in rules and "generated" in rules
    prompt = second_opinion.system_prompt()
    assert rules in prompt and '"reject"' in prompt and "{rules}" not in prompt


def test_claude_answers_by_default_without_trying_codex(monkeypatch, tmp_path):
    # The owner's ChatGPT plan ended on 2026-11-17: Codex is no longer asked unless ORDER names it
    codex = type("Codex", (_Fake,), {"name": "codex-cli"})
    claude = type("Claude", (_Fake,), {"name": "claude-cli"})
    _register(monkeypatch, **{"codex-cli": codex, "claude-cli": claude})
    opinion = second_opinion.ask_second_opinion("toy", second_opinion.PLAN_QUESTION, {"idea": {"title": "t"}},
                                                outputs_root=tmp_path)
    assert opinion["verdict"] == "ok" and opinion["backend"] == "claude-cli"
    assert [name for name, _ in _Fake.seen] == ["claude-cli"]
    name, request = _Fake.seen[0]
    assert request.schema is second_opinion.SCHEMA
    assert second_opinion.PLAN_QUESTION in request.prompt and '"title": "t"' in request.prompt
    assert json.loads(open(opinion["log_file"]).read())["purpose"] == "second-opinion"


def test_claude_answers_when_codex_cannot(monkeypatch, tmp_path):
    codex = type("Codex", (_Fake,), {"name": "codex-cli", "fail": "usage limit"})
    claude = type("Claude", (_Fake,), {"name": "claude-cli"})
    _register(monkeypatch, **{"codex-cli": codex, "claude-cli": claude})
    opinion = second_opinion.ask_second_opinion("toy", "q", {}, outputs_root=tmp_path,
                                                backends=["codex-cli", "claude-cli"])
    assert opinion["backend"] == "claude-cli" and [name for name, _ in _Fake.seen] == ["codex-cli", "claude-cli"]


def test_off_schema_answers_are_refused(monkeypatch, tmp_path):
    from src.genML.llm import LLMError

    sloppy = type("Sloppy", (_Fake,), {"name": "codex-cli", "answer": {"verdict": "maybe"}})
    _register(monkeypatch, **{"codex-cli": sloppy})
    with pytest.raises(LLMError):
        second_opinion.ask_second_opinion("toy", "q", {}, outputs_root=tmp_path, backends=["codex-cli"])


def test_gain_material_cuts_long_diffs():
    material = second_opinion.gain_material({"title": "t"}, {"gain": 0.01, "verdict": "suspicious"}, build=None,
                                            diff="x" * (second_opinion.MAX_DIFF_CHARS + 50), changes=["a"])
    assert material["code_diff"].endswith("(diff cut)") and material["judge"]["gain"] == 0.01


def test_dossier_reads_only_and_names_no_banned_ideas(tmp_path):
    from src.genML.autopilot.dossier import SPEC_CAPABILITIES, build_dossier
    from src.genML.flow import run_pipeline
    from tests.test_competitive import _spec, _write_problem

    data = _write_problem(tmp_path / "data")
    root = tmp_path / "out"
    spec = _spec(data, models=["XGBoost", "Logistic Regression"],
                 feature_engineering={"base": "raw", "encodings": {"target_encode": ["shop"]}})
    summary = run_pipeline(spec=spec, setup_log=False, outputs_root=root)
    assert summary["status"] == "completed", summary.get("error")
    ledger = root / "toy" / "experiments.json"
    ledger.write_text(json.dumps([{"id": "E001", "status": "running", "run_id": "nope"}]))
    before = ledger.read_bytes()

    dossier = build_dossier("toy", outputs_root=root)
    assert ledger.read_bytes() == before                                  # building it changed nothing
    assert dossier["reviewed_run"]["run_id"] == summary["run_id"]
    assert dossier["problem"]["metric"] == "roc_auc"
    facts = second_opinion.measurements(dossier)
    assert facts["best_run"]["run_id"] == summary["run_id"] and "acceptance_rule" in facts
    text = json.dumps(SPEC_CAPABILITIES).lower()
    assert "customer rebuilt" not in text and "transactiondt" not in text
