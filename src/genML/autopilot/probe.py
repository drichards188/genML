"""Cheap probes: test an idea's effect in a minute or two before a full run is spent on it.

A plan or explore session calls this from its own scripts::

    from src.genML.autopilot import probe

    p = probe.load()                                   # the cycle's probe context (written by the loop)
    t = p.raw([p.time_column])[p.time_column].to_numpy()
    p.compare("recency weights, 60 days",
              weights=lambda rows: np.exp(-(t[rows].max() - t[rows]) / (60 * 86400)))
    p.compare("amount per card, past only", add={"amt_per_card": values})

It fits one LightGBM with fixed settings on the best run's own feature matrix (with its nested
target encodings) and scores the base and the variant on the same rows, with two seeds. The rows
are chosen by trusted code (``probe_rows``): a probe never scores a row the judge holds back, and
for time splits it does not use the judge's block at all (it trains on the blocks before the
second-to-last and scores that block), so that many probes do not wear out the judge's rows (reusing
one holdout for many choices overfits it). Every call is logged to the worktree's ``.autopilot/probes.jsonl``; the
loop copies the log to ``outputs/<problem>/autopilot/probes.jsonl`` and shows it to later cycles.

A probe is a screen, not a verdict: one untuned model on a different block. Its numbers are
advisory (a session could write the log itself); the judge still measures the full run.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

CONTEXT_FILE = "probe_context.json"
ROWS_FILE = "probe_rows.npy"        # per training row: 1 the probe trains on it, 2 it scores it, 0 never used
LOG_FILE = "probes.jsonl"
CACHE_DIR = "probe_cache"
CONTEXT_ENV = "GENML_PROBE_CONTEXT"
PROBE_TRAIN, PROBE_EVAL = 1, 2
PROBE_SEED = 20261117
EVAL_CHUNKS = 5
N_BOOT = 100
DEFAULT_SEEDS = (0, 1)
# One LightGBM, fixed (no tuning, no early stopping on the scored rows): fast and the same for base and variant
DEFAULT_PARAMS = {"learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 100, "subsample": 0.8,
                  "subsample_freq": 1, "colsample_bytree": 0.5, "n_estimators": 600, "verbose": -1}


# --------------------------------------------------------------------------- #
# Trusted side: which rows probes may use, and the context file
# --------------------------------------------------------------------------- #

def probe_rows(fold_ids: np.ndarray, codes: np.ndarray, cv_method: Optional[str]) -> Tuple[np.ndarray, Dict]:
    """Per-row codes for probes (PROBE_TRAIN, PROBE_EVAL, 0) and a description.

    ``codes`` are the judge's row codes (``judge.ROWS_FILE``); a probe never scores a held-back row,
    and the codes must not tell a session which rows those are. Time splits: train on the blocks
    before the second-to-last and score the second-to-last; the judge's block (the last) is not used
    at all. Other splits: score a random half of fold 0's rows that are not held back and train on the
    other folds; the rest of fold 0 is not used. There the judge decides on most rows, so probes cannot
    avoid them (a known limit)."""
    from src.genML.autopilot.judge import HOLDOUT

    fold_ids, codes = np.asarray(fold_ids), np.asarray(codes)
    rows = np.zeros(len(fold_ids), dtype=np.int8)
    if cv_method == "time_blocks":
        last = int(fold_ids.max())
        fold = last - 1
        if fold < 1:
            raise ValueError("time-block probes need at least three blocks")
        if np.any((codes == HOLDOUT) & (fold_ids != last)):
            raise ValueError("held-back rows outside the judge's block: these are not time-block codes")
        rows[fold_ids < fold] = PROBE_TRAIN
        rows[fold_ids == fold] = PROBE_EVAL
        info = {"kind": "time_forward", "fold": fold, "train_blocks": list(range(fold)),
                "why": f"trains on blocks 0-{fold - 1} and scores block {fold}; the judge's block {last} is not "
                       "used, so probing does not wear out the judge's rows"}
    else:
        fold = 0
        candidates = np.where((fold_ids == fold) & (codes != HOLDOUT))[0]
        chosen = np.random.default_rng(PROBE_SEED).choice(candidates, size=len(candidates) // 2, replace=False)
        rows[fold_ids != fold] = PROBE_TRAIN
        rows[chosen] = PROBE_EVAL
        info = {"kind": "one_fold", "fold": fold,
                "why": "trains on the other folds and scores half of fold 0; it never scores a row the judge "
                       "holds back"}
    return rows, info


def write_context(folder: Path, *, problem: str, outputs_root: Path, run_id: str, anchors: Mapping[str, Any],
                  judge_bar: Optional[Mapping[str, Any]] = None) -> Path:
    """Write ``probe_context.json`` and ``probe_rows.npy`` into ``folder`` (a worktree's ``.autopilot``)."""
    folder = Path(folder)
    problem_dir = Path(outputs_root) / problem
    features = problem_dir / run_id / "features"
    for name in ("X_train.npy", "feature_names.txt"):
        if not (features / name).exists():
            raise FileNotFoundError(f"the best run {run_id} has no {name} (was it pruned?)")
    fold_ids = np.load(problem_dir / f"folds_{anchors['fold_signature']}.npy")
    codes = np.load(anchors["rows"]["file"])
    rows, info = probe_rows(fold_ids, codes, anchors.get("cv_method"))
    resolved = json.loads((problem_dir / run_id / "reports" / "resolved_spec.json").read_text())
    time_column = ((resolved.get("feature_engineering") or {}).get("time_column")
                   or (resolved.get("cv") or {}).get("time_column"))
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / ROWS_FILE, rows)
    context = {
        "problem": problem, "outputs_root": str(outputs_root), "run_id": run_id,
        "features_dir": str(features), "data_dir": str(problem_dir / run_id / "data"),
        "fold_signature": anchors["fold_signature"], "task": anchors["task"], "metric": anchors["metric"],
        "target": anchors.get("target"), "id_column": resolved.get("id_column"), "time_column": time_column,
        "eval": info, "n_train": int((rows == PROBE_TRAIN).sum()), "n_eval": int((rows == PROBE_EVAL).sum()),
        "judge_bar": dict(judge_bar) if judge_bar else None,
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = folder / CONTEXT_FILE
    path.write_text(json.dumps(context, indent=2))
    return path


def read_log(path: Path) -> List[Dict[str, Any]]:
    """The probe records in a log, skipping lines that are not JSON objects."""
    records = []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


# --------------------------------------------------------------------------- #
# Session side
# --------------------------------------------------------------------------- #

def _context_path(path: Optional[Union[str, Path]] = None) -> Path:
    if path is not None:
        return Path(path)
    if os.environ.get(CONTEXT_ENV):
        return Path(os.environ[CONTEXT_ENV])
    here = Path.cwd()
    for folder in (here, *here.parents):
        candidate = folder / ".autopilot" / CONTEXT_FILE
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no {CONTEXT_FILE} found (the loop writes it into the worktree's .autopilot folder)")


def load(context: Optional[Union[str, Path]] = None) -> "Probe":
    """The probe for this cycle (reads ``.autopilot/probe_context.json``)."""
    return Probe(_context_path(context))


class Probe:
    def __init__(self, context_path: Path):
        from src.genML.pipeline.metrics import get_metric
        from src.genML.pipeline.oof import ExperimentStore, FoldFeatures

        self.context_path = Path(context_path)
        self.folder = self.context_path.parent
        self.context = json.loads(self.context_path.read_text())
        features = Path(self.context["features_dir"])
        self.X = np.load(features / "X_train.npy", mmap_mode="r")
        self.feature_names = [n for n in (features / "feature_names.txt").read_text().splitlines() if n.strip()]
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError(f"{len(self.feature_names)} feature names for {self.X.shape[1]} columns")
        self.fold_features = FoldFeatures.load(features)
        rows = np.load(self.folder / ROWS_FILE)
        if len(rows) != self.X.shape[0]:
            raise ValueError("the probe rows do not match the feature matrix")
        self.train_rows = np.where(rows == PROBE_TRAIN)[0]
        self.eval_rows = np.where(rows == PROBE_EVAL)[0]
        self.fold = int(self.context["eval"]["fold"])
        store = ExperimentStore(Path(self.context["outputs_root"]) / self.context["problem"])
        self.y, self.transform = store.load_targets(self.context["fold_signature"])
        self.task = self.context["task"]
        self.metric = get_metric(self.context["metric"])
        self.n_classes = 0 if self.task == "regression" else int(np.max(self.y)) + 1
        self.time_column = self.context.get("time_column")
        self.judge_bar = (self.context.get("judge_bar") or {}).get("threshold")
        self._matrices: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}
        self._base: Dict[str, List[np.ndarray]] = {}

    # ---- inputs ----------------------------------------------------------
    def raw(self, columns: Sequence[str]) -> "Any":
        """Raw training columns (the loaded tables, before feature engineering), row-aligned with the
        feature matrix. The target is not a probe input."""
        import pandas as pd

        from src.genML.pipeline.dataset import _missing_text_as_nan, load_stage_table

        columns = [str(c) for c in columns]
        if self.context.get("target") in columns:
            raise ValueError("the target is not a probe input; build features from the other columns")
        data_dir = Path(self.context["data_dir"])
        parquet = data_dir / "train_data.parquet"
        if parquet.exists():
            frame = _missing_text_as_nan(pd.read_parquet(parquet, columns=columns))
        else:
            frame = load_stage_table("train", data_dir)[columns]
        if len(frame) != self.X.shape[0]:
            raise ValueError("the raw table and the feature matrix have different row counts")
        return frame.reset_index(drop=True)

    def _split(self, sample: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
        train = self.train_rows
        if sample is not None:
            if not 0 < sample <= 1:
                raise ValueError("sample must be in (0, 1]")
            rng = np.random.default_rng(PROBE_SEED)
            train = np.sort(rng.choice(train, size=max(1, int(round(len(train) * sample))), replace=False))
        return train, self.eval_rows

    def _base_matrices(self, sample: Optional[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        train, evaluate = self._split(sample)
        if sample not in self._matrices:
            if self.fold_features is not None:
                X_train, X_eval, _ = self.fold_features.split(self.X, None, self.fold, train, evaluate)
            else:
                X_train, X_eval = np.asarray(self.X[train]), np.asarray(self.X[evaluate])
            self._matrices = {sample: (X_train, X_eval)}          # keep one sample's matrices in memory
        X_train, X_eval = self._matrices[sample]
        return X_train, X_eval, train

    def _columns(self, add: Optional[Mapping[str, Any]], train: np.ndarray) -> Tuple[List[str], Optional[np.ndarray],
                                                                                    Optional[np.ndarray]]:
        if not add:
            return [], None, None
        if not isinstance(add, Mapping):
            raise TypeError("add must map new feature names to one value per training row")
        names, columns = [], []
        n = self.X.shape[0]
        for name, values in add.items():
            name = str(name)
            if name in self.feature_names or name in names:
                raise ValueError(f"feature {name!r} already exists")
            values = np.asarray(values)
            if values.shape != (n,):
                raise ValueError(f"feature {name!r} needs one value per training row ({n}), got shape {values.shape}")
            if not np.issubdtype(values.dtype, np.number) and values.dtype != bool:
                raise TypeError(f"feature {name!r} must be numeric (encode categories as codes)")
            values = values.astype(np.float32)
            if np.isinf(values).any():
                raise ValueError(f"feature {name!r} has infinite values")
            names.append(name)
            columns.append(values)
        stacked = np.column_stack(columns)
        return names, stacked[train], stacked[self.eval_rows]

    def _weights(self, weights: Union[None, np.ndarray, Callable], train: np.ndarray) -> Optional[np.ndarray]:
        if weights is None:
            return None
        values = weights(train.copy()) if callable(weights) else np.asarray(weights)[train]
        values = np.asarray(values, dtype=float)
        if values.shape != (len(train),):
            raise ValueError(f"weights need one value per probe training row ({len(train)}), got {values.shape}")
        if not np.all(np.isfinite(values)) or values.min() < 0 or values.sum() <= 0:
            raise ValueError("weights must be finite, non-negative and not all zero")
        return values

    # ---- fitting ---------------------------------------------------------
    def _fit_predict(self, X_train: np.ndarray, X_eval: np.ndarray, y_train: np.ndarray,
                     weights: Optional[np.ndarray], params: Mapping[str, Any], seed: int) -> np.ndarray:
        import lightgbm as lgb

        from src.genML.pipeline.oof import predict_matrix

        kind = lgb.LGBMRegressor if self.task == "regression" else lgb.LGBMClassifier
        model = kind(**{**DEFAULT_PARAMS, "n_jobs": os.cpu_count() or 1, **dict(params), "random_state": int(seed)})
        model.fit(X_train, y_train, sample_weight=weights)
        return predict_matrix(model, X_eval, self.task, self.n_classes)

    def _base_key(self, params: Mapping[str, Any], seeds: Sequence[int], sample: Optional[float]) -> str:
        import lightgbm as lgb

        text = json.dumps({"run": self.context["run_id"], "eval": self.context["eval"], "n_train": len(self.train_rows),
                           "n_eval": len(self.eval_rows), "params": dict(params), "seeds": list(seeds),
                           "sample": sample, "lightgbm": lgb.__version__}, sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _base_predictions(self, params: Mapping[str, Any], seeds: Sequence[int],
                          sample: Optional[float]) -> List[np.ndarray]:
        key = self._base_key(params, seeds, sample)
        if key in self._base:
            return self._base[key]
        path = self.folder / CACHE_DIR / f"base_{key}.npy"
        if path.exists():
            self._base[key] = list(np.load(path))
            return self._base[key]
        X_train, X_eval, train = self._base_matrices(sample)
        preds = [self._fit_predict(X_train, X_eval, self.y[train], None, params, s) for s in seeds]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.stack(preds))
        self._base[key] = preds
        return preds

    # ---- the probe -------------------------------------------------------
    def compare(self, label: str, *, add: Optional[Mapping[str, Any]] = None, drop: Sequence[str] = (),
                weights: Union[None, np.ndarray, Callable] = None, params: Optional[Mapping[str, Any]] = None,
                variant_params: Optional[Mapping[str, Any]] = None, seeds: Sequence[int] = DEFAULT_SEEDS,
                sample: Optional[float] = None) -> Dict[str, Any]:
        """Score the base and one variant on the probe's rows and log the result.

        ``add``: new features, name -> one value per training row (all rows, in the feature matrix's
        order; build them without the target). ``drop``: base feature names to leave out. ``weights``:
        sample weights per training row, or ``weights(rows) -> values`` for the probe's training rows.
        ``params``: LightGBM settings for both fits; ``variant_params``: for the variant only.
        ``sample``: train on this share of the probe's training rows (faster, noisier)."""
        if not str(label or "").strip():
            raise ValueError("label: say what this probe tests")
        started = time.time()
        params, variant_params = dict(params or {}), dict(variant_params or {})
        seeds = [int(s) for s in seeds]
        unknown = [name for name in drop if name not in self.feature_names]
        if unknown:
            raise ValueError(f"drop names features that do not exist: {unknown[:10]}")
        if not (add or drop or weights is not None or variant_params):
            raise ValueError("nothing to compare: give add, drop, weights or variant_params")
        train, _ = self._split(sample)
        names, extra_train, extra_eval = self._columns(add, train)          # check the inputs before any fit
        w = self._weights(weights, train)
        base = self._base_predictions(params, seeds, sample)
        X_train, X_eval, train = self._base_matrices(sample)
        dropped = set(drop)
        keep = np.array([name not in dropped for name in self.feature_names])
        if drop:
            X_train, X_eval = X_train[:, keep], X_eval[:, keep]
        if extra_train is not None:
            X_train, X_eval = np.hstack([X_train, extra_train]), np.hstack([X_eval, extra_eval])
        variant = [self._fit_predict(X_train, X_eval, self.y[train], w, {**params, **variant_params}, s)
                   for s in seeds]
        record = {"label": str(label), "added": names, "dropped": list(drop), "weights": w is not None,
                  "params": params, "variant_params": variant_params, "seeds": seeds, "sample": sample,
                  "eval": self.context["eval"]["kind"], "train_rows": int(len(train)),
                  "eval_rows": int(len(self.eval_rows)), **self._measure(base, variant),
                  "seconds": round(time.time() - started, 1)}
        return self._log(record)

    def _measure(self, base: List[np.ndarray], variant: List[np.ndarray]) -> Dict[str, Any]:
        from src.genML.pipeline.diagnostics import _signed_scorer, paired_compare

        y_eval = self.y[self.eval_rows]
        score = _signed_scorer(SimpleNamespace(metric_obj=self.metric, task=self.task), y_eval, self.transform)
        every = np.arange(len(y_eval))
        mean_base, mean_variant = np.mean(base, axis=0), np.mean(variant, axis=0)
        chunks = np.random.default_rng(PROBE_SEED).permutation(len(y_eval)) % EVAL_CHUNKS
        comparison = paired_compare(score, mean_base, mean_variant, chunks, n_boot=N_BOOT, seed=PROBE_SEED)
        per_seed = [score(every, v) - score(every, b) for b, v in zip(base, variant)]
        seed_sd = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None
        noise = float(np.sqrt(comparison["se_rows"] ** 2 + ((seed_sd or 0.0) ** 2) / len(per_seed)))
        gain = comparison["diff"]
        bar = self.judge_bar
        return {
            "base": float(self.metric.signed(score(every, mean_base))),
            "variant": float(self.metric.signed(score(every, mean_variant))),
            "gain": float(gain), "se_rows": comparison["se_rows"], "seed_sd": seed_sd, "noise": noise,
            "z": float(gain / noise) if noise > 0 else None,
            "per_seed_gain": [float(d) for d in per_seed],
            "chunks_won": f"{comparison['folds_won']}/{comparison['n_folds']}",
            "judge_bar": bar, "gain_vs_bar": float(gain / bar) if bar else None,
        }

    def _log(self, record: Dict[str, Any]) -> Dict[str, Any]:
        path = self.folder / LOG_FILE
        record = {"id": f"P{len(read_log(path)) + 1:02d}", "at": datetime.now().isoformat(timespec="seconds"),
                  **record}
        with open(path, "a") as handle:
            handle.write(json.dumps(record, default=float) + "\n")
        ratio = f", {record['gain_vs_bar']:+.2f} x the judge's bar" if record.get("gain_vs_bar") is not None else ""
        print(f"[{record['id']}] {record['label']}: {record['base']:.5f} -> {record['variant']:.5f} "
              f"(gain {record['gain']:+.5f}, noise {record['noise']:.5f}{ratio}; {record['seconds']}s)", flush=True)
        return record
