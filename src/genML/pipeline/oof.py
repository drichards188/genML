"""Fixed folds and the out-of-fold (OOF) prediction store.

Every model trained for a problem is evaluated on the same fold assignment, and
its OOF predictions (for train rows) and fold-averaged test predictions are
saved under ``outputs/<problem>/oof/<experiment_id>/``. Ensembles can then mix
experiments from any run of the same problem, as long as they share folds.

Predictions are stored on the model's own scale: class-probability matrices for
classification, and values on the training-target scale for regression (after
any log1p/logit transform). ``targets.inverse_regression`` maps them back.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #

def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(values).tobytes()).hexdigest()[:12]


def fold_signature(y: np.ndarray, cv: Dict[str, Any]) -> str:
    """Identifies a fold assignment: same target rows and same CV settings."""
    settings = json.dumps({k: cv.get(k) for k in ("method", "n_splits", "random_state", "shuffle")},
                          sort_keys=True)
    return hashlib.sha1((_array_hash(np.asarray(y)) + settings).encode()).hexdigest()[:12]


def make_folds(y: np.ndarray, cv: Dict[str, Any]) -> np.ndarray:
    """Assign each training row to a fold (array of fold ids)."""
    n_splits = int(cv.get("n_splits", 5))
    seed = int(cv.get("random_state", 42))
    shuffle = bool(cv.get("shuffle", True))
    method = cv.get("method", "kfold")
    splitter = (
        StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        if method == "stratified_kfold"
        else KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    )
    fold_ids = np.full(len(y), -1, dtype=np.int16)
    for fold, (_, valid_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
        fold_ids[valid_idx] = fold
    return fold_ids


def load_or_make_folds(y: np.ndarray, cv: Dict[str, Any], folder: Path) -> tuple[np.ndarray, str]:
    """Folds shared by every run of a problem (cached as folds_<signature>.npy)."""
    signature = fold_signature(y, cv)
    path = Path(folder) / f"folds_{signature}.npy"
    if path.exists():
        return np.load(path), signature
    fold_ids = make_folds(y, cv)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, fold_ids)
    return fold_ids, signature


# --------------------------------------------------------------------------- #
# Fitting with OOF predictions
# --------------------------------------------------------------------------- #

def predict_matrix(model, X: np.ndarray, task: str, n_classes: int) -> np.ndarray:
    """Probability matrix for classifiers, 1-D values for regressors."""
    if task == "regression":
        return np.asarray(model.predict(X), dtype=float).ravel()
    proba = np.asarray(model.predict_proba(X), dtype=float)
    if proba.ndim == 1:
        proba = np.column_stack([1 - proba, proba])
    if proba.shape[1] != n_classes:
        # A fold that missed a rare class: expand to every class
        full = np.zeros((proba.shape[0], n_classes))
        full[:, np.asarray(model.classes_, dtype=int)] = proba
        proba = full
    return proba


def _with_seed(factory: Callable, params: Optional[Dict[str, Any]], seed: int):
    model = factory(params)
    for name in ("random_state", "random_seed", "seed"):
        try:
            if name in model.get_params():
                model.set_params(**{name: seed})
                break
        except Exception:
            continue
    return model


def fit_oof(factory: Callable, params: Optional[Dict[str, Any]], X: np.ndarray, y: np.ndarray,
            X_test: np.ndarray, fold_ids: np.ndarray, task: str, n_classes: int,
            seeds: Sequence[int] = (42,), on_fold: Optional[Callable[[int, int], None]] = None):
    """Train one model per fold (and per seed); return OOF and averaged test predictions."""
    n_folds = int(fold_ids.max()) + 1
    shape = (len(y),) if task == "regression" else (len(y), n_classes)
    test_shape = (len(X_test),) if task == "regression" else (len(X_test), n_classes)
    oof = np.zeros(shape)
    test_pred = np.zeros(test_shape)
    total = n_folds * len(seeds)
    done = 0
    for seed in seeds:
        for fold in range(n_folds):
            train_idx = np.where(fold_ids != fold)[0]
            valid_idx = np.where(fold_ids == fold)[0]
            model = _with_seed(factory, params, seed)
            model.fit(X[train_idx], y[train_idx])
            oof[valid_idx] += predict_matrix(model, X[valid_idx], task, n_classes) / len(seeds)
            test_pred += predict_matrix(model, X_test, task, n_classes) / total
            del model
            done += 1
            if on_fold:
                on_fold(done, total)
    return oof, test_pred


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #

@dataclass
class Experiment:
    experiment_id: str
    path: Path
    meta: Dict[str, Any]

    @property
    def oof(self) -> np.ndarray:
        return np.load(self.path / "oof.npy")

    @property
    def test(self) -> np.ndarray:
        return np.load(self.path / "test.npy")


class ExperimentStore:
    """OOF/test predictions of every experiment for one problem."""

    def __init__(self, problem_dir: Path):
        self.root = Path(problem_dir) / "oof"

    def save(self, name: str, run_id: str, oof: np.ndarray, test: np.ndarray, meta: Dict[str, Any]) -> Experiment:
        slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
        experiment_id = f"{run_id}__{slug}"
        path = self.root / experiment_id
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "oof.npy", oof.astype(np.float32))
        np.save(path / "test.npy", test.astype(np.float32))
        meta = {
            **meta,
            "experiment_id": experiment_id,
            "name": name,
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        return Experiment(experiment_id, path, meta)

    def save_targets(self, signature: str, y: np.ndarray, transform_info: Dict[str, Any]) -> None:
        """Training target (model scale) for a fold signature, for later re-blending."""
        folder = self.root / "_targets"
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / f"y_{signature}.npy", y)
        (folder / f"transform_{signature}.json").write_text(json.dumps(transform_info, default=str))

    def load_targets(self, signature: str) -> tuple[np.ndarray, Dict[str, Any]]:
        folder = self.root / "_targets"
        return (np.load(folder / f"y_{signature}.npy"),
                json.loads((folder / f"transform_{signature}.json").read_text()))

    def list(self, fold_signature: Optional[str] = None) -> List[Experiment]:
        experiments = []
        for meta_file in sorted(self.root.glob("*/meta.json")):
            if meta_file.parent.name.startswith("_"):
                continue
            try:
                meta = json.loads(meta_file.read_text())
            except Exception as exc:
                logger.warning("Skipping unreadable %s: %s", meta_file, exc)
                continue
            if fold_signature and meta.get("fold_signature") != fold_signature:
                continue
            experiments.append(Experiment(meta["experiment_id"], meta_file.parent, meta))
        return experiments

    def get(self, experiment_id: str) -> Experiment:
        path = self.root / experiment_id
        if not (path / "meta.json").exists():
            raise KeyError(f"No experiment '{experiment_id}' in {self.root}")
        return Experiment(experiment_id, path, json.loads((path / "meta.json").read_text()))

    def remove(self, experiment_id: str) -> None:
        shutil.rmtree(self.root / experiment_id)
