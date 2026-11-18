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
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #

def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(values).tobytes()).hexdigest()[:12]


# Methods whose folds depend on a column of the data (cv.group or cv.time_column)
COLUMN_METHODS = ("group_kfold", "stratified_group_kfold", "time_blocks")
FOLD_SIGNATURE_FILE = "fold_signature.txt"


def needs_column(cv: Dict[str, Any]) -> bool:
    return (cv or {}).get("method") in COLUMN_METHODS


def cv_column(cv: Dict[str, Any]) -> Optional[str]:
    """The column a fold method needs: cv.time_column for time_blocks, else cv.group."""
    if not needs_column(cv):
        return None
    return cv.get("time_column") if cv.get("method") == "time_blocks" else cv.get("group")


def fold_signature(y: np.ndarray, cv: Dict[str, Any], groups: Optional[np.ndarray] = None) -> str:
    """Identifies a fold assignment: same target rows, same CV settings (and same groups or times)."""
    keys = ("method", "n_splits", "random_state", "shuffle")
    if needs_column(cv):
        keys += ("group", "time_column")
    settings = json.dumps({k: cv.get(k) for k in keys}, sort_keys=True)
    base = _array_hash(np.asarray(y)) + settings
    if needs_column(cv) and groups is not None:
        base += _array_hash(_group_codes(groups))
    return hashlib.sha1(base.encode()).hexdigest()[:12]


def _group_codes(values: np.ndarray) -> np.ndarray:
    """Integer codes for group values (missing values form one group)."""
    import pandas as pd

    codes, _ = pd.factorize(pd.Series(values).astype(str), sort=True)
    return codes.astype(np.int64)


def _time_blocks(times: np.ndarray, n_splits: int) -> np.ndarray:
    """Contiguous blocks in time order with about equal rows (fold 0 is the earliest)."""
    import pandas as pd

    series = pd.Series(times)
    if not pd.api.types.is_numeric_dtype(series):
        series = pd.to_datetime(series, errors="coerce").astype("int64")
    order = np.lexsort((np.arange(len(series)), series.to_numpy(dtype=float, na_value=-np.inf)))
    blocks = np.empty(len(series), dtype=np.int16)
    blocks[order] = (np.arange(len(series)) * n_splits // max(len(series), 1)).astype(np.int16)
    return blocks


def make_folds(y: np.ndarray, cv: Dict[str, Any], groups: Optional[np.ndarray] = None) -> np.ndarray:
    """Assign each training row to a fold (array of fold ids).

    ``kfold`` / ``stratified_kfold`` split rows at random. ``group_kfold`` /
    ``stratified_group_kfold`` keep every value of ``groups`` (cv.group: a customer,
    a card, a month) in one fold. ``time_blocks`` cuts the rows, ordered by ``groups``
    (cv.time_column), into contiguous blocks, so the last fold is the latest period.
    """
    n_splits = int(cv.get("n_splits", 5))
    seed = int(cv.get("random_state", 42))
    shuffle = bool(cv.get("shuffle", True))
    method = cv.get("method", "kfold")
    if method in COLUMN_METHODS and groups is None:
        raise ValueError(f"cv.method '{method}' needs the values of column '{cv_column(cv)}'")
    if method == "time_blocks":
        return _time_blocks(groups, n_splits)
    if method == "stratified_group_kfold":
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
    elif method == "group_kfold":
        splitter = (GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed) if shuffle
                    else GroupKFold(n_splits=n_splits))
    elif method == "stratified_kfold":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    fold_ids = np.full(len(y), -1, dtype=np.int16)
    split_groups = _group_codes(groups) if method in COLUMN_METHODS else None
    for fold, (_, valid_idx) in enumerate(splitter.split(np.zeros(len(y)), y, split_groups)):
        fold_ids[valid_idx] = fold
    return fold_ids


def load_or_make_folds(y: np.ndarray, cv: Dict[str, Any], folder: Path,
                       groups: Optional[np.ndarray] = None) -> tuple[np.ndarray, str]:
    """Folds shared by every run of a problem (cached as folds_<signature>.npy)."""
    signature = fold_signature(y, cv, groups)
    path = Path(folder) / f"folds_{signature}.npy"
    if path.exists():
        return np.load(path), signature
    fold_ids = make_folds(y, cv, groups)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, fold_ids)
    return fold_ids, signature


def run_folds(y: np.ndarray, cv: Dict[str, Any], problem_dir: Path, features_dir: Path,
              train_df=None) -> tuple[np.ndarray, str]:
    """The run's folds. Methods that need a column get its values from ``train_df`` (the
    feature stage), which records the signature in ``features_dir`` for later stages."""
    if not needs_column(cv):
        return load_or_make_folds(y, cv, problem_dir)
    record = Path(features_dir) / FOLD_SIGNATURE_FILE
    if train_df is not None:
        column = cv_column(cv)
        if column not in train_df.columns:
            raise ValueError(f"cv needs column '{column}', which the training data does not have")
        fold_ids, signature = load_or_make_folds(y, cv, problem_dir, train_df[column].to_numpy())
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(signature)
        return fold_ids, signature
    if not record.exists():
        raise FileNotFoundError(f"{record} is missing: the feature stage did not record this run's folds")
    signature = record.read_text().strip()
    return np.load(Path(problem_dir) / f"folds_{signature}.npy"), signature


def tuning_fold_indices(cv: Dict[str, Any], n_folds: int, count: int) -> List[int]:
    """Folds used for tuning: the latest blocks for time_blocks, else the first ones."""
    count = max(1, min(int(count), n_folds))
    if (cv or {}).get("method") == "time_blocks":
        return list(range(n_folds - count, n_folds))
    return list(range(count))


# --------------------------------------------------------------------------- #
# Per-fold feature blocks (nested target encoding)
# --------------------------------------------------------------------------- #

FOLD_BLOCKS_FILE = "fold_blocks.json"


class FoldFeatures:
    """Fold-specific replacements for some feature columns.

    The feature stage saves, for every outer fold k, the columns whose values must
    not depend on fold k's labels (nested target encodings): ``fold_train_<k>.npy``
    holds them for all training rows and ``fold_test_<k>.npy`` for the test rows.
    ``split`` returns the matrices a fold-k model trains, validates and predicts on.
    """

    def __init__(self, folder: Path, indices: List[int], columns: List[str]):
        self.folder = Path(folder)
        self.indices = np.asarray(indices, dtype=int)
        self.columns = list(columns)

    @classmethod
    def save(cls, folder: Path, indices: Sequence[int], columns: Sequence[str],
             train_blocks: Sequence[np.ndarray], test_blocks: Sequence[np.ndarray]) -> "FoldFeatures":
        folder = Path(folder)
        for fold, (train_block, test_block) in enumerate(zip(train_blocks, test_blocks)):
            np.save(folder / f"fold_train_{fold}.npy", np.asarray(train_block, dtype=np.float32))
            np.save(folder / f"fold_test_{fold}.npy", np.asarray(test_block, dtype=np.float32))
        (folder / FOLD_BLOCKS_FILE).write_text(json.dumps(
            {"indices": [int(i) for i in indices], "columns": list(columns), "n_folds": len(train_blocks)}))
        return cls(folder, list(indices), list(columns))

    @classmethod
    def load(cls, folder: Path) -> Optional["FoldFeatures"]:
        """The saved blocks, or None when the feature stage produced none."""
        path = Path(folder) / FOLD_BLOCKS_FILE
        if not path.exists():
            return None
        meta = json.loads(path.read_text())
        return cls(folder, meta["indices"], meta["columns"])

    def split(self, X: np.ndarray, X_test: Optional[np.ndarray], fold: int,
              train_idx: np.ndarray, valid_idx: np.ndarray):
        X_train, X_valid = X[train_idx], X[valid_idx]
        block = np.load(self.folder / f"fold_train_{fold}.npy", mmap_mode="r")
        X_train[:, self.indices] = block[train_idx]
        X_valid[:, self.indices] = block[valid_idx]
        X_fold_test = None
        if X_test is not None:
            X_fold_test = np.array(X_test, copy=True)
            X_fold_test[:, self.indices] = np.load(self.folder / f"fold_test_{fold}.npy")
        return X_train, X_valid, X_fold_test


def fold_split(X: np.ndarray, X_test: Optional[np.ndarray], fold_ids: np.ndarray, fold: int,
               fold_features: Optional[FoldFeatures] = None):
    """(train_idx, valid_idx, X_train, X_valid, X_test) for one fold, using fold blocks if any."""
    train_idx = np.where(fold_ids != fold)[0]
    valid_idx = np.where(fold_ids == fold)[0]
    if fold_features is not None:
        X_train, X_valid, X_fold_test = fold_features.split(X, X_test, fold, train_idx, valid_idx)
    else:
        X_train, X_valid, X_fold_test = X[train_idx], X[valid_idx], X_test
    return train_idx, valid_idx, X_train, X_valid, X_fold_test


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
            seeds: Sequence[int] = (42,), on_fold: Optional[Callable[[int, int], None]] = None,
            fold_features: Optional[FoldFeatures] = None):
    """Train one model per fold (and per seed); return OOF and averaged test predictions.

    With ``fold_features`` each fold-k model sees fold k's nested encodings (train,
    validation and test rows alike), so fold k's labels never reach its features.
    """
    n_folds = int(fold_ids.max()) + 1
    shape = (len(y),) if task == "regression" else (len(y), n_classes)
    test_shape = (len(X_test),) if task == "regression" else (len(X_test), n_classes)
    oof = np.zeros(shape)
    test_pred = np.zeros(test_shape)
    total = n_folds * len(seeds)
    done = 0
    for seed in seeds:
        for fold in range(n_folds):
            train_idx, valid_idx, X_train, X_valid, X_fold_test = fold_split(
                X, X_test, fold_ids, fold, fold_features)
            model = _with_seed(factory, params, seed)
            model.fit(X_train, y[train_idx])
            oof[valid_idx] += predict_matrix(model, X_valid, task, n_classes) / len(seeds)
            test_pred += predict_matrix(model, X_fold_test, task, n_classes) / total
            del model, X_train, X_valid, X_fold_test
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
