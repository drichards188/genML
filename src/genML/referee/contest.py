"""Contests: a real dataset split into rows a contestant sees and hidden rows the referee scores.

Layout of ``outputs/referee/<contest>/``::

    public/train.parquet     rows a contestant trains on, with the target
    public/hidden.parquet    rows it predicts, target removed (same order as the private labels)
    public/contest.json      what a contestant may know: target, id and time columns, task, metric
    private/labels.npy       the hidden rows' labels (never visible inside the sandbox)
    private/rows.npy         per hidden row: 1 decides verdicts, 2 holdout (reported, never decides)
    private/contest.json     how the contest was made and the hashes of the public files
    entries/<entry>/         one folder per contestant run (arena.py)

Time-ordered data (a time column): the rows are cut into equal blocks in time order and the latest
block is hidden, so every contestant predicts forward in time, as a deployed model would. Otherwise a
stratified share of rows is hidden at random. Hidden rows are halved at random into deciding rows and
holdout rows; the holdout score shows whether a verdict holds on rows nobody chose anything on.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

DECIDE, HOLDOUT = 1, 2
ROWS_SEED = 20261118
HIDDEN_FRACTION = 0.2          # random splits: share of rows hidden
DEFAULT_BLOCKS = 6             # time splits: the latest of this many blocks is hidden


def referee_root(outputs_root: Optional[Path] = None) -> Path:
    from src.genML.pipeline import config

    return Path(outputs_root or config.OUTPUTS_DIR) / "referee"


def contest_dir(name: str, outputs_root: Optional[Path] = None) -> Path:
    if not name or "/" in name or name.startswith("."):
        raise ValueError(f"bad contest name {name!r}")
    return referee_root(outputs_root) / name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task(y: pd.Series) -> str:
    values = y.dropna().unique()
    if len(values) == 2:
        return "binary"
    if pd.api.types.is_float_dtype(y) and not np.allclose(values, np.round(values)):
        return "regression"
    return "multiclass" if len(values) <= 50 else "regression"


def _hidden_rows(frame: pd.DataFrame, y: np.ndarray, task: str, time_column: Optional[str],
                 blocks: int, rng: np.random.Generator) -> tuple:
    """(order, hidden mask over ``order``, description). ``order`` sorts the rows by time when there is one."""
    if time_column:
        order = np.argsort(frame[time_column].to_numpy(), kind="stable")
        cut = int(round(len(order) * (blocks - 1) / blocks))
        hidden = np.zeros(len(order), dtype=bool)
        hidden[cut:] = True
        return order, hidden, {"kind": "time_forward", "blocks": blocks, "time_column": time_column,
                               "why": "the latest block is hidden, so contestants predict forward in time"}
    order = np.arange(len(frame))
    hidden = np.zeros(len(order), dtype=bool)
    groups = [np.where(y == value)[0] for value in np.unique(y)] if task != "regression" else [order]
    for rows in groups:
        hidden[rng.choice(rows, size=int(round(len(rows) * HIDDEN_FRACTION)), replace=False)] = True
    return order, hidden, {"kind": "random", "fraction": HIDDEN_FRACTION}


def create_contest(frame: pd.DataFrame, name: str, *, target: str, metric: str,
                   id_column: Optional[str] = None, time_column: Optional[str] = None,
                   task: Optional[str] = None, sample: Optional[int] = None, blocks: int = DEFAULT_BLOCKS,
                   source: Optional[Dict[str, Any]] = None, outputs_root: Optional[Path] = None,
                   overwrite: bool = False) -> Path:
    """Write a contest folder from a training table and return it."""
    from src.genML.pipeline.metrics import get_metric

    get_metric(metric)                                    # refuse unknown metrics now, not at scoring time
    for column in [target, id_column, time_column]:
        if column and column not in frame.columns:
            raise ValueError(f"column {column!r} is not in the table")
    folder = contest_dir(name, outputs_root)
    if folder.exists() and not overwrite:
        raise FileExistsError(f"contest {name!r} exists ({folder})")
    rng = np.random.default_rng(ROWS_SEED)
    frame = frame[frame[target].notna()]
    if sample and sample < len(frame):
        frame = frame.iloc[np.sort(rng.choice(len(frame), size=sample, replace=False))]
    frame = frame.reset_index(drop=True)
    y_all = frame[target].to_numpy()
    task = task or _task(frame[target])
    order, hidden, scheme = _hidden_rows(frame, y_all, task, time_column, blocks, rng)
    train = frame.iloc[order[~hidden]].reset_index(drop=True)
    held = frame.iloc[order[hidden]].reset_index(drop=True)
    codes = np.full(len(held), HOLDOUT, dtype=np.int8)
    codes[rng.permutation(len(held))[: len(held) // 2]] = DECIDE

    public, private = folder / "public", folder / "private"
    for path in (public, private):
        path.mkdir(parents=True, exist_ok=True)
    train.to_parquet(public / "train.parquet", index=False)
    held.drop(columns=[target]).to_parquet(public / "hidden.parquet", index=False)
    np.save(private / "labels.npy", held[target].to_numpy())
    np.save(private / "rows.npy", codes)
    facts = {"name": name, "target": target, "id_column": id_column, "time_column": time_column,
             "task": task, "metric": metric, "train_rows": len(train), "hidden_rows": len(held),
             "split": scheme["kind"]}
    if task == "binary":
        facts["classes"] = sorted(pd.unique(train[target]).tolist())
    (public / "contest.json").write_text(json.dumps(facts, indent=2, default=str))
    record = {**facts, "scheme": scheme, "sample": sample, "source": source or {},
              "created_at": datetime.now().isoformat(timespec="seconds"),
              "decide_rows": int((codes == DECIDE).sum()), "holdout_rows": int((codes == HOLDOUT).sum()),
              "hashes": {p.name: sha256(p) for p in sorted(public.iterdir())}}
    (private / "contest.json").write_text(json.dumps(record, indent=2, default=str))
    return folder


def contest_from_problem(problem: str, name: str, sample: Optional[int] = None,
                         outputs_root: Optional[Path] = None, overwrite: bool = False) -> Path:
    """A contest from a declared problem's training table (several tables joined as genML does)."""
    from src.genML.pipeline import config
    from src.genML.pipeline.problem import load_problem, locate_data_dir

    spec = load_problem(problem)
    data_dir = locate_data_dir(spec)
    if spec.tables:
        from src.genML.pipeline.tables import assemble

        cache = Path(outputs_root or config.OUTPUTS_DIR) / problem / "tables_cache"
        frame, _, _ = assemble(spec.tables, data_dir, cache)
    else:
        frame = pd.read_csv(data_dir / "train.csv")
    if not spec.target or not spec.metric:
        raise ValueError(f"problem {problem!r} must declare its target and metric")
    cv = spec.cv or {}
    time_column = cv.get("time_column") if cv.get("method") == "time_blocks" else None
    return create_contest(frame, name, target=spec.target, metric=spec.metric, id_column=spec.id_column,
                          time_column=time_column, task=spec.task, sample=sample,
                          blocks=int(cv.get("n_splits") or DEFAULT_BLOCKS),
                          source={"problem": problem, "data_dir": str(data_dir)},
                          outputs_root=outputs_root, overwrite=overwrite)


def load_contest(name: str, outputs_root: Optional[Path] = None) -> Dict[str, Any]:
    """The contest's private record, after checking that its public files are unchanged."""
    folder = contest_dir(name, outputs_root)
    record = json.loads((folder / "private" / "contest.json").read_text())
    for file_name, digest in record["hashes"].items():
        if sha256(folder / "public" / file_name) != digest:
            raise ValueError(f"contest {name!r}: public/{file_name} changed since the contest was made")
    record["folder"] = str(folder)
    return record


def labels_and_rows(name: str, outputs_root: Optional[Path] = None) -> tuple:
    private = contest_dir(name, outputs_root) / "private"
    return np.load(private / "labels.npy", allow_pickle=True), np.load(private / "rows.npy")
