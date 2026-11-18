"""Problems whose data spans several tables: join them, roll them up, cache one modelling table.

Declared in problem.yaml under ``tables``. Exactly one table has no ``parent``: the
main table, one row per prediction, with ``train`` and ``test`` files. Every other
table hangs off a parent through a key column ``key`` (same name in both;
``on`` would be read by YAML as a boolean):

.. code-block:: yaml

    tables:
      application: {train: application_train.csv, test: application_test.csv}
      bureau: {file: bureau.csv, parent: application, key: SK_ID_CURR}           # one-to-many
      bureau_balance: {file: bureau_balance.csv, parent: bureau, key: SK_ID_BUREAU,
                       order_by: MONTHS_BALANCE}
      identity: {train: train_identity.csv, test: test_identity.csv, parent: application,
                 key: SK_ID_CURR, relation: one}                                   # one-to-one join

* ``relation: one`` joins the table's columns onto its parent (left join).
* ``relation: many`` (default) rolls the table up per key and joins the result:
  ``count`` of rows, numeric ``mean/std/min/max`` (``aggregate.numeric`` to change),
  categorical ``nunique``; with ``order_by`` also ``last`` and ``trend`` (last minus
  first) and ``aggregate.last_n: [3]`` (mean of the last n rows). Children of a table
  are rolled up onto it first, so chains (bureau_balance -> bureau -> applicant) work.
* A table's rows may sit in one ``file`` (rows for train and test entities alike) or
  in separate ``train``/``test`` files. ``columns`` keeps only some columns, ``drop``
  removes some. Test files whose column names differ from train only in punctuation
  (``id-01`` vs ``id_01``) are aligned to the train names.

Everything runs in Polars (lazy, streaming where possible). Each CSV is converted
once to Parquet with 64-bit floats narrowed to 32 bits, and the assembled train/test
tables are cached by a hash of the table settings and the source files' sizes and
times, so later runs load them in seconds. Nothing here reads the target.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

TABLES_VERSION = 1
TABLE_KEYS = {"train", "test", "file", "parent", "key", "relation", "aggregate", "order_by", "columns", "drop",
              "prefix"}
AGGREGATE_KEYS = {"numeric", "categorical", "count", "last_n", "partitions"}
NUMERIC_STATS = ("mean", "std", "min", "max", "sum", "median", "first", "last", "trend", "nunique")
CATEGORICAL_STATS = ("nunique", "first", "last", "mode")
ORDERED_STATS = {"first", "last", "trend"}
DEFAULT_NUMERIC = ("mean", "std", "min", "max")
DEFAULT_CATEGORICAL = ("nunique",)


# --------------------------------------------------------------------------- #
# Checking a tables block
# --------------------------------------------------------------------------- #

def main_table(tables: Dict[str, Any]) -> str:
    roots = [name for name, table in tables.items() if not (table or {}).get("parent")]
    if len(roots) != 1:
        raise ValueError(f"tables needs exactly one main table (without 'parent'), found {roots}")
    return roots[0]


def check_tables(tables: Any) -> List[str]:
    """Problems with a ``tables`` block (empty list = acceptable)."""
    if not isinstance(tables, dict) or not tables:
        return ["tables must be a non-empty mapping of table name to settings"]
    errors = []
    for name, table in tables.items():
        if not isinstance(table, dict):
            errors.append(f"tables.{name} must be a mapping")
            continue
        unknown = set(table) - TABLE_KEYS
        if unknown:
            errors.append(f"tables.{name}: unknown setting(s) {sorted(unknown)} (known: {sorted(TABLE_KEYS)})")
        has_split = "train" in table or "test" in table
        if has_split and not ("train" in table and "test" in table):
            errors.append(f"tables.{name} needs both train and test files")
        if has_split == ("file" in table):
            errors.append(f"tables.{name} needs either 'file' or 'train' and 'test'")
        parent = table.get("parent")
        if parent is None:
            if "file" in table:
                errors.append(f"tables.{name} is the main table: it needs separate train and test files")
            continue
        if parent not in tables:
            errors.append(f"tables.{name}.parent '{parent}' is not a declared table")
        if not table.get("key"):
            errors.append(f"tables.{name} needs 'key' (the key column shared with its parent)")
        if table.get("relation", "many") not in ("one", "many"):
            errors.append(f"tables.{name}.relation must be 'one' or 'many'")
        aggregate = table.get("aggregate") or {}
        if not isinstance(aggregate, dict):
            errors.append(f"tables.{name}.aggregate must be a mapping")
            continue
        unknown = set(aggregate) - AGGREGATE_KEYS
        if unknown:
            errors.append(f"tables.{name}.aggregate: unknown key(s) {sorted(unknown)}")
        for key, allowed in (("numeric", NUMERIC_STATS), ("categorical", CATEGORICAL_STATS)):
            stats = aggregate.get(key)
            if stats is not None and (not isinstance(stats, list) or set(stats) - set(allowed)):
                errors.append(f"tables.{name}.aggregate.{key} must be a list from {list(allowed)}")
            elif stats and set(stats) & ORDERED_STATS and not table.get("order_by"):
                errors.append(f"tables.{name}: {sorted(set(stats) & ORDERED_STATS)} need order_by")
        partitions = aggregate.get("partitions")
        if partitions is not None and not (isinstance(partitions, int) and 1 <= partitions <= 256):
            errors.append(f"tables.{name}.aggregate.partitions must be an integer from 1 to 256")
        last_n = aggregate.get("last_n")
        if last_n is not None:
            if not (isinstance(last_n, list) and all(isinstance(n, int) and n > 0 for n in last_n)):
                errors.append(f"tables.{name}.aggregate.last_n must be a list of positive integers")
            elif not table.get("order_by"):
                errors.append(f"tables.{name}.aggregate.last_n needs order_by")
    if not errors:
        try:
            main = main_table(tables)
        except ValueError as exc:
            return [str(exc)]
        for name in tables:  # every table must reach the main table without cycles
            seen, current = set(), name
            while current != main:
                if current in seen:
                    errors.append(f"tables: parent chain of '{name}' has a cycle")
                    break
                seen.add(current)
                current = tables[current]["parent"]
    return errors


def main_files(tables: Dict[str, Any]) -> Tuple[str, str]:
    table = tables[main_table(tables)]
    return table["train"], table["test"]


# --------------------------------------------------------------------------- #
# Reading (CSV -> Parquet cache)
# --------------------------------------------------------------------------- #

def _canonical(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower()


def _source_stamp(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime": int(stat.st_mtime)}


def _parquet(path: Path, raw_dir: Path) -> Path:
    """The file as Parquet with 64-bit floats as 32-bit (converted once, then reused)."""
    import polars as pl

    if path.suffix == ".parquet":
        return path
    stamp = _source_stamp(path)
    out = raw_dir / f"{path.stem}.parquet"
    meta = raw_dir / f"{path.stem}.json"
    if out.exists() and meta.exists() and json.loads(meta.read_text()) == stamp:
        return out
    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    logger.info("Converting %s to Parquet", path.name)
    for infer in (10_000, None):  # None: scan the whole file when the sample guessed a type wrong
        try:
            frame = pl.scan_csv(path, infer_schema_length=infer, null_values=["", "NA", "NaN", "nan"])
            frame = frame.with_columns(pl.col(pl.Float64).cast(pl.Float32))
            frame.sink_parquet(tmp)
            break
        except pl.exceptions.ComputeError:
            if infer is None:
                raise
            logger.info("Type inference on a sample failed for %s; reading the whole file", path.name)
    tmp.replace(out)
    meta.write_text(json.dumps(stamp))
    return out


def _scan(table: Dict[str, Any], data_dir: Path, raw_dir: Path, part: Optional[str] = None):
    """Lazy frame of a table (one part, or the whole table with train and test rows together)."""
    import polars as pl

    def read(name: str):
        return pl.scan_parquet(_parquet(data_dir / name, raw_dir))

    if "file" in table:
        frame = read(table["file"])
    elif part is not None:
        frame = read(table[part])
    else:
        train, test = read(table["train"]), read(table["test"])
        test = _align(test, train)
        frame = pl.concat([train, test], how="diagonal_relaxed")
    if part == "test" and "file" not in table:
        frame = _align(frame, read(table["train"]))
    keep = table.get("columns")
    keys = [table["key"]] if table.get("key") else []
    if keep:
        names = frame.collect_schema().names()
        frame = frame.select([c for c in dict.fromkeys(keys + list(keep)) if c in names])
    if table.get("drop"):
        frame = frame.drop([c for c in table["drop"] if c not in keys], strict=False)
    return frame


def _align(test, train):
    """Rename test columns that match a train column except for punctuation or case."""
    train_names = train.collect_schema().names()
    canonical = {_canonical(c): c for c in train_names}
    renames = {}
    for column in test.collect_schema().names():
        target = canonical.get(_canonical(column))
        if target and target != column and target not in test.collect_schema().names():
            renames[column] = target
    return test.rename(renames) if renames else test


# --------------------------------------------------------------------------- #
# Rolling up and joining
# --------------------------------------------------------------------------- #

def _children(tables: Dict[str, Any], name: str) -> List[str]:
    return [child for child, table in tables.items() if table.get("parent") == name]


def _key_columns(tables: Dict[str, Any]) -> set:
    return {table["key"] for table in tables.values() if table.get("key")}


def _rolled_up(tables: Dict[str, Any], name: str, data_dir: Path, raw_dir: Path, report: Dict[str, Any]):
    """Table ``name`` (all rows) with its own children already joined or rolled up onto it."""
    frame = _scan(tables[name], data_dir, raw_dir)
    return _attach_children(tables, name, frame, data_dir, raw_dir, report)


def _attach_children(tables, name, frame, data_dir, raw_dir, report):
    import polars as pl

    for child in _children(tables, name):
        table = tables[child]
        child_frame = _rolled_up(tables, child, data_dir, raw_dir, report)
        if table.get("relation", "many") == "one":
            frame = _join_one(frame, child_frame, child, table, report)
        else:
            frame = frame.join(_aggregate(child_frame, child, table, _key_columns(tables), report),
                               on=table["key"], how="left")
            count = f"{table.get('prefix') or child}__count"
            if count in frame.collect_schema().names():  # rows without children have 0 of them
                frame = frame.with_columns(pl.col(count).fill_null(0))
    return frame


def _join_one(parent, child, name: str, table: Dict[str, Any], report: Dict[str, Any]):
    key = table["key"]
    parent_names = set(parent.collect_schema().names())
    child = child.unique(subset=[key], keep="first")
    renames = {c: f"{table.get('prefix') or name}__{c}" for c in child.collect_schema().names()
               if c != key and (c in parent_names or table.get("prefix"))}
    if renames:
        child = child.rename(renames)
    report["tables"][name] = {"relation": "one", "key": key,
                              "columns": [c for c in child.collect_schema().names() if c != key]}
    return parent.join(child, on=key, how="left", coalesce=True)


def _aggregate(child, name: str, table: Dict[str, Any], key_columns: set, report: Dict[str, Any]):
    """One row per key: count, numeric and categorical statistics, ordered statistics."""
    import polars as pl

    key = table["key"]
    order_by = table.get("order_by")
    settings = table.get("aggregate") or {}
    prefix = table.get("prefix") or name
    schema = child.collect_schema()
    skip = key_columns | {key, order_by}
    numeric_cols = [c for c, t in schema.items() if c not in skip and t.is_numeric()]
    categorical_cols = [c for c, t in schema.items() if c not in skip and not t.is_numeric()]
    numeric_stats = list(settings.get("numeric") or (
        list(DEFAULT_NUMERIC) + (["last", "trend"] if order_by else [])))
    categorical_stats = list(settings.get("categorical") or (
        list(DEFAULT_CATEGORICAL) + (["last"] if order_by else [])))

    def ordered(column: str):
        return pl.col(column).sort_by(order_by) if order_by else pl.col(column)

    exprs = [pl.len().alias(f"{prefix}__count")] if settings.get("count", True) else []
    for column in numeric_cols:
        for stat in numeric_stats:
            out = f"{prefix}__{column}__{stat}"
            if stat == "trend":
                exprs.append((ordered(column).drop_nulls().last() - ordered(column).drop_nulls().first()).alias(out))
            elif stat in ("first", "last"):
                exprs.append(getattr(ordered(column).drop_nulls(), stat)().alias(out))
            elif stat == "nunique":
                exprs.append(pl.col(column).n_unique().alias(out))
            else:
                exprs.append(getattr(pl.col(column), stat)().alias(out))
        for n in settings.get("last_n") or []:
            exprs.append(ordered(column).tail(n).mean().alias(f"{prefix}__{column}__last{n}_mean"))
    for column in categorical_cols:
        for stat in categorical_stats:
            out = f"{prefix}__{column}__{stat}"
            if stat == "nunique":
                exprs.append(pl.col(column).n_unique().alias(out))
            elif stat == "mode":
                exprs.append(pl.col(column).drop_nulls().mode().first().alias(out))
            else:
                exprs.append(getattr(ordered(column).drop_nulls(), stat)().alias(out))
    partitions = int(settings.get("partitions") or 1)
    report["tables"][name] = {"relation": "many", "key": key, "order_by": order_by,
                              "numeric_columns": len(numeric_cols), "categorical_columns": len(categorical_cols),
                              "features": len(exprs), "partitions": partitions}
    if partitions == 1:
        return child.group_by(key).agg(exprs).with_columns(pl.col(pl.Float64).cast(pl.Float32))
    # Large tables: roll up one slice of keys at a time (each key's rows all fall in one slice),
    # so memory holds 1/partitions of the table instead of all of it
    parts = []
    for part in range(partitions):
        sliced = child.filter(pl.col(key).hash(seed=0) % partitions == part)
        parts.append(sliced.group_by(key).agg(exprs).with_columns(pl.col(pl.Float64).cast(pl.Float32))
                     .collect(engine="streaming"))
        logger.info("Rolled up %s: slice %d/%d", name, part + 1, partitions)
    return pl.concat(parts, how="diagonal_relaxed").lazy()


# --------------------------------------------------------------------------- #
# Assembling
# --------------------------------------------------------------------------- #

def cache_key(tables: Dict[str, Any], data_dir: Path) -> str:
    files = sorted({t[k] for t in tables.values() for k in ("train", "test", "file") if k in t})
    stamps = {f: _source_stamp(Path(data_dir) / f) for f in files}
    payload = json.dumps({"tables": tables, "files": stamps, "version": TABLES_VERSION}, sort_keys=True,
                         default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def assemble(tables: Dict[str, Any], data_dir: Path, cache_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                                                    Dict[str, Any]]:
    """Train and test modelling tables (pandas) for a ``tables`` block, cached as Parquet."""
    import polars as pl

    errors = check_tables(tables)
    if errors:
        raise ValueError("; ".join(errors))
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    folder = cache_dir / cache_key(tables, data_dir)
    if (folder / "report.json").exists():
        report = json.loads((folder / "report.json").read_text())
        return _read(folder / "train.parquet"), _read(folder / "test.parquet"), {**report, "cached": True}

    raw_dir = cache_dir / "raw"
    main = main_table(tables)
    report: Dict[str, Any] = {"main": main, "tables": {}, "cache": str(folder)}
    frames = {}
    for part in ("train", "test"):
        frame = _scan(tables[main], data_dir, raw_dir, part).with_row_index("__row__")
        frame = _attach_children(tables, main, frame, data_dir, raw_dir, report)
        # Joins and streaming may reorder rows: restore the main file's order, narrow floats
        frame = frame.sort("__row__").drop("__row__").with_columns(pl.col(pl.Float64).cast(pl.Float32))
        frames[part] = frame.collect(engine="streaming")
    # Columns the roll-ups made for train but not test (or the reverse) are dropped
    test_names = set(frames["test"].columns)
    folder.mkdir(parents=True, exist_ok=True)
    for part, frame in frames.items():
        frame.write_parquet(folder / f"{part}.parquet")
    report.update(train_shape=list(frames["train"].shape), test_shape=list(frames["test"].shape),
                  n_columns=len(frames["train"].columns),
                  test_only_missing=[c for c in frames["train"].columns if c not in test_names][:50])
    (folder / "report.json").write_text(json.dumps(report, indent=2, default=str))
    del frames
    logger.info("Assembled %s: train %s, test %s", main, report["train_shape"], report["test_shape"])
    return _read(folder / "train.parquet"), _read(folder / "test.parquet"), report


def _read(path: Path) -> pd.DataFrame:
    """Parquet to pandas with 64-bit floats narrowed (nullable integers arrive as float64)."""
    frame = pd.read_parquet(path)
    wide = frame.select_dtypes("float64").columns
    if len(wide):
        frame[wide] = frame[wide].astype("float32")
    return frame


__all__ = ["assemble", "cache_key", "check_tables", "main_files", "main_table"]
