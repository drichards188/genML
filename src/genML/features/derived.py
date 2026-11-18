"""Label-free features from a problem's own columns, set under ``feature_engineering``.

* ``derived: {name: "expression"}``: new numeric columns from arithmetic on columns,
  e.g. ``day: "floor(t / 86400)"``, ``ratio: "a / (b + 1)"``. Expressions are
  parsed, never executed as code: numbers, column names (or ``col("odd name")``),
  ``+ - * / // % **``, comparisons (giving 0/1) and the functions in ``FUNCTIONS``.
* ``keys: {name: [col, col, ...]}``: composite keys, the columns' values joined into
  one categorical column (e.g. a store and a product category).
  Keys are categoricals like any other, so target and count encodings can use them.
* ``group_stats: [{by: key or [cols], cols: [numeric cols], stats: [count, mean, std,
  min, max, nunique, deviation]}]``: statistics of other columns per group, computed
  on train and test rows together. When ``feature_engineering.time_column`` is set
  (or the item has ``time_column``) they are **past-only**: each row sees only rows
  of its group at or before its own time, as a live system would. ``past_only: false``
  uses every row, including later ones; that is not available at prediction time in
  real use, so it must be asked for explicitly.
* ``drift_filter: {threshold: 0.8, sample: 50000}``: drops columns that tell train rows
  from test rows on their own (adversarial validation, one small model per column);
  such columns describe when or where a row came from more than what it is.

None of these read the target.
"""

from __future__ import annotations

import ast
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FUNCTIONS = {
    "floor": np.floor, "ceil": np.ceil, "round": np.round, "abs": np.abs, "sqrt": np.sqrt,
    "log": np.log, "log1p": np.log1p, "exp": np.exp, "sign": np.sign,
    "isnull": lambda x: pd.isna(x).astype(float),
    "fillna": lambda x, v: x.fillna(v) if isinstance(x, pd.Series) else np.where(pd.isna(x), v, x),
    "minimum": np.minimum, "maximum": np.maximum,
}
GROUP_STATS = ("count", "mean", "std", "min", "max", "nunique", "deviation")
DERIVED_KEYS = ("derived", "keys", "group_stats", "drift_filter", "time_column")


# --------------------------------------------------------------------------- #
# Safe expressions
# --------------------------------------------------------------------------- #

_BINARY = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.true_divide,
           ast.FloorDiv: np.floor_divide, ast.Mod: np.mod, ast.Pow: np.power}
_COMPARE = {ast.Gt: np.greater, ast.GtE: np.greater_equal, ast.Lt: np.less, ast.LtE: np.less_equal,
            ast.Eq: np.equal, ast.NotEq: np.not_equal}


def expression_columns(expression: str) -> List[str]:
    """Column names an expression reads (raises ValueError on anything not allowed)."""
    columns: List[str] = []
    _walk(ast.parse(expression, mode="eval").body, columns)
    return columns


def _walk(node, columns: List[str]) -> None:
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        _walk(node.left, columns), _walk(node.right, columns)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        _walk(node.operand, columns)
    elif isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARE:
        _walk(node.left, columns), _walk(node.comparators[0], columns)
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        pass
    elif isinstance(node, ast.Name):
        columns.append(node.id)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        if node.func.id == "col" and len(node.args) == 1 and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            columns.append(node.args[0].value)
        elif node.func.id in FUNCTIONS:
            for arg in node.args:
                _walk(arg, columns)
        else:
            raise ValueError(f"function '{node.func.id}' is not allowed (allowed: {sorted(FUNCTIONS)}, col)")
    else:
        raise ValueError(f"'{ast.unparse(node)}' is not allowed in a derived-column expression")


def evaluate(expression: str, frame: pd.DataFrame) -> pd.Series:
    """Value of an expression on a frame's columns (float)."""
    tree = ast.parse(expression, mode="eval").body
    _walk(tree, [])
    missing = [c for c in expression_columns(expression) if c not in frame.columns]
    if missing:
        raise ValueError(f"expression '{expression}' uses unknown column(s) {missing}")

    def value(node):
        if isinstance(node, ast.BinOp):
            return _BINARY[type(node.op)](value(node.left), value(node.right))
        if isinstance(node, ast.UnaryOp):
            operand = value(node.operand)
            return -operand if isinstance(node.op, ast.USub) else operand
        if isinstance(node, ast.Compare):
            left, right = value(node.left), value(node.comparators[0])
            return _COMPARE[type(node.ops[0])](left, right).astype(float)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return pd.to_numeric(frame[node.id], errors="coerce").astype(float)
        if node.func.id == "col":
            return pd.to_numeric(frame[node.args[0].value], errors="coerce").astype(float)
        return FUNCTIONS[node.func.id](*[value(arg) for arg in node.args])

    with np.errstate(all="ignore"):
        result = value(tree)
    series = result if isinstance(result, pd.Series) else pd.Series(result, index=frame.index)
    return series.astype(float).replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# Checking settings
# --------------------------------------------------------------------------- #

def label_references(settings: Dict[str, Any], target: str) -> List[str]:
    """Where these settings read the target, directly or through a derived column or key built from it.

    Every feature here is computed on train and test rows together, outside the folds, so one that
    reads the label would hand each row its own label."""
    tainted, found = {target}, []
    for name, expression in (settings.get("derived") or {}).items():
        try:
            if set(expression_columns(str(expression))) & tainted:
                tainted.add(name)
                found.append(f"derived.{name}")
        except (SyntaxError, ValueError):
            continue  # reported by check_settings
    for name, columns in (settings.get("keys") or {}).items():
        if isinstance(columns, list) and set(columns) & tainted:
            tainted.add(name)
            found.append(f"keys.{name}")
    for i, item in enumerate(settings.get("group_stats") or []):
        if isinstance(item, dict):
            used = set(_as_list(item.get("by") or [])) | set(item.get("cols") or [])
            if used & tainted:
                found.append(f"group_stats[{i}]")
    return found


def check_settings(settings: Dict[str, Any], target: Optional[str] = None) -> List[str]:
    """Problems with derived / keys / group_stats / drift_filter settings (with ``target``: also any
    use of the label, which these out-of-fold-free features must never read)."""
    errors = []
    if target:
        errors += [f"{where} reads the target '{target}' (these features are built outside the folds)"
                   for where in label_references(settings, target)]
    derived = settings.get("derived")
    if derived is not None:
        if not isinstance(derived, dict):
            errors.append("derived must map new column names to expressions")
        else:
            for name, expression in derived.items():
                try:
                    expression_columns(str(expression))
                except (SyntaxError, ValueError) as exc:
                    errors.append(f"derived.{name}: {exc}")
    keys = settings.get("keys")
    if keys is not None and not (isinstance(keys, dict) and all(
            isinstance(v, list) and len(v) >= 2 and all(isinstance(c, str) for c in v) for v in keys.values())):
        errors.append("keys must map new key names to lists of at least two columns")
    stats = settings.get("group_stats")
    if stats is not None:
        if not isinstance(stats, list):
            errors.append("group_stats must be a list of {by, cols, stats}")
        else:
            for i, item in enumerate(stats):
                if not isinstance(item, dict) or not item.get("by") or not item.get("cols"):
                    errors.append(f"group_stats[{i}] needs 'by' and 'cols'")
                    continue
                unknown = set(item.get("stats") or []) - set(GROUP_STATS)
                if unknown:
                    errors.append(f"group_stats[{i}].stats: unknown {sorted(unknown)} (known: {list(GROUP_STATS)})")
    drift = settings.get("drift_filter")
    if drift is not None and drift is not False:
        if not isinstance(drift, dict):
            errors.append("drift_filter must be a mapping like {threshold: 0.8}")
        elif not 0.5 < float(drift.get("threshold", 0.8)) <= 1.0:
            errors.append("drift_filter.threshold must be in (0.5, 1.0]")
    return errors


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #

def _as_list(value) -> List[str]:
    return [value] if isinstance(value, str) else list(value)


def add_derived(train: pd.DataFrame, test: pd.DataFrame, derived: Dict[str, str]) -> List[str]:
    """Add derived columns in order (later ones may use earlier ones)."""
    for name, expression in derived.items():
        train[name] = evaluate(str(expression), train).astype("float32")
        test[name] = evaluate(str(expression), test).astype("float32")
    return list(derived)


def add_keys(train: pd.DataFrame, test: pd.DataFrame, keys: Dict[str, List[str]]) -> List[str]:
    for name, columns in keys.items():
        missing = [c for c in columns if c not in train.columns or c not in test.columns]
        if missing:
            raise ValueError(f"keys.{name} uses unknown column(s) {missing}")
        for frame in (train, test):
            frame[name] = frame[columns].astype(str).agg("_".join, axis=1)
    return list(keys)


def _past_only_stats(values: pd.Series, groups: pd.Series, order: np.ndarray, stats: List[str]) -> Dict[str, np.ndarray]:
    """Expanding statistics per group in time order: each row sees its group's rows up to itself."""
    n = len(values)
    v = values.to_numpy(dtype=float)[order]
    g = pd.Series(pd.factorize(groups.astype(str))[0][order])
    valid = ~np.isnan(v)
    filled = np.where(valid, v, 0.0)
    count_rows = g.groupby(g).cumcount().to_numpy() + 1
    count_valid = pd.Series(valid.astype(float)).groupby(g).cumsum().to_numpy()
    total = pd.Series(filled).groupby(g).cumsum().to_numpy()
    squares = pd.Series(filled ** 2).groupby(g).cumsum().to_numpy()
    with np.errstate(all="ignore"):
        mean = np.where(count_valid > 0, total / count_valid, np.nan)
        var = np.where(count_valid > 1, (squares - count_valid * mean ** 2) / (count_valid - 1), np.nan)
        std = np.sqrt(np.clip(var, 0, None))
    out: Dict[str, np.ndarray] = {}
    for stat in stats:
        if stat == "count":
            result = count_rows.astype(float)
        elif stat == "mean":
            result = mean
        elif stat == "std":
            result = std
        elif stat in ("min", "max"):
            series = pd.Series(np.where(valid, v, np.nan))
            result = (series.groupby(g).cummin() if stat == "min" else series.groupby(g).cummax()).to_numpy()
            result = pd.Series(result).groupby(g).ffill().to_numpy()
        elif stat == "nunique":
            pair = pd.Series(list(zip(g.to_numpy(), np.where(valid, v, np.nan).round(9))))
            first = (~pair.duplicated()).astype(float) * valid
            result = pd.Series(first).groupby(g).cumsum().to_numpy()
        else:  # deviation: how unusual this row's value is for its group so far
            with np.errstate(all="ignore"):
                result = np.where(std > 0, (v - mean) / std, np.nan)
        unsorted = np.empty(n)
        unsorted[order] = result
        out[stat] = unsorted
    return out


def _all_rows_stats(values: pd.Series, groups: pd.Series, stats: List[str]) -> Dict[str, np.ndarray]:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    grouped = numeric.groupby(groups.astype(str))
    out = {}
    for stat in stats:
        if stat == "count":
            out[stat] = groups.astype(str).map(groups.astype(str).value_counts()).to_numpy(dtype=float)
        elif stat == "deviation":
            mean, std = grouped.transform("mean"), grouped.transform("std")
            with np.errstate(all="ignore"):
                out[stat] = np.where(std > 0, (numeric - mean) / std, np.nan)
        else:
            out[stat] = grouped.transform(stat).to_numpy(dtype=float)
    return out


def add_group_stats(train: pd.DataFrame, test: pd.DataFrame, items: List[Dict[str, Any]],
                    time_column: Optional[str] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Group statistics on train and test rows together; past-only when a time column is known."""
    combined = pd.concat([train, test], axis=0, ignore_index=True, sort=False)
    n_train = len(train)
    added, report = [], []
    for item in items:
        by = _as_list(item["by"])
        missing = [c for c in by + list(item["cols"]) if c not in combined.columns]
        if missing:
            raise ValueError(f"group_stats by {by}: unknown column(s) {missing}")
        groups = combined[by].astype(str).agg("_".join, axis=1) if len(by) > 1 else combined[by[0]]
        stats = list(item.get("stats") or ["count", "mean", "std", "deviation"])
        time_col = item.get("time_column", time_column)
        past_only = bool(item.get("past_only", time_col is not None))
        if past_only and not time_col:
            raise ValueError(f"group_stats by {by}: past_only needs a time_column")
        order = None
        if past_only:
            times = pd.to_numeric(combined[time_col], errors="coerce").to_numpy(dtype=float)
            order = np.lexsort((np.arange(len(combined)), np.nan_to_num(times, nan=-np.inf)))
        by_name = "+".join(by)
        stats_without_count = [s for s in stats if s != "count"]
        if "count" in stats:  # one count per group, not per column
            values = combined[item["cols"][0]]
            counts = (_past_only_stats(values, groups, order, ["count"]) if past_only
                      else _all_rows_stats(values, groups, ["count"]))["count"]
            name = f"gs__{by_name}__count"
            train[name], test[name] = counts[:n_train].astype("float32"), counts[n_train:].astype("float32")
            added.append(name)
        for column in item["cols"]:
            values = pd.to_numeric(combined[column], errors="coerce")
            results = (_past_only_stats(values, groups, order, stats_without_count) if past_only
                       else _all_rows_stats(values, groups, stats_without_count))
            for stat, result in results.items():
                name = f"gs__{by_name}__{column}__{stat}"
                train[name], test[name] = result[:n_train].astype("float32"), result[n_train:].astype("float32")
                added.append(name)
        report.append({"by": by, "cols": list(item["cols"]), "stats": stats, "past_only": past_only,
                       "time_column": time_col if past_only else None})
    return added, report


def drift_scores(train: pd.DataFrame, test: pd.DataFrame, columns: List[str], sample: int = 50_000,
                 seed: int = 0) -> Dict[str, float]:
    """Per column, how well it alone tells train rows from test rows (cross-validated AUC)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    rng = np.random.default_rng(seed)
    take_train = rng.choice(len(train), min(sample, len(train)), replace=False)
    take_test = rng.choice(len(test), min(sample, len(test)), replace=False)
    is_test = np.r_[np.zeros(len(take_train)), np.ones(len(take_test))]
    scores = {}
    for column in columns:
        values = pd.concat([train[column].iloc[take_train], test[column].iloc[take_test]], ignore_index=True)
        if not pd.api.types.is_numeric_dtype(values):
            values = pd.Series(pd.factorize(values.astype(str))[0], dtype=float)
        X = values.to_numpy(dtype=float).reshape(-1, 1)
        if np.all(np.isnan(X)) or np.nanstd(X) == 0:
            scores[column] = 0.5
            continue
        model = HistGradientBoostingClassifier(max_iter=40, max_depth=3, learning_rate=0.2, random_state=seed)
        proba = cross_val_predict(model, X, is_test, cv=2, method="predict_proba")[:, 1]
        auc = roc_auc_score(is_test, proba)
        scores[column] = float(max(auc, 1 - auc))
    return scores


def apply(train: pd.DataFrame, test: pd.DataFrame, settings: Dict[str, Any],
          protected: List[str], target: Optional[str] = None) -> Dict[str, Any]:
    """Add derived columns, keys and group statistics, then drop drifting columns (in place).

    ``protected`` columns (target, id, cv column) are never dropped. Settings that read ``target``
    are refused.
    """
    errors = check_settings(settings, target)
    if errors:
        raise ValueError("; ".join(errors))
    report: Dict[str, Any] = {}
    if settings.get("derived"):
        report["derived"] = add_derived(train, test, settings["derived"])
    if settings.get("keys"):
        report["keys"] = add_keys(train, test, settings["keys"])
    if settings.get("group_stats"):
        added, items = add_group_stats(train, test, settings["group_stats"], settings.get("time_column"))
        report["group_stats"] = {"columns": added, "items": items}
    drift = settings.get("drift_filter")
    if drift:
        drift = drift if isinstance(drift, dict) else {}
        threshold = float(drift.get("threshold", 0.8))
        candidates = [c for c in train.columns if c in test.columns and c not in protected
                      and c not in (drift.get("keep") or [])]
        scores = drift_scores(train, test, candidates, int(drift.get("sample", 50_000)))
        dropped = sorted((c for c, s in scores.items() if s > threshold), key=lambda c: -scores[c])
        train.drop(columns=dropped, inplace=True)
        test.drop(columns=dropped, inplace=True)
        worst = sorted(scores.items(), key=lambda kv: -kv[1])[:20]
        report["drift_filter"] = {"threshold": threshold, "dropped": dropped,
                                  "highest": [{"column": c, "auc": round(s, 4)} for c, s in worst]}
    return report
