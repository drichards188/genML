"""Shuffled-label data for the canary run of a code cycle (trusted code; agents cannot change it).

``prepare_canary_data`` copies the problem's data folder as links, except the main training file,
which is rewritten with its target column permuted. Every column is read and written as text, so
nothing but the order of the labels changes. The canary run (the cycle's code and settings on this
copy, in its own outputs folder) must then not beat a constant prediction (``judge.canary_verdict``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def main_train_file(spec) -> str:
    if spec.tables:
        from src.genML.pipeline.tables import main_files

        return main_files(spec.tables)[0]
    return "train.csv"


def prepare_canary_data(problem: str, dest: Path, seed: int) -> Path:
    """A data folder like the problem's, with the training labels shuffled. Returns ``dest``."""
    import polars as pl

    from src.genML.pipeline.problem import load_problem, locate_data_dir

    spec = load_problem(problem)
    source = Path(locate_data_dir(spec))
    train_name = main_train_file(spec)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        link = dest / entry.name
        if entry.name != train_name and not link.exists():
            link.symlink_to(entry.resolve())
    shuffled = dest / train_name
    if not shuffled.exists():
        frame = pl.read_csv(source / train_name, infer_schema=False)          # all columns as text
        labels = frame[spec.target].to_numpy()
        order = np.random.default_rng(seed).permutation(len(labels))
        frame = frame.with_columns(pl.Series(spec.target, labels[order], dtype=pl.Utf8))
        partial = shuffled.with_suffix(".partial")
        frame.write_csv(partial)
        partial.replace(shuffled)
    return dest


__all__ = ["main_train_file", "prepare_canary_data"]
