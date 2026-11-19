"""The referee: honest scoring of other tools' models on rows they never see.

A contest (``contest.py``) splits a real dataset into rows a contestant may train on and hidden rows
whose labels the referee keeps. A contestant (``arena.py``) is any command that reads the public files
and writes predictions for the hidden rows; it runs sandboxed with no network and no view of the labels.
The verdict (``verdict.py``) compares entries on the hidden rows with a noise-aware, graded rule, and the
checks (``checks.py``) look for leaks in the data itself.
"""
