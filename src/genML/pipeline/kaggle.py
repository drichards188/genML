"""Kaggle integration: credentials, competition metadata and data download.

Authentication, in order: the ``KAGGLE_API_TOKEN`` environment variable,
``~/.kaggle/access_token``, then legacy ``~/.kaggle/kaggle.json``
(username/key). Tokens are never written to disk by this module.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

API_BASE = "https://www.kaggle.com/api/v1"


class KaggleAuthError(RuntimeError):
    pass


def _auth_header() -> Dict[str, str]:
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        token_file = Path.home() / ".kaggle" / "access_token"
        if token_file.exists():
            token = token_file.read_text().strip()
    if token:
        return {"Authorization": f"Bearer {token}"}

    legacy = Path.home() / ".kaggle" / "kaggle.json"
    if legacy.exists():
        creds = json.loads(legacy.read_text())
        basic = base64.b64encode(f"{creds['username']}:{creds['key']}".encode()).decode()
        return {"Authorization": f"Basic {basic}"}

    raise KaggleAuthError(
        "No Kaggle credentials. Set KAGGLE_API_TOKEN or create ~/.kaggle/access_token "
        "(token from https://www.kaggle.com/settings/api)."
    )


def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = f"{API_BASE}/{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={**_auth_header(), "User-Agent": "genml"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode())


def competition_info(competition: str) -> Dict[str, Any]:
    """Metadata for one competition (title, evaluationMetric, deadline, ...)."""
    # Plain search only covers active competitions; ended ones you entered
    # are found through group=entered
    for params in ({"search": competition}, {"search": competition, "group": "entered"}):
        payload = _get("competitions/list", params)
        entries = payload if isinstance(payload, list) else payload.get("competitions", [])
        for entry in entries:
            if str(entry.get("ref", "")).rstrip("/").split("/")[-1] == competition:
                return entry
    raise LookupError(f"Competition '{competition}' not found via the Kaggle API")


def init_problem_from_kaggle(competition: str, name: Optional[str] = None, download: bool = True):
    """Create ``problems/<name>/problem.yaml`` for a Kaggle competition.

    The metric comes from the competition's ``evaluationMetric``; target, ID
    column and task are inferred from the data and written out explicitly so
    they can be reviewed and edited.
    """
    import pandas as pd

    from src.genML.pipeline.metrics import metric_from_kaggle_name
    from src.genML.pipeline.problem import (
        ProblemSpec, find_sample_submission, locate_data_dir, resolve_problem, save_problem,
    )

    name = name or competition
    spec = ProblemSpec(name=name, source={"type": "kaggle", "competition": competition})
    data_dir = locate_data_dir(spec, download=download)

    info = competition_info(competition)
    kaggle_metric = info.get("evaluationMetric")
    metric = metric_from_kaggle_name(kaggle_metric)
    if metric is None:
        logger.warning("Kaggle metric %r is not in the registry; declare 'metric' by hand", kaggle_metric)

    train_df = pd.read_csv(data_dir / "train.csv", nrows=200_000)
    test_df = pd.read_csv(data_dir / "test.csv", nrows=1_000)
    sample_path = find_sample_submission(data_dir)
    sample_df = pd.read_csv(sample_path, nrows=1_000) if sample_path else None
    if metric is not None:
        spec.metric = metric.name
    resolved = resolve_problem(spec, train_df, test_df, sample_df, data_dir=data_dir)

    spec.description = f"{info.get('title', competition)} (Kaggle metric: {kaggle_metric})"
    spec.target = resolved.target
    spec.id_column = resolved.id_column
    spec.task = resolved.task
    spec.metric = resolved.metric
    path = save_problem(spec)
    return path, resolved


def _kaggle_api():
    """Authenticated client from the official ``kaggle`` package (needed to submit)."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Submitting needs the kaggle package: pip install kaggle (Python 3.11+)") from exc
    _auth_header()  # clear error if no credentials
    api = KaggleApi()
    api.authenticate()
    return api


def submit_file(competition: str, file_path: Path, message: str) -> str:
    """Upload a submission file; returns Kaggle's response message."""
    response = _kaggle_api().competition_submit(file_name=str(file_path), message=message,
                                                competition=competition, quiet=True)
    return str(getattr(response, "message", "") or response)


def list_submissions(competition: str) -> list[Dict[str, Any]]:
    """Your submissions to a competition, newest first."""
    rows = []
    for item in _kaggle_api().competition_submissions(competition) or []:
        if item is None:
            continue
        rows.append({
            "ref": str(item.ref),
            "date": str(item.date),
            "description": item.description or "",
            "status": str(item.status).split(".")[-1],
            "public_score": item.public_score or None,
            "private_score": item.private_score or None,
            "file_name": item.file_name,
        })
    return rows


def download_competition(competition: str) -> Path:
    """Download competition files with kagglehub and return the folder."""
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("kagglehub is required for Kaggle downloads: pip install kagglehub") from exc
    _auth_header()  # fail early with a clear message if no credentials
    path = Path(kagglehub.competition_download(competition))
    logger.info("Downloaded %s to %s", competition, path)
    return path
