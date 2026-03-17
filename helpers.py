# ABOUTME: Shared helpers for lakeFS data operations and chart generation.
# ABOUTME: Single source of shared logic for both demo notebooks.
"""Shared helpers for the Flight Delay Demo project.

This module is the single source of shared logic imported by both notebooks
(``01_data_prep.ipynb`` and ``02_experiments.ipynb``).  It provides:

- **lakeFS helpers** -- read/write parquet to lakeFS, commit wrapper, branch
  creation.  All functions accept an explicit ``client`` so notebooks control
  their own connection lifecycle.
- **Chart helpers** -- save matplotlib figures with consistent seaborn styling
  and ``dpi=150`` to ``outputs/charts/``.

Future phases will add data-cleaning, feature-engineering, and modeling
utilities to this same module.

Design principles
-----------------
* Every public function includes type hints, a docstring, and debug logging.
* lakeFS functions are thin wrappers around the SDK -- they keep notebook cells
  short while adding logging/traceability.
* Chart helpers enforce a uniform visual style for slide-quality screenshots.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import lakefs
from lakefs.exceptions import BadRequestException
import matplotlib

try:
    from IPython.core.getipython import get_ipython
except Exception:  # pragma: no cover - defensive import fallback
    get_ipython = None

if get_ipython is None or get_ipython() is None:
    matplotlib.use("Agg")  # non-interactive backend for scripts/CI only

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import seaborn as sns

LOGGER = logging.getLogger(__name__)

# Default chart output directory (relative to project root).
_DEFAULT_CHART_DIR = os.path.join("outputs", "charts")

# Default DPI for saved charts (presentation quality).
_DEFAULT_DPI = 150

# Seaborn style applied to every saved chart.
_SEABORN_STYLE = "whitegrid"


# ---------------------------------------------------------------------------
# lakeFS helpers
# ---------------------------------------------------------------------------


def write_parquet(
    client: Any,
    repo_name: str,
    branch_name: str,
    path: str,
    df: pd.DataFrame,
) -> None:
    """Serialize *df* to Parquet and upload it to a lakeFS branch.

    Parameters
    ----------
    client:
        An initialised ``lakefs.Client`` instance.
    repo_name:
        Name of the lakeFS repository (e.g. ``"flight-delay-demo"``).
    branch_name:
        Target branch (e.g. ``"main"``, ``"silver"``).
    path:
        Object path inside the repository (e.g.
        ``"bronze/flights_2023.parquet"``).
    df:
        The DataFrame to persist.
    """
    LOGGER.debug(
        "write_parquet: uploading %d rows to lakefs://%s/%s/%s",
        len(df),
        repo_name,
        branch_name,
        path,
    )

    # Serialize to in-memory Parquet bytes.
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    parquet_bytes = buf.getvalue()

    # Address the lakeFS object and upload.
    repo = lakefs.Repository(repo_name, client=client)
    branch = repo.branch(branch_name)
    obj = branch.object(path)
    obj.upload(data=parquet_bytes, mode="wb", content_type="application/octet-stream")

    LOGGER.debug(
        "write_parquet: upload complete (%d bytes) to %s",
        len(parquet_bytes),
        path,
    )


def read_parquet(
    client: Any,
    repo_name: str,
    branch_name: str,
    path: str,
) -> pd.DataFrame:
    """Download a Parquet object from a lakeFS branch and return a DataFrame.

    Parameters
    ----------
    client:
        An initialised ``lakefs.Client`` instance.
    repo_name:
        Name of the lakeFS repository.
    branch_name:
        Branch (or any ref expression) to read from.
    path:
        Object path inside the repository.

    Returns
    -------
    pd.DataFrame
        The deserialised DataFrame.
    """
    LOGGER.debug(
        "read_parquet: reading from lakefs://%s/%s/%s",
        repo_name,
        branch_name,
        path,
    )

    repo = lakefs.Repository(repo_name, client=client)
    ref = repo.branch(branch_name)
    obj = ref.object(path)

    raw_content: bytes | str = b""
    with obj.reader(mode="rb") as reader:
        raw_content = reader.read()

    raw_bytes = raw_content if isinstance(raw_content, bytes) else raw_content.encode()

    df = pd.read_parquet(io.BytesIO(raw_bytes))

    LOGGER.debug(
        "read_parquet: loaded DataFrame with shape %s from %s",
        df.shape,
        path,
    )
    return df


def lakefs_commit(
    client: Any,
    repo_name: str,
    branch_name: str,
    message: str,
    metadata: Optional[Dict[str, str]] = None,
) -> Any:
    """Commit staged changes on a lakeFS branch.

    Parameters
    ----------
    client:
        An initialised ``lakefs.Client`` instance.
    repo_name:
        Name of the lakeFS repository.
    branch_name:
        Branch to commit on.
    message:
        Human-readable commit message.
    metadata:
        Optional key-value metadata to attach to the commit.

    Returns
    -------
    Reference
        The commit reference object returned by the lakeFS SDK.
    """
    LOGGER.debug(
        "lakefs_commit: committing on %s/%s -- %s",
        repo_name,
        branch_name,
        message,
    )

    repo = lakefs.Repository(repo_name, client=client)
    branch = repo.branch(branch_name)
    commit_kwargs: Dict[str, Any] = {"message": message}
    if metadata is not None:
        commit_kwargs["metadata"] = metadata
    try:
        ref = branch.commit(**commit_kwargs)
    except BadRequestException as exc:
        if "no changes" in str(exc).lower():
            LOGGER.debug(
                "lakefs_commit: no changes to commit on %s/%s",
                repo_name,
                branch_name,
            )
            return None
        raise

    LOGGER.debug(
        "lakefs_commit: commit ref=%s on branch %s",
        getattr(ref, "id", ref),
        branch_name,
    )
    return ref


def create_branch(
    client: Any,
    repo_name: str,
    branch_name: str,
    source_branch: str,
) -> Any:
    """Create a new lakeFS branch from an existing source branch.

    Uses ``exist_ok=True`` so the call is idempotent -- safe to re-run in a
    notebook without raising ``ConflictException``.

    Parameters
    ----------
    client:
        An initialised ``lakefs.Client`` instance.
    repo_name:
        Name of the lakeFS repository.
    branch_name:
        Name of the new branch to create.
    source_branch:
        Name of the branch (or ref) to branch from.

    Returns
    -------
    Branch
        The lakeFS SDK ``Branch`` object for the newly-created branch.
    """
    LOGGER.debug(
        "create_branch: creating branch '%s' from '%s' in repo '%s'",
        branch_name,
        source_branch,
        repo_name,
    )

    repo = lakefs.Repository(repo_name, client=client)
    branch = repo.branch(branch_name)
    branch.create(source_reference=source_branch, exist_ok=True)

    LOGGER.debug(
        "create_branch: branch '%s' ready (source: '%s')",
        branch_name,
        source_branch,
    )
    return branch


# ---------------------------------------------------------------------------
# Phase 2 data prep helpers
# ---------------------------------------------------------------------------


def filter_to_year(
    df: pd.DataFrame,
    year: int,
    year_column: str = "year",
) -> pd.DataFrame:
    """Return only rows where *year_column* equals *year*.

    Parameters
    ----------
    df:
        Input DataFrame.
    year:
        Year value to keep (e.g. ``2023``).
    year_column:
        Name of the column containing year values.
    """
    LOGGER.debug(
        "filter_to_year: filtering %d rows where %s == %s",
        len(df),
        year_column,
        year,
    )
    filtered = df.loc[df[year_column] == year].copy()
    LOGGER.debug("filter_to_year: retained %d rows", len(filtered))
    return filtered


def add_delay_precursor(
    df: pd.DataFrame,
    arrival_delay_col: str = "arrival_delay",
    threshold_minutes: float = 15.0,
    output_col: str = "is_delayed_pre",
) -> pd.DataFrame:
    """Add a binary delay precursor column derived from arrival delay.

    The output is ``1`` when ``arrival_delay_col > threshold_minutes`` else
    ``0``.
    """
    LOGGER.debug(
        "add_delay_precursor: creating '%s' from '%s' with threshold=%s",
        output_col,
        arrival_delay_col,
        threshold_minutes,
    )
    tagged = df.copy()
    tagged[output_col] = (tagged[arrival_delay_col] > threshold_minutes).astype(int)
    return tagged


def stratified_sample_with_row_cap(
    df: pd.DataFrame,
    target_col: str,
    row_cap: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Return a stratified sample capped at *row_cap* rows.

    If the input already has ``<= row_cap`` rows, a copy is returned unchanged.
    Sampling is proportional to class frequency in ``target_col`` while ensuring
    the final sample size is exactly ``row_cap``.
    """
    if row_cap <= 0:
        raise ValueError("row_cap must be a positive integer")

    total_rows = len(df)
    LOGGER.debug(
        "stratified_sample_with_row_cap: input_rows=%d row_cap=%d target_col=%s",
        total_rows,
        row_cap,
        target_col,
    )

    if total_rows <= row_cap:
        LOGGER.debug(
            "stratified_sample_with_row_cap: input below cap, returning copy"
        )
        return df.copy()

    proportions = df[target_col].value_counts(normalize=True)
    desired_counts = (proportions * row_cap).round().astype(int)

    diff = row_cap - int(desired_counts.sum())
    if diff != 0:
        remainders = (proportions * row_cap) - (proportions * row_cap).round()
        adjust_order = remainders.sort_values(ascending=(diff < 0)).index.tolist()
        idx = 0
        while diff != 0 and adjust_order:
            label = adjust_order[idx % len(adjust_order)]
            candidate = desired_counts.loc[label] + (1 if diff > 0 else -1)
            if candidate >= 0:
                desired_counts.loc[label] = candidate
                diff += -1 if diff > 0 else 1
            idx += 1

    sampled_parts = []
    for label, count in desired_counts.items():
        class_rows = df[df[target_col] == label]
        take = min(count, len(class_rows))
        if take > 0:
            sampled_parts.append(class_rows.sample(n=take, random_state=random_seed))

    sampled = pd.concat(sampled_parts, axis=0)

    if len(sampled) < row_cap:
        remaining = df.drop(sampled.index)
        needed = row_cap - len(sampled)
        sampled = pd.concat(
            [sampled, remaining.sample(n=needed, random_state=random_seed)],
            axis=0,
        )

    sampled = sampled.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    LOGGER.debug("stratified_sample_with_row_cap: sampled_rows=%d", len(sampled))
    return sampled


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------


def save_chart(
    fig: Figure,
    filename: str,
    dpi: int = _DEFAULT_DPI,
    chart_dir: Optional[str] = None,
) -> Path:
    """Save a matplotlib figure as a PNG with consistent seaborn styling.

    The function applies the ``whitegrid`` seaborn theme before saving so that
    all charts share a uniform look for slide embedding.

    Parameters
    ----------
    fig:
        The matplotlib ``Figure`` to save.
    filename:
        File name (e.g. ``"delay_histogram.png"``).  Saved inside *chart_dir*.
    dpi:
        Resolution in dots per inch (default ``150``).
    chart_dir:
        Override the output directory.  Defaults to ``outputs/charts/``.

    Returns
    -------
    pathlib.Path
        Absolute path of the saved PNG.
    """
    target_dir = Path(chart_dir) if chart_dir else Path(_DEFAULT_CHART_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    output_path = target_dir / filename

    # Apply seaborn style context for consistent visuals.
    with sns.axes_style(_SEABORN_STYLE):
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )

    LOGGER.debug(
        "save_chart: saved chart to %s (dpi=%d, size=%d bytes)",
        output_path,
        dpi,
        output_path.stat().st_size,
    )
    return output_path
