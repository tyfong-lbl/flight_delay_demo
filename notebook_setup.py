"""Utilities for Notebook 01 setup and lakeFS connectivity.

The helpers in this module keep notebook bootstrap logic small, testable, and
well-documented. They are designed for explicit configuration, robust logging,
and predictable behavior in local/demo environments.

Credentials and endpoint are read from the lakectl CLI config file
(``~/.config/lakefs/lakectl.yaml``) by default, with environment-variable
overrides still supported.

If ``LAKEFS_STORAGE_NAMESPACE`` is not provided, a local default is derived as
``local://<repo_name>`` so repository creation can proceed in notebook flows.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


LOGGER = logging.getLogger(__name__)

# Default location of the lakectl CLI configuration file.
_DEFAULT_LAKECTL_CONFIG = Path.home() / ".config" / "lakefs" / "lakectl.yaml"


@dataclass(frozen=True)
class NotebookConfig:
    """Runtime configuration used by Notebook 01 setup cells."""

    endpoint: str
    access_key: str
    secret_key: str
    repo_name: str
    storage_namespace: str
    raw_csv_path: str
    sample_size_cap: int
    random_seed: int
    default_branch: str = "main"


@dataclass(frozen=True)
class LakeFSInitResult:
    """Structured result describing lakeFS repository initialization status."""

    connected: bool
    repository: str
    repository_created: bool
    default_branch_exists: bool
    message: str


def _parse_int_env(var_name: str, default: int) -> int:
    raw_value = os.getenv(var_name)
    if raw_value is None:
        return default

    stripped = raw_value.strip()
    try:
        return int(stripped)
    except ValueError:
        LOGGER.warning(
            "Invalid integer environment value; using default",
            extra={"variable": var_name, "value": raw_value, "default": default},
        )
        return default


def _strip_api_suffix(endpoint: str) -> str:
    """Strip trailing ``/api/v1`` (or ``/api/v1/``) from a lakeFS endpoint URL.

    The lakectl CLI config stores the full API path, but the Python SDK
    ``Client(host=...)`` expects the bare server URL.
    """
    for suffix in ("/api/v1/", "/api/v1"):
        if endpoint.endswith(suffix):
            return endpoint[: -len(suffix)]
    return endpoint


def _load_lakectl_config(
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Read the lakectl CLI YAML config and return flattened credential/endpoint info.

    Returns a dict with keys ``endpoint``, ``access_key``, ``secret_key``.
    Missing keys are omitted so the caller can detect gaps.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    RuntimeError
        If the config file cannot be parsed.
    """
    path = config_path or _DEFAULT_LAKECTL_CONFIG

    if not path.is_file():
        raise FileNotFoundError(
            f"lakectl config not found at {path}. "
            "Either place a valid lakectl.yaml there or set the LAKEFS_ENDPOINT, "
            "LAKEFS_ACCESS_KEY, and LAKEFS_SECRET_KEY environment variables."
        )

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception as exc:
        raise RuntimeError(f"Failed to parse lakectl config at {path}") from exc

    result: Dict[str, Any] = {}

    # credentials block
    creds = data.get("credentials", {})
    if creds.get("access_key_id"):
        result["access_key"] = creds["access_key_id"]
    if creds.get("secret_access_key"):
        result["secret_key"] = creds["secret_access_key"]

    # server block — strip /api/v1 for the Python SDK
    server = data.get("server", {})
    if server.get("endpoint_url"):
        result["endpoint"] = _strip_api_suffix(server["endpoint_url"])

    LOGGER.debug("Loaded lakectl config from %s", path)
    return result


def build_notebook_config(
    lakectl_config_path: Optional[Path] = None,
) -> NotebookConfig:
    """Build notebook configuration from lakectl YAML config + environment overrides.

    Resolution order (highest priority wins):
    1. Environment variables (``LAKEFS_ENDPOINT``, ``LAKEFS_ACCESS_KEY``, etc.)
    2. Values parsed from the lakectl CLI config file

    If the lakectl YAML file is missing **and** the required environment
    variables (``LAKEFS_ENDPOINT``, ``LAKEFS_ACCESS_KEY``, ``LAKEFS_SECRET_KEY``)
    are not all set, a ``FileNotFoundError`` is raised.  There is no silent
    fallback to a local lakeFS instance.

    Parameters
    ----------
    lakectl_config_path:
        Override path to the lakectl YAML file.  Useful for testing.
    """
    # If all three required env vars are present we can skip the yaml file.
    env_endpoint = os.getenv("LAKEFS_ENDPOINT")
    env_access_key = os.getenv("LAKEFS_ACCESS_KEY")
    env_secret_key = os.getenv("LAKEFS_SECRET_KEY")
    have_env = all([env_endpoint, env_access_key, env_secret_key])

    if have_env:
        yaml_cfg: Dict[str, Any] = {}
    else:
        # Will raise FileNotFoundError / RuntimeError if the file is
        # missing or malformed — intentionally no silent fallback.
        yaml_cfg = _load_lakectl_config(lakectl_config_path)

    endpoint = env_endpoint or yaml_cfg.get("endpoint")
    access_key = env_access_key or yaml_cfg.get("access_key")
    secret_key = env_secret_key or yaml_cfg.get("secret_key")

    # Final guard: refuse to continue with incomplete credentials.
    missing = []
    if not endpoint:
        missing.append("endpoint")
    if not access_key:
        missing.append("access_key")
    if not secret_key:
        missing.append("secret_key")
    if missing:
        raise ValueError(
            f"Incomplete lakeFS configuration — missing: {', '.join(missing)}. "
            "Provide a complete lakectl.yaml or set LAKEFS_ENDPOINT, "
            "LAKEFS_ACCESS_KEY, and LAKEFS_SECRET_KEY environment variables."
        )

    # At this point all three are guaranteed non-None/non-empty strings.
    assert isinstance(endpoint, str)
    assert isinstance(access_key, str)
    assert isinstance(secret_key, str)

    repo_name = os.getenv("LAKEFS_REPO", "flight-delay-demo")
    storage_namespace = os.getenv("LAKEFS_STORAGE_NAMESPACE") or f"local://{repo_name}"

    config = NotebookConfig(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        repo_name=repo_name,
        storage_namespace=storage_namespace,
        raw_csv_path=os.getenv("RAW_CSV_PATH", "data/raw/flights_sample_3m.csv"),
        sample_size_cap=_parse_int_env("SAMPLE_SIZE_CAP", 500_000),
        random_seed=_parse_int_env("RANDOM_SEED", 42),
    )
    LOGGER.debug(
        "Built notebook configuration",
        extra={
            "endpoint": config.endpoint,
            "repo_name": config.repo_name,
            "raw_csv_path": config.raw_csv_path,
            "sample_size_cap": config.sample_size_cap,
            "random_seed": config.random_seed,
        },
    )
    return config


def initialize_lakefs_repository(config: NotebookConfig) -> LakeFSInitResult:
    """Create/connect to a lakeFS repository and verify branch visibility.

    Returns a structured status object for notebook-friendly status printing.
    Any operational error is raised with contextual logging for easier debugging.
    """

    LOGGER.debug(
        "Initializing lakeFS repository",
        extra={"repo_name": config.repo_name, "endpoint": config.endpoint},
    )

    try:
        lakefs_module = importlib.import_module("lakefs")
    except Exception as exc:
        LOGGER.exception("Failed to import lakefs SDK")
        raise RuntimeError("lakefs SDK import failed") from exc

    client = lakefs_module.Client(
        host=config.endpoint,
        username=config.access_key,
        password=config.secret_key,
    )

    repository = lakefs_module.Repository(config.repo_name, client=client)
    repository_created = False

    LOGGER.debug(
        "Ensuring repository exists",
        extra={
            "repo_name": config.repo_name,
            "storage_namespace": config.storage_namespace,
            "default_branch": config.default_branch,
        },
    )
    repository.create(
        storage_namespace=config.storage_namespace,
        default_branch=config.default_branch,
        exist_ok=True,
    )
    repository_created = True

    branches = list(repository.branches())
    branch_ids = {branch.id for branch in branches}
    default_branch_exists = config.default_branch in branch_ids

    message = (
        f"Connected to lakeFS repository '{config.repo_name}' at {config.endpoint}. "
        f"default_branch_exists={default_branch_exists}."
    )
    LOGGER.debug(
        "lakeFS repository initialized",
        extra={
            "repo_name": config.repo_name,
            "repository_created": repository_created,
            "default_branch_exists": default_branch_exists,
        },
    )

    return LakeFSInitResult(
        connected=True,
        repository=config.repo_name,
        repository_created=repository_created,
        default_branch_exists=default_branch_exists,
        message=message,
    )
