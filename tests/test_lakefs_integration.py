# ABOUTME: Integration tests for lakeFS helpers against a live lakeFS instance.
# ABOUTME: Creates temp repos, tests parquet round-trips, branching, and commit logs.
"""Integration tests for helpers.py lakeFS functions against a live lakeFS instance.

These tests require a running lakeFS server. They are marked with
``@pytest.mark.integration`` so they can be selected/excluded easily::

    # Run only integration tests
    pytest tests/test_lakefs_integration.py -v -m integration

    # Skip integration tests
    pytest -m "not integration"

The tests create a temporary repository, exercise read/write/commit/branch
operations, and clean up afterward.

Environment
-----------
lakeFS credentials are read from ``~/.config/lakefs/lakectl.yaml`` via the
existing ``notebook_setup`` module, falling back to environment variables.
"""

from __future__ import annotations

import io
import logging
import time
import uuid
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)

# Unique test repo name to avoid collisions.
_TEST_REPO_PREFIX = "test-helpers-integ"


def _make_test_repo_name() -> str:
    """Generate a unique repo name for this test session."""
    short_id = uuid.uuid4().hex[:8]
    return f"{_TEST_REPO_PREFIX}-{short_id}"


def _get_lakefs_client():
    """Create a lakeFS client from the standard config sources.

    Returns None if the server is unreachable.
    """
    try:
        from notebook_setup import build_notebook_config
        config = build_notebook_config()
    except Exception as exc:
        LOGGER.warning("Could not build notebook config: %s", exc)
        return None, None

    try:
        import lakefs
        client = lakefs.Client(
            host=config.endpoint,
            username=config.access_key,
            password=config.secret_key,
        )
        return client, config
    except Exception as exc:
        LOGGER.warning("Could not create lakeFS client: %s", exc)
        return None, None


def _server_is_reachable(client) -> bool:
    """Quick check whether the lakeFS server responds."""
    if client is None:
        return False
    try:
        import urllib.request
        from notebook_setup import build_notebook_config
        config = build_notebook_config()
        url = config.endpoint.rstrip("/") + "/api/v1/healthcheck"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status in (200, 204)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Module-level skip if server unreachable
# ---------------------------------------------------------------------------

_client, _config = _get_lakefs_client()
_server_up = _server_is_reachable(_client)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _server_up,
        reason="lakeFS server is not reachable -- skipping integration tests",
    ),
]


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lakefs_client():
    """Return a lakeFS client for the test session."""
    client, _ = _get_lakefs_client()
    assert client is not None, "lakeFS client should be available"
    return client


@pytest.fixture(scope="module")
def lakefs_config():
    """Return the notebook config for the test session."""
    _, config = _get_lakefs_client()
    assert config is not None, "Notebook config should be available"
    return config


@pytest.fixture(scope="module")
def temp_repo(lakefs_client, lakefs_config):
    """Create a temporary lakeFS repository and delete it after tests."""
    import lakefs

    repo_name = _make_test_repo_name()
    storage_ns = lakefs_config.storage_namespace or f"local://{repo_name}"

    LOGGER.info("Creating temp repo: %s (storage: %s)", repo_name, storage_ns)

    repo = lakefs.Repository(repo_name, client=lakefs_client)
    repo.create(
        storage_namespace=storage_ns,
        default_branch="main",
        exist_ok=False,
    )

    yield repo_name

    # Cleanup
    LOGGER.info("Deleting temp repo: %s", repo_name)
    try:
        repo.delete()
    except Exception as exc:
        LOGGER.warning("Failed to delete temp repo %s: %s", repo_name, exc)


@pytest.fixture
def sample_df():
    """Small DataFrame for parquet round-trip tests."""
    return pd.DataFrame({
        "flight_id": [100, 200, 300, 400],
        "airline": ["AA", "UA", "DL", "WN"],
        "arr_delay": [12.5, -5.0, 45.2, 0.0],
        "cancelled": [0, 0, 0, 1],
    })


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestCreateTempRepo:
    """Verify that we can create and access a temporary test repository."""

    def test_repo_exists(self, lakefs_client, temp_repo):
        import lakefs

        repo = lakefs.Repository(temp_repo, client=lakefs_client)
        branches = list(repo.branches())
        branch_ids = {b.id for b in branches}
        assert "main" in branch_ids, "Temp repo should have a 'main' branch"


class TestWriteReadParquetRoundtrip:
    """Verify parquet write -> read produces identical DataFrame."""

    def test_roundtrip(self, lakefs_client, temp_repo, sample_df):
        from helpers import write_parquet, read_parquet, lakefs_commit

        # Write
        write_parquet(
            client=lakefs_client,
            repo_name=temp_repo,
            branch_name="main",
            path="test/roundtrip.parquet",
            df=sample_df,
        )

        # Commit so the data is visible on the branch
        lakefs_commit(
            client=lakefs_client,
            repo_name=temp_repo,
            branch_name="main",
            message="Integration test: write roundtrip data",
        )

        # Read back
        result_df = read_parquet(
            client=lakefs_client,
            repo_name=temp_repo,
            branch_name="main",
            path="test/roundtrip.parquet",
        )

        pd.testing.assert_frame_equal(result_df, sample_df)


class TestCreateBranchAndCommit:
    """Verify branch creation, writing, committing, and log inspection."""

    def test_branch_create_write_commit(self, lakefs_client, temp_repo, sample_df):
        from helpers import create_branch, write_parquet, lakefs_commit
        import lakefs

        # Create a branch
        branch = create_branch(
            client=lakefs_client,
            repo_name=temp_repo,
            branch_name="test-branch",
            source_branch="main",
        )

        # Write data on the new branch
        write_parquet(
            client=lakefs_client,
            repo_name=temp_repo,
            branch_name="test-branch",
            path="test/branch_data.parquet",
            df=sample_df,
        )

        # Commit
        ref = lakefs_commit(
            client=lakefs_client,
            repo_name=temp_repo,
            branch_name="test-branch",
            message="Integration test: branch commit",
            metadata={"test": "true"},
        )

        # Verify commit log has our message
        repo = lakefs.Repository(temp_repo, client=lakefs_client)
        test_branch = repo.branch("test-branch")
        commits = list(test_branch.log(max_amount=5))

        messages = [c.message for c in commits]
        assert "Integration test: branch commit" in messages, \
            f"Expected commit message in log. Got: {messages}"

    def test_commit_returns_ref_with_id(self, lakefs_client, temp_repo, sample_df):
        from helpers import write_parquet, lakefs_commit

        write_parquet(
            client=lakefs_client,
            repo_name=temp_repo,
            branch_name="main",
            path="test/ref_check.parquet",
            df=sample_df,
        )

        ref = lakefs_commit(
            client=lakefs_client,
            repo_name=temp_repo,
            branch_name="main",
            message="Integration test: ref id check",
        )

        assert hasattr(ref, "id"), "Commit reference should have an 'id' attribute"
        assert isinstance(ref.id, str) and len(ref.id) > 0, \
            f"Commit ref id should be a non-empty string, got: {ref.id!r}"


class TestBranchIsolation:
    """Verify that writes to one branch are not visible on another."""

    def test_data_is_isolated_between_branches(self, lakefs_client, temp_repo):
        from helpers import create_branch, write_parquet, read_parquet, lakefs_commit

        # Create two branches from main
        create_branch(
            client=lakefs_client,
            repo_name=temp_repo,
            branch_name="branch-a",
            source_branch="main",
        )
        create_branch(
            client=lakefs_client,
            repo_name=temp_repo,
            branch_name="branch-b",
            source_branch="main",
        )

        # Write different data to each branch
        df_a = pd.DataFrame({"value": [1, 2, 3], "label": ["a", "a", "a"]})
        df_b = pd.DataFrame({"value": [10, 20, 30], "label": ["b", "b", "b"]})

        write_parquet(lakefs_client, temp_repo, "branch-a", "test/isolated.parquet", df_a)
        lakefs_commit(lakefs_client, temp_repo, "branch-a", "Write data A")

        write_parquet(lakefs_client, temp_repo, "branch-b", "test/isolated.parquet", df_b)
        lakefs_commit(lakefs_client, temp_repo, "branch-b", "Write data B")

        # Read from each branch and verify isolation
        result_a = read_parquet(lakefs_client, temp_repo, "branch-a", "test/isolated.parquet")
        result_b = read_parquet(lakefs_client, temp_repo, "branch-b", "test/isolated.parquet")

        pd.testing.assert_frame_equal(result_a, df_a)
        pd.testing.assert_frame_equal(result_b, df_b)

        # Verify branch-a does NOT see branch-b's data
        assert result_a["label"].iloc[0] == "a"
        assert result_b["label"].iloc[0] == "b"
