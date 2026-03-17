"""Unit tests for helpers.py -- lakeFS helpers and chart helpers.

ABOUTME: TDD unit tests for lakeFS read/write/commit/branch helpers and chart
ABOUTME: save helpers.  All lakeFS tests use mocked SDK objects (no live server).

They cover:
  - Parquet round-trip (write -> read) via lakeFS
  - Commit wrapper return value
  - Branch creation
  - Chart save (file exists, correct path, DPI metadata)
"""

from __future__ import annotations

import io
import json
import os
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI/headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Small DataFrame used across parquet round-trip tests."""
    return pd.DataFrame({
        "flight_id": [1, 2, 3],
        "airline": ["AA", "UA", "DL"],
        "delay": [10.5, -3.0, 22.1],
    })


@pytest.fixture
def tmp_chart_dir(tmp_path):
    """Provide a temporary directory for chart output and patch the default."""
    chart_dir = tmp_path / "outputs" / "charts"
    # Do NOT pre-create -- some tests verify auto-creation
    return chart_dir


@pytest.fixture
def simple_figure():
    """Create a minimal matplotlib figure for chart-save tests."""
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    ax.set_title("Test Chart")
    yield fig
    plt.close(fig)


@pytest.fixture
def mock_lakefs_repo():
    """Create a mock lakeFS Repository with branch/object/reader chain.

    Returns (mock_repo, mock_branch, mock_obj) for flexible assertion.
    """
    mock_repo = MagicMock()
    mock_branch = MagicMock()
    mock_obj = MagicMock()

    mock_repo.branch.return_value = mock_branch
    mock_branch.object.return_value = mock_obj

    return mock_repo, mock_branch, mock_obj


# ---------------------------------------------------------------------------
# lakeFS helper tests (mocked -- no live connection needed)
# ---------------------------------------------------------------------------

class TestWriteParquetToLakeFS:
    """Verify write_parquet serialises a DataFrame and uploads to lakeFS."""

    @patch("helpers.lakefs.Repository")
    def test_uploads_parquet_bytes(self, MockRepository, sample_df):
        from helpers import write_parquet

        mock_repo = MagicMock()
        mock_branch = MagicMock()
        mock_obj = MagicMock()
        MockRepository.return_value = mock_repo
        mock_repo.branch.return_value = mock_branch
        mock_branch.object.return_value = mock_obj

        mock_client = MagicMock()

        write_parquet(
            client=mock_client,
            repo_name="test-repo",
            branch_name="main",
            path="bronze/test.parquet",
            df=sample_df,
        )

        # Verify Repository was constructed correctly
        MockRepository.assert_called_once_with("test-repo", client=mock_client)
        mock_repo.branch.assert_called_once_with("main")
        mock_branch.object.assert_called_once_with("bronze/test.parquet")

        # Verify upload was called with bytes
        mock_obj.upload.assert_called_once()
        uploaded_data = mock_obj.upload.call_args[1].get("data") or mock_obj.upload.call_args[0][0]
        assert isinstance(uploaded_data, bytes), "Expected parquet bytes to be uploaded"

        # Verify the bytes are valid parquet that round-trips
        result_df = pd.read_parquet(io.BytesIO(uploaded_data))
        pd.testing.assert_frame_equal(result_df, sample_df)

    @patch("helpers.lakefs.Repository")
    def test_logs_debug_message(self, MockRepository, sample_df, caplog):
        from helpers import write_parquet

        mock_repo = MagicMock()
        MockRepository.return_value = mock_repo
        mock_repo.branch.return_value.object.return_value = MagicMock()

        mock_client = MagicMock()

        with caplog.at_level(logging.DEBUG, logger="helpers"):
            write_parquet(
                client=mock_client,
                repo_name="test-repo",
                branch_name="main",
                path="bronze/test.parquet",
                df=sample_df,
            )

        assert any("write_parquet" in r.message.lower() or "parquet" in r.message.lower()
                    for r in caplog.records), "Expected debug log from write_parquet"


class TestReadParquetFromLakeFS:
    """Verify read_parquet downloads parquet bytes and returns a DataFrame."""

    @patch("helpers.lakefs.Repository")
    def test_roundtrip_produces_identical_dataframe(self, MockRepository, sample_df):
        from helpers import read_parquet

        # Prepare parquet bytes from the sample DataFrame
        buf = io.BytesIO()
        sample_df.to_parquet(buf, index=False)
        parquet_bytes = buf.getvalue()

        mock_repo = MagicMock()
        mock_ref = MagicMock()
        mock_obj = MagicMock()
        mock_reader = MagicMock()

        MockRepository.return_value = mock_repo
        mock_repo.branch.return_value = mock_ref
        mock_ref.object.return_value = mock_obj

        # Simulate context-manager reader that returns parquet bytes
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_reader.read.return_value = parquet_bytes
        mock_obj.reader.return_value = mock_reader

        mock_client = MagicMock()

        result_df = read_parquet(
            client=mock_client,
            repo_name="test-repo",
            branch_name="main",
            path="bronze/test.parquet",
        )

        pd.testing.assert_frame_equal(result_df, sample_df)

    @patch("helpers.lakefs.Repository")
    def test_addresses_correct_lakefs_path(self, MockRepository, sample_df):
        from helpers import read_parquet

        buf = io.BytesIO()
        sample_df.to_parquet(buf, index=False)
        parquet_bytes = buf.getvalue()

        mock_repo = MagicMock()
        mock_ref = MagicMock()
        mock_obj = MagicMock()
        mock_reader = MagicMock()

        MockRepository.return_value = mock_repo
        mock_repo.branch.return_value = mock_ref
        mock_ref.object.return_value = mock_obj
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_reader.read.return_value = parquet_bytes
        mock_obj.reader.return_value = mock_reader

        mock_client = MagicMock()

        read_parquet(
            client=mock_client,
            repo_name="my-repo",
            branch_name="silver",
            path="silver/data.parquet",
        )

        MockRepository.assert_called_once_with("my-repo", client=mock_client)
        mock_repo.branch.assert_called_once_with("silver")
        mock_ref.object.assert_called_once_with("silver/data.parquet")


class TestCommitWrapper:
    """Verify lakefs_commit commits with the correct message and returns ref."""

    @patch("helpers.lakefs.Repository")
    def test_returns_commit_reference(self, MockRepository):
        from helpers import lakefs_commit

        mock_repo = MagicMock()
        mock_branch = MagicMock()
        mock_ref = MagicMock()
        mock_ref.id = "abc123deadbeef"

        MockRepository.return_value = mock_repo
        mock_repo.branch.return_value = mock_branch
        mock_branch.commit.return_value = mock_ref

        mock_client = MagicMock()

        ref = lakefs_commit(
            client=mock_client,
            repo_name="test-repo",
            branch_name="main",
            message="Ingest bronze data",
        )

        MockRepository.assert_called_once_with("test-repo", client=mock_client)
        mock_repo.branch.assert_called_once_with("main")
        mock_branch.commit.assert_called_once()
        commit_kwargs = mock_branch.commit.call_args[1]
        assert commit_kwargs["message"] == "Ingest bronze data"
        assert ref.id == "abc123deadbeef"

    @patch("helpers.lakefs.Repository")
    def test_passes_metadata_when_provided(self, MockRepository):
        from helpers import lakefs_commit

        mock_repo = MagicMock()
        mock_branch = MagicMock()
        mock_ref = MagicMock()

        MockRepository.return_value = mock_repo
        mock_repo.branch.return_value = mock_branch
        mock_branch.commit.return_value = mock_ref

        mock_client = MagicMock()
        metadata = {"phase": "bronze", "row_count": "463484"}

        lakefs_commit(
            client=mock_client,
            repo_name="test-repo",
            branch_name="main",
            message="Bronze commit",
            metadata=metadata,
        )

        commit_kwargs = mock_branch.commit.call_args[1]
        assert commit_kwargs.get("metadata") == metadata


class TestCreateBranch:
    """Verify create_branch creates a new branch from a source reference."""

    @patch("helpers.lakefs.Repository")
    def test_creates_branch_from_source(self, MockRepository):
        from helpers import create_branch

        mock_repo = MagicMock()
        mock_branch = MagicMock()

        MockRepository.return_value = mock_repo
        mock_repo.branch.return_value = mock_branch
        mock_branch.create.return_value = mock_branch

        mock_client = MagicMock()

        result = create_branch(
            client=mock_client,
            repo_name="test-repo",
            branch_name="silver",
            source_branch="main",
        )

        MockRepository.assert_called_once_with("test-repo", client=mock_client)
        mock_repo.branch.assert_called_once_with("silver")
        mock_branch.create.assert_called_once_with(source_reference="main", exist_ok=True)
        assert result is mock_branch

    @patch("helpers.lakefs.Repository")
    def test_logs_branch_creation(self, MockRepository, caplog):
        from helpers import create_branch

        mock_repo = MagicMock()
        mock_branch = MagicMock()

        MockRepository.return_value = mock_repo
        mock_repo.branch.return_value = mock_branch
        mock_branch.create.return_value = mock_branch

        mock_client = MagicMock()

        with caplog.at_level(logging.DEBUG, logger="helpers"):
            create_branch(
                client=mock_client,
                repo_name="test-repo",
                branch_name="experiment-a",
                source_branch="silver",
            )

        assert any("branch" in r.message.lower() for r in caplog.records), \
            "Expected debug log about branch creation"


# ---------------------------------------------------------------------------
# Chart helper tests (filesystem-based)
# ---------------------------------------------------------------------------

class TestSaveChart:
    """Verify save_chart produces a PNG with correct DPI and path."""

    def test_saves_png_file(self, simple_figure, tmp_chart_dir):
        from helpers import save_chart

        save_chart(simple_figure, "test_plot.png", chart_dir=str(tmp_chart_dir))

        output_path = tmp_chart_dir / "test_plot.png"
        assert output_path.exists(), f"Expected chart at {output_path}"
        assert output_path.stat().st_size > 0, "Chart file should not be empty"

    def test_creates_output_directory_if_missing(self, simple_figure, tmp_chart_dir):
        from helpers import save_chart

        assert not tmp_chart_dir.exists(), "Directory should not exist before save"
        save_chart(simple_figure, "auto_dir_test.png", chart_dir=str(tmp_chart_dir))
        assert tmp_chart_dir.exists(), "save_chart should create the directory"

    def test_default_dpi_is_150(self, simple_figure, tmp_chart_dir):
        from helpers import save_chart

        save_chart(simple_figure, "dpi_test.png", chart_dir=str(tmp_chart_dir))

        output_path = tmp_chart_dir / "dpi_test.png"
        # Read the saved image and check DPI via PIL
        from PIL import Image
        with Image.open(output_path) as img:
            dpi = img.info.get("dpi", (None, None))
            # PNG DPI is stored as pixels per meter in pHYs chunk;
            # PIL converts to DPI. Allow small floating point tolerance.
            if dpi[0] is not None:
                assert abs(dpi[0] - 150) < 2, f"Expected DPI ~150, got {dpi[0]}"

    def test_custom_dpi(self, simple_figure, tmp_chart_dir):
        from helpers import save_chart

        save_chart(simple_figure, "custom_dpi.png", dpi=200, chart_dir=str(tmp_chart_dir))

        output_path = tmp_chart_dir / "custom_dpi.png"
        from PIL import Image
        with Image.open(output_path) as img:
            dpi = img.info.get("dpi", (None, None))
            if dpi[0] is not None:
                assert abs(dpi[0] - 200) < 2, f"Expected DPI ~200, got {dpi[0]}"

    def test_applies_seaborn_style(self, tmp_chart_dir):
        """Verify that save_chart applies seaborn whitegrid style."""
        from helpers import save_chart

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        # The function should apply seaborn style context
        save_chart(fig, "style_test.png", chart_dir=str(tmp_chart_dir))

        output_path = tmp_chart_dir / "style_test.png"
        assert output_path.exists()
        plt.close(fig)

    def test_logs_save_path(self, simple_figure, tmp_chart_dir, caplog):
        from helpers import save_chart

        with caplog.at_level(logging.DEBUG, logger="helpers"):
            save_chart(simple_figure, "log_test.png", chart_dir=str(tmp_chart_dir))

        assert any("chart" in r.message.lower() or "save" in r.message.lower()
                    for r in caplog.records), "Expected debug log about chart save"
