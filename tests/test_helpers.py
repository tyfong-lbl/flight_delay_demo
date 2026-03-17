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
# Phase 2 TDD tests (expected to fail before implementation)
# ---------------------------------------------------------------------------

class TestPhase2FilteringAndSampling:
    """Define expected behavior for year filtering and stratified sampling."""

    def test_filter_to_year_keeps_only_2023_rows(self):
        from helpers import filter_to_year

        flights = pd.DataFrame(
            {
                "flight_id": [1, 2, 3, 4],
                "year": [2022, 2023, 2024, 2023],
                "arrival_delay": [10.0, 20.0, 5.0, -3.0],
            }
        )

        filtered = filter_to_year(flights, year=2023, year_column="year")

        assert set(filtered["year"].unique()) == {2023}
        assert filtered["flight_id"].tolist() == [2, 4]

    def test_stratified_sampling_preserves_class_balance_within_tolerance(self):
        from helpers import add_delay_precursor, stratified_sample_with_row_cap

        rng = np.random.default_rng(42)
        flights = pd.DataFrame(
            {
                "flight_id": np.arange(10_000),
                "year": np.full(10_000, 2023),
                "arrival_delay": np.where(
                    rng.random(10_000) < 0.3,
                    30.0,
                    0.0,
                ),
            }
        )

        tagged = add_delay_precursor(flights, arrival_delay_col="arrival_delay")
        sampled = stratified_sample_with_row_cap(
            tagged,
            target_col="is_delayed_pre",
            row_cap=2_000,
            random_seed=42,
        )

        source_ratio = tagged["is_delayed_pre"].mean()
        sampled_ratio = sampled["is_delayed_pre"].mean()

        assert abs(sampled_ratio - source_ratio) <= 0.02

    def test_sampling_respects_row_cap(self):
        from helpers import add_delay_precursor, stratified_sample_with_row_cap

        flights = pd.DataFrame(
            {
                "flight_id": np.arange(4_000),
                "year": np.full(4_000, 2023),
                "arrival_delay": np.where(np.arange(4_000) % 4 == 0, 40.0, 0.0),
            }
        )

        tagged = add_delay_precursor(flights, arrival_delay_col="arrival_delay")
        sampled = stratified_sample_with_row_cap(
            tagged,
            target_col="is_delayed_pre",
            row_cap=500,
            random_seed=7,
        )

        assert len(sampled) == 500


# ---------------------------------------------------------------------------
# Phase 3 TDD tests (expected to fail before implementation)
# ---------------------------------------------------------------------------


class TestPhase3SilverCleaning:
    """Define expected cleaning and target creation behavior for silver layer."""

    def test_drop_cancelled_flights(self):
        from helpers import drop_cancelled_flights

        flights = pd.DataFrame(
            {
                "flight_id": [1, 2, 3, 4],
                "CANCELLED": [0.0, 1.0, 0.0, 1.0],
            }
        )

        cleaned = drop_cancelled_flights(flights, cancelled_col="CANCELLED")

        assert cleaned["flight_id"].tolist() == [1, 3]
        assert (cleaned["CANCELLED"] == 0.0).all()

    def test_drop_missing_arrival_delay(self):
        from helpers import drop_missing_arrival_delay

        flights = pd.DataFrame(
            {
                "flight_id": [10, 20, 30],
                "arrival_delay": [5.0, np.nan, 18.0],
            }
        )

        cleaned = drop_missing_arrival_delay(flights, arrival_delay_col="arrival_delay")

        assert cleaned["flight_id"].tolist() == [10, 30]
        assert cleaned["arrival_delay"].isna().sum() == 0

    def test_remove_negative_air_time(self):
        from helpers import remove_negative_air_time

        flights = pd.DataFrame(
            {
                "flight_id": [1, 2, 3],
                "AIR_TIME": [100.0, -5.0, 45.0],
            }
        )

        cleaned = remove_negative_air_time(flights, air_time_col="AIR_TIME")

        assert cleaned["flight_id"].tolist() == [1, 3]
        assert (cleaned["AIR_TIME"] >= 0).all()

    def test_create_is_delayed_target_uses_15_min_threshold(self):
        from helpers import create_is_delayed_target

        flights = pd.DataFrame(
            {
                "flight_id": [1, 2, 3, 4],
                "arrival_delay": [15.0, 15.01, 0.0, 120.0],
            }
        )

        with_target = create_is_delayed_target(
            flights,
            arrival_delay_col="arrival_delay",
            threshold_minutes=15.0,
            output_col="is_delayed",
        )

        assert with_target["is_delayed"].tolist() == [0, 1, 0, 1]


class TestPhase3MissingValuesAndDtypes:
    """Define explicit missing-value policy and dtype normalization behavior."""

    def test_handle_remaining_missing_values_drops_required_and_imputes_optional(self):
        from helpers import handle_remaining_missing_values

        flights = pd.DataFrame(
            {
                "flight_id": [1, 2, 3, 4],
                "FL_DATE": ["2023-01-01", "2023-01-02", None, "2023-01-04"],
                "AIRLINE_CODE": ["AA", "UA", "DL", "WN"],
                "ORIGIN": ["SFO", "LAX", "JFK", "SEA"],
                "DEST": ["LAX", "SFO", "SEA", "JFK"],
                "DISTANCE": [337.0, 337.0, 2421.0, 954.0],
                "AIR_TIME": [62.0, 60.0, 320.0, 120.0],
                "DEP_DELAY": [10.0, np.nan, 5.0, np.nan],
            }
        )

        cleaned, audit = handle_remaining_missing_values(
            flights,
            drop_columns=["FL_DATE"],
            impute_numeric_columns=["DEP_DELAY"],
        )

        assert cleaned["flight_id"].tolist() == [1, 2, 4]
        assert cleaned["DEP_DELAY"].isna().sum() == 0
        assert audit["drop"]["FL_DATE"] == 1
        assert audit["impute"]["DEP_DELAY"]["imputed"] == 2
        assert audit["rows_removed_total"] == 1

    def test_handle_remaining_missing_values_audit_uses_zero_when_median_nan(self):
        from helpers import handle_remaining_missing_values

        flights = pd.DataFrame(
            {
                "flight_id": [1, 2],
                "FL_DATE": ["2023-01-01", "2023-01-02"],
                "DEP_DELAY": [np.nan, np.nan],
            }
        )

        cleaned, audit = handle_remaining_missing_values(
            flights,
            drop_columns=["FL_DATE"],
            impute_numeric_columns=["DEP_DELAY"],
        )

        assert cleaned["DEP_DELAY"].tolist() == [0.0, 0.0]
        assert audit["impute"]["DEP_DELAY"]["fill_value"] == 0.0

    def test_normalize_silver_dtypes_converts_datetime_categorical_numeric(self):
        from helpers import normalize_silver_dtypes

        flights = pd.DataFrame(
            {
                "FL_DATE": ["2023-01-01", "not-a-date"],
                "AIRLINE_CODE": ["AA", "UA"],
                "ORIGIN": ["SFO", "LAX"],
                "DEST": ["SEA", "JFK"],
                "DISTANCE": ["337", "bad-number"],
                "arrival_delay": ["5", "16"],
            }
        )

        normalized, audit = normalize_silver_dtypes(
            flights,
            datetime_columns=["FL_DATE"],
            categorical_columns=["AIRLINE_CODE", "ORIGIN", "DEST"],
            numeric_columns=["DISTANCE", "arrival_delay"],
        )

        assert pd.api.types.is_datetime64_any_dtype(normalized["FL_DATE"])
        assert pd.api.types.is_categorical_dtype(normalized["AIRLINE_CODE"])
        assert pd.api.types.is_categorical_dtype(normalized["ORIGIN"])
        assert pd.api.types.is_categorical_dtype(normalized["DEST"])
        assert pd.api.types.is_numeric_dtype(normalized["DISTANCE"])
        assert pd.api.types.is_numeric_dtype(normalized["arrival_delay"])
        assert audit["datetime"]["FL_DATE"]["coerced_to_na"] == 1
        assert audit["numeric"]["DISTANCE"]["coerced_to_na"] == 1


# ---------------------------------------------------------------------------
# Phase 4 TDD tests (expected to fail before implementation)
# ---------------------------------------------------------------------------


class TestPhase4ModelingUtilities:
    """Define expected behavior for shared modeling utility helpers."""

    def test_deterministic_train_test_split_is_stratified_and_reproducible(self):
        from helpers import deterministic_train_test_split

        rng = np.random.default_rng(42)
        rows = 1_000
        labels = np.where(rng.random(rows) < 0.35, 1, 0)
        frame = pd.DataFrame(
            {
                "feature_a": rng.normal(size=rows),
                "feature_b": rng.integers(0, 10, size=rows),
                "is_delayed": labels,
            }
        )

        split_one = deterministic_train_test_split(
            frame,
            target_col="is_delayed",
            test_size=0.2,
            random_state=42,
        )
        split_two = deterministic_train_test_split(
            frame,
            target_col="is_delayed",
            test_size=0.2,
            random_state=42,
        )

        X_train_1, X_test_1, y_train_1, y_test_1 = split_one
        X_train_2, X_test_2, y_train_2, y_test_2 = split_two

        assert len(X_train_1) == 800
        assert len(X_test_1) == 200
        pd.testing.assert_frame_equal(X_train_1, X_train_2)
        pd.testing.assert_frame_equal(X_test_1, X_test_2)
        pd.testing.assert_series_equal(y_train_1, y_train_2)
        pd.testing.assert_series_equal(y_test_1, y_test_2)

        source_rate = frame["is_delayed"].mean()
        train_rate = y_train_1.mean()
        test_rate = y_test_1.mean()
        assert abs(train_rate - source_rate) <= 0.02
        assert abs(test_rate - source_rate) <= 0.02

    def test_compute_binary_classification_metrics_returns_expected_keys_and_values(self):
        from helpers import compute_binary_classification_metrics

        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        y_scores = [0.1, 0.9, 0.2, 0.8]

        metrics = compute_binary_classification_metrics(y_true, y_pred, y_scores)

        assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "auc_pr"}
        assert metrics["accuracy"] == pytest.approx(0.5)
        assert metrics["precision"] == pytest.approx(0.5)
        assert metrics["recall"] == pytest.approx(0.5)
        assert metrics["f1"] == pytest.approx(0.5)
        assert metrics["auc_pr"] == pytest.approx(0.4166666, abs=1e-6)

    def test_metrics_json_round_trip(self, tmp_path):
        from helpers import load_metrics_json, save_metrics_json

        metrics = {
            "accuracy": 0.8125,
            "precision": 0.701,
            "recall": 0.645,
            "f1": 0.672,
            "auc_pr": 0.755,
        }
        output = tmp_path / "gold" / "metrics_time.json"

        save_metrics_json(metrics, output)
        loaded = load_metrics_json(output)

        assert output.exists()
        assert loaded == metrics

    def test_predictions_parquet_round_trip(self, tmp_path):
        from helpers import load_predictions_parquet, save_predictions_parquet

        predictions = pd.DataFrame(
            {
                "y_true": [0, 1, 0, 1],
                "y_scores": [0.11, 0.83, 0.21, 0.92],
            }
        )
        output = tmp_path / "gold" / "predictions_time.parquet"

        save_predictions_parquet(predictions, output)
        loaded = load_predictions_parquet(output)

        assert output.exists()
        pd.testing.assert_frame_equal(loaded, predictions)


class TestTemporalTrainTestSplit:
    """Verify temporal_train_test_split partitions by date cutoff."""

    @pytest.fixture
    def temporal_df(self):
        """Synthetic DataFrame spanning Jan-Dec 2023."""
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "FL_DATE": dates,
                "feature_a": rng.normal(size=len(dates)),
                "feature_b": rng.integers(0, 10, size=len(dates)),
                "is_delayed": rng.choice([0, 1], size=len(dates), p=[0.7, 0.3]),
            }
        )

    def test_all_train_dates_before_cutoff(self, temporal_df):
        from helpers import temporal_train_test_split

        X_train, X_test, y_train, y_test = temporal_train_test_split(
            temporal_df, target_col="is_delayed", cutoff_date="2023-11-01"
        )
        # Train dates should all be before cutoff
        train_dates = temporal_df.loc[X_train.index, "FL_DATE"]
        assert (train_dates < pd.Timestamp("2023-11-01")).all()

    def test_all_test_dates_at_or_after_cutoff(self, temporal_df):
        from helpers import temporal_train_test_split

        X_train, X_test, y_train, y_test = temporal_train_test_split(
            temporal_df, target_col="is_delayed", cutoff_date="2023-11-01"
        )
        test_dates = temporal_df.loc[X_test.index, "FL_DATE"]
        assert (test_dates >= pd.Timestamp("2023-11-01")).all()

    def test_both_splits_are_non_empty(self, temporal_df):
        from helpers import temporal_train_test_split

        X_train, X_test, y_train, y_test = temporal_train_test_split(
            temporal_df, target_col="is_delayed", cutoff_date="2023-11-01"
        )
        assert len(X_train) > 0
        assert len(X_test) > 0

    def test_splits_cover_all_rows(self, temporal_df):
        from helpers import temporal_train_test_split

        X_train, X_test, y_train, y_test = temporal_train_test_split(
            temporal_df, target_col="is_delayed", cutoff_date="2023-11-01"
        )
        assert len(X_train) + len(X_test) == len(temporal_df)

    def test_date_col_and_target_col_excluded_from_features(self, temporal_df):
        from helpers import temporal_train_test_split

        X_train, X_test, _, _ = temporal_train_test_split(
            temporal_df, target_col="is_delayed", cutoff_date="2023-11-01"
        )
        assert "FL_DATE" not in X_train.columns
        assert "is_delayed" not in X_train.columns
        assert "FL_DATE" not in X_test.columns
        assert "is_delayed" not in X_test.columns

    def test_raises_on_missing_target_col(self, temporal_df):
        from helpers import temporal_train_test_split

        with pytest.raises(KeyError, match="target_col"):
            temporal_train_test_split(
                temporal_df, target_col="nonexistent", cutoff_date="2023-11-01"
            )

    def test_raises_on_missing_date_col(self, temporal_df):
        from helpers import temporal_train_test_split

        with pytest.raises(KeyError, match="date_col"):
            temporal_train_test_split(
                temporal_df, target_col="is_delayed", date_col="MISSING",
                cutoff_date="2023-11-01"
            )

    def test_raises_when_train_empty(self):
        from helpers import temporal_train_test_split

        df = pd.DataFrame(
            {
                "FL_DATE": pd.to_datetime(["2023-12-01", "2023-12-15"]),
                "feature": [1, 2],
                "is_delayed": [0, 1],
            }
        )
        with pytest.raises(ValueError, match="No training rows"):
            temporal_train_test_split(
                df, target_col="is_delayed", cutoff_date="2023-01-01"
            )

    def test_raises_when_test_empty(self):
        from helpers import temporal_train_test_split

        df = pd.DataFrame(
            {
                "FL_DATE": pd.to_datetime(["2023-01-01", "2023-02-01"]),
                "feature": [1, 2],
                "is_delayed": [0, 1],
            }
        )
        with pytest.raises(ValueError, match="No test rows"):
            temporal_train_test_split(
                df, target_col="is_delayed", cutoff_date="2024-01-01"
            )

    def test_deterministic_same_cutoff_same_result(self, temporal_df):
        from helpers import temporal_train_test_split

        r1 = temporal_train_test_split(
            temporal_df, target_col="is_delayed", cutoff_date="2023-11-01"
        )
        r2 = temporal_train_test_split(
            temporal_df, target_col="is_delayed", cutoff_date="2023-11-01"
        )
        pd.testing.assert_frame_equal(r1[0], r2[0])
        pd.testing.assert_frame_equal(r1[1], r2[1])
        pd.testing.assert_series_equal(r1[2], r2[2])
        pd.testing.assert_series_equal(r1[3], r2[3])


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


# ---------------------------------------------------------------------------
# Phase 5: Time-based feature engineering tests
# ---------------------------------------------------------------------------


class TestPhase5TimeFeatures:
    """Test time-based feature engineering functions for Experiment A."""

    @pytest.fixture
    def silver_df(self):
        """Synthetic silver-like DataFrame for time feature tests."""
        return pd.DataFrame(
            {
                "FL_DATE": pd.to_datetime(
                    [
                        "2023-01-02",  # Monday
                        "2023-07-04",  # Tuesday (holiday period)
                        "2023-12-25",  # Monday (holiday period)
                        "2023-03-18",  # Saturday
                        "2023-06-11",  # Sunday
                        "2023-11-23",  # Thursday (Thanksgiving week)
                    ]
                ),
                "DEP_TIME": [630.0, 1430.0, 0.0, 2359.0, 1200.0, float("nan")],
                "AIRLINE_CODE": pd.Categorical(
                    ["AA", "UA", "DL", "WN", "AA", "UA"]
                ),
                "ORIGIN": pd.Categorical(
                    ["ATL", "ORD", "LAX", "DFW", "JFK", "ATL"]
                ),
                "DEST": pd.Categorical(
                    ["ORD", "LAX", "DFW", "JFK", "ATL", "ORD"]
                ),
                "DISTANCE": [600.0, 1700.0, 1200.0, 1400.0, 800.0, 600.0],
                "arrival_delay": [5.0, 20.0, -3.0, 45.0, 10.0, 30.0],
                "is_delayed": [0, 1, 0, 1, 0, 1],
            }
        )

    def test_engineer_time_features_hour_of_day(self, silver_df):
        from helpers import engineer_time_features

        result = engineer_time_features(silver_df)
        # 630 -> 6, 1430 -> 14, 0 -> 0, 2359 -> 23, 1200 -> 12, NaN -> -1
        assert result["hour_of_day"].iloc[0] == 6
        assert result["hour_of_day"].iloc[1] == 14
        assert result["hour_of_day"].iloc[2] == 0
        assert result["hour_of_day"].iloc[3] == 23
        assert result["hour_of_day"].iloc[4] == 12
        assert result["hour_of_day"].iloc[5] == -1  # NaN sentinel

    def test_engineer_time_features_day_of_week(self, silver_df):
        from helpers import engineer_time_features

        result = engineer_time_features(silver_df)
        # 2023-01-02 = Monday=0, 2023-07-04=Tuesday=1, 2023-12-25=Monday=0
        # 2023-03-18=Saturday=5, 2023-06-11=Sunday=6, 2023-11-23=Thursday=3
        expected = [0, 1, 0, 5, 6, 3]
        assert list(result["day_of_week"]) == expected

    def test_engineer_time_features_month(self, silver_df):
        from helpers import engineer_time_features

        result = engineer_time_features(silver_df)
        expected = [1, 7, 12, 3, 6, 11]
        assert list(result["month"]) == expected

    def test_engineer_time_features_is_weekend(self, silver_df):
        from helpers import engineer_time_features

        result = engineer_time_features(silver_df)
        # Saturday=1, Sunday=1, rest=0
        expected = [0, 0, 0, 1, 1, 0]
        assert list(result["is_weekend"]) == expected

    def test_engineer_time_features_is_holiday_period(self, silver_df):
        from helpers import engineer_time_features

        result = engineer_time_features(silver_df)
        # Jan 2 = holiday (Dec 20 - Jan 3), Jul 4 = holiday (Jun 15 - Sep 5)
        # Dec 25 = holiday (Dec 20 - Jan 3), Mar 18 = NOT holiday
        # Jun 11 = NOT holiday (< Jun 15), Nov 23 = holiday (Nov 20 - Nov 30)
        expected = [1, 1, 1, 0, 0, 1]
        assert list(result["is_holiday_period"]) == expected

    def test_engineer_time_features_time_of_day_bucket(self, silver_df):
        from helpers import engineer_time_features

        result = engineer_time_features(silver_df)
        # hour 6 -> early_morning (5-8), 14 -> afternoon (12-17), 0 -> night (22-4),
        # 23 -> night, 12 -> afternoon, -1 (NaN) -> unknown
        expected = ["early_morning", "afternoon", "night", "night", "afternoon", "unknown"]
        assert list(result["time_of_day_bucket"]) == expected

    def test_engineer_time_features_all_columns_present(self, silver_df):
        from helpers import engineer_time_features

        result = engineer_time_features(silver_df)
        expected_cols = {
            "hour_of_day",
            "day_of_week",
            "month",
            "is_weekend",
            "is_holiday_period",
            "time_of_day_bucket",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_engineer_time_features_preserves_original_columns(self, silver_df):
        from helpers import engineer_time_features

        original_cols = set(silver_df.columns)
        result = engineer_time_features(silver_df)
        assert original_cols.issubset(set(result.columns))

    def test_engineer_time_features_handles_dep_time_2400(self):
        """DEP_TIME=2400 should map to hour_of_day=0."""
        from helpers import engineer_time_features

        df = pd.DataFrame(
            {
                "FL_DATE": pd.to_datetime(["2023-06-15"]),
                "DEP_TIME": [2400.0],
                "AIRLINE_CODE": pd.Categorical(["AA"]),
                "ORIGIN": pd.Categorical(["ATL"]),
                "DEST": pd.Categorical(["ORD"]),
                "DISTANCE": [600.0],
                "arrival_delay": [5.0],
                "is_delayed": [0],
            }
        )
        result = engineer_time_features(df)
        assert result["hour_of_day"].iloc[0] == 0


# ---------------------------------------------------------------------------
# Phase 6: Route-based feature engineering tests
# ---------------------------------------------------------------------------


class TestPhase6RouteFeatures:
    """Test route-based feature engineering and frequency encoding."""

    @pytest.fixture
    def silver_df(self):
        """Synthetic silver-like DataFrame for route feature tests."""
        return pd.DataFrame(
            {
                "FL_DATE": pd.to_datetime(
                    [
                        "2023-01-01",
                        "2023-01-02",
                        "2023-01-03",
                        "2023-01-04",
                        "2023-01-05",
                        "2023-01-06",
                        "2023-01-07",
                        "2023-01-08",
                    ]
                ),
                "AIRLINE_CODE": pd.Categorical(
                    ["AA", "AA", "UA", "UA", "DL", "DL", "AA", "WN"]
                ),
                "ORIGIN": pd.Categorical(
                    ["ATL", "ATL", "ORD", "ORD", "LAX", "LAX", "ATL", "DFW"]
                ),
                "DEST": pd.Categorical(
                    ["ORD", "ORD", "LAX", "LAX", "DFW", "DFW", "ORD", "JFK"]
                ),
                "DISTANCE": [600.0, 600.0, 1700.0, 1700.0, 1200.0, 1200.0, 600.0, 1400.0],
                "arrival_delay": [5.0, 20.0, -3.0, 45.0, 10.0, 30.0, 15.0, 60.0],
                "is_delayed": [0, 1, 0, 1, 0, 1, 0, 1],
                "DEP_TIME": [800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0],
            }
        )

    def test_engineer_route_features_creates_route(self, silver_df):
        from helpers import engineer_route_features

        result = engineer_route_features(silver_df)
        # route = ORIGIN-DEST
        assert result["route"].iloc[0] == "ATL-ORD"
        assert result["route"].iloc[2] == "ORD-LAX"
        assert result["route"].iloc[7] == "DFW-JFK"

    def test_engineer_route_features_creates_distance_bucket(self, silver_df):
        from helpers import engineer_route_features

        result = engineer_route_features(silver_df)
        assert "distance_bucket" in result.columns
        # All values should be one of: short, medium, long, very_long
        valid_buckets = {"short", "medium", "long", "very_long"}
        assert set(result["distance_bucket"].unique()).issubset(valid_buckets)
        # 600 -> short, 1700 -> long, 1200 -> medium, 1400 -> long
        assert result["distance_bucket"].iloc[0] == "short"
        assert result["distance_bucket"].iloc[2] == "long"

    def test_frequency_encode_produces_correct_frequencies(self):
        from helpers import frequency_encode

        train = pd.Series(["A", "A", "A", "B", "B", "C"])
        test = pd.Series(["A", "B", "C", "D"])
        train_enc, test_enc = frequency_encode(train, test)
        # A appears 3/6=0.5, B appears 2/6=0.333, C appears 1/6=0.167
        assert abs(train_enc.iloc[0] - 0.5) < 0.01
        assert abs(train_enc.iloc[3] - 1 / 3) < 0.01
        assert abs(test_enc.iloc[0] - 0.5) < 0.01
        # D is unseen, should get 0.0
        assert test_enc.iloc[3] == 0.0

    def test_compute_delay_rates_from_train_only(self, silver_df):
        from helpers import compute_delay_rates

        # Use first 6 rows as "train"
        train = silver_df.iloc[:6].copy()
        rates = compute_delay_rates(train, group_col="AIRLINE_CODE", target_col="is_delayed")
        # AA: rows 0,1 -> is_delayed=[0,1] -> rate=0.5
        # UA: rows 2,3 -> is_delayed=[0,1] -> rate=0.5
        # DL: rows 4,5 -> is_delayed=[0,1] -> rate=0.5
        assert abs(rates["AA"] - 0.5) < 0.01
        assert abs(rates["UA"] - 0.5) < 0.01
        assert abs(rates["DL"] - 0.5) < 0.01

    def test_apply_delay_rates_with_fallback_for_unseen(self):
        from helpers import apply_delay_rates

        df = pd.DataFrame({"airline": ["AA", "UA", "UNKNOWN"]})
        rate_map = {"AA": 0.3, "UA": 0.5}
        result = apply_delay_rates(df, rate_map, col="airline", default_rate=0.25)
        assert abs(result.iloc[0] - 0.3) < 0.01
        assert abs(result.iloc[1] - 0.5) < 0.01
        assert abs(result.iloc[2] - 0.25) < 0.01  # fallback

    def test_frequency_encode_handles_all_unseen(self):
        from helpers import frequency_encode

        train = pd.Series(["A", "B"])
        test = pd.Series(["C", "D"])
        _, test_enc = frequency_encode(train, test)
        assert all(test_enc == 0.0)

    def test_engineer_route_features_preserves_original_columns(self, silver_df):
        from helpers import engineer_route_features

        original_cols = set(silver_df.columns)
        result = engineer_route_features(silver_df)
        assert original_cols.issubset(set(result.columns))
