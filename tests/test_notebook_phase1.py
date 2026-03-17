from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from notebook_setup import (
    _load_lakectl_config,
    _strip_api_suffix,
    build_notebook_config,
    initialize_lakefs_repository,
)


class NotebookPhase1Tests(unittest.TestCase):
    def test_notebook_has_required_phase1_sections(self) -> None:
        with open("01_data_prep.ipynb", "r", encoding="utf-8") as notebook_file:
            notebook = json.load(notebook_file)

        markdown_cells = [
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "markdown"
        ]

        expected_headers = [
            "## Section 1.1 - Setup & Configuration",
            "## Section 1.2 - Bronze Layer (main branch)",
            "## Section 1.3 - EDA (Exploratory Data Analysis)",
            "## Section 1.4 - Silver Layer (silver branch)",
        ]
        for header in expected_headers:
            self.assertTrue(
                any(header in cell_text for cell_text in markdown_cells),
                msg=f"Missing markdown section header: {header}",
            )

    def test_notebook_contains_configuration_and_connection_cells(self) -> None:
        with open("01_data_prep.ipynb", "r", encoding="utf-8") as notebook_file:
            notebook = json.load(notebook_file)

        code_cells = [
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        ]

        self.assertTrue(
            any("build_notebook_config" in source for source in code_cells),
            msg="Notebook is missing config builder usage cell",
        )
        self.assertTrue(
            any("initialize_lakefs_repository" in source for source in code_cells),
            msg="Notebook is missing lakeFS initialization cell",
        )
        self.assertTrue(
            any("sample_size_cap" in source.lower() for source in code_cells),
            msg="Notebook configuration cell should expose sample size cap",
        )
        self.assertTrue(
            any("random_seed" in source.lower() for source in code_cells),
            msg="Notebook configuration cell should expose random seed",
        )


class StripApiSuffixTests(unittest.TestCase):
    def test_strips_api_v1(self) -> None:
        self.assertEqual(
            _strip_api_suffix("https://lakefs.bk.lbl.gov/api/v1"),
            "https://lakefs.bk.lbl.gov",
        )

    def test_strips_api_v1_with_trailing_slash(self) -> None:
        self.assertEqual(
            _strip_api_suffix("https://lakefs.bk.lbl.gov/api/v1/"),
            "https://lakefs.bk.lbl.gov",
        )

    def test_no_suffix_unchanged(self) -> None:
        self.assertEqual(
            _strip_api_suffix("https://lakefs.bk.lbl.gov"),
            "https://lakefs.bk.lbl.gov",
        )


class LoadLakectlConfigTests(unittest.TestCase):
    def test_raises_when_file_missing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            _load_lakectl_config(Path("/nonexistent/path/lakectl.yaml"))

    def test_parses_valid_yaml(self) -> None:
        yaml_content = (
            "credentials:\n"
            "  access_key_id: TESTKEY\n"
            "  secret_access_key: TESTSECRET\n"
            "server:\n"
            "  endpoint_url: https://example.com/api/v1\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp:
            tmp.write(yaml_content)
            tmp.flush()
            result = _load_lakectl_config(Path(tmp.name))
        os.unlink(tmp.name)

        self.assertEqual(result["access_key"], "TESTKEY")
        self.assertEqual(result["secret_key"], "TESTSECRET")
        self.assertEqual(result["endpoint"], "https://example.com")

    def test_raises_on_malformed_yaml(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp:
            tmp.write(": : : bad yaml [[[")
            tmp.flush()
            with self.assertRaises(RuntimeError):
                _load_lakectl_config(Path(tmp.name))
        os.unlink(tmp.name)


class NotebookSetupUnitTests(unittest.TestCase):
    """Test build_notebook_config with various config sources."""

    # Use a nonexistent yaml path so tests are isolated from the real config.
    _NO_YAML = Path("/nonexistent/lakectl.yaml")

    def test_build_notebook_config_raises_without_yaml_or_env(self) -> None:
        """With no yaml file and no env vars, should raise FileNotFoundError."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(FileNotFoundError):
                build_notebook_config(lakectl_config_path=self._NO_YAML)

    def test_build_notebook_config_reads_from_yaml(self) -> None:
        """Values from lakectl.yaml are used when env vars are absent."""
        yaml_content = (
            "credentials:\n"
            "  access_key_id: YAMLKEY\n"
            "  secret_access_key: YAMLSECRET\n"
            "server:\n"
            "  endpoint_url: https://my-lakefs.example.com/api/v1\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp:
            tmp.write(yaml_content)
            tmp.flush()
            yaml_path = Path(tmp.name)

        with patch.dict(os.environ, {}, clear=True):
            config = build_notebook_config(lakectl_config_path=yaml_path)
        os.unlink(tmp.name)

        self.assertEqual(config.endpoint, "https://my-lakefs.example.com")
        self.assertEqual(config.access_key, "YAMLKEY")
        self.assertEqual(config.secret_key, "YAMLSECRET")

    def test_build_notebook_config_env_overrides_yaml(self) -> None:
        """Environment variables take precedence over lakectl.yaml values."""
        yaml_content = (
            "credentials:\n"
            "  access_key_id: YAMLKEY\n"
            "  secret_access_key: YAMLSECRET\n"
            "server:\n"
            "  endpoint_url: https://yaml-server.example.com/api/v1\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp:
            tmp.write(yaml_content)
            tmp.flush()
            yaml_path = Path(tmp.name)

        with patch.dict(
            os.environ,
            {
                "LAKEFS_ENDPOINT": "http://env-override.local:8000",
                "LAKEFS_ACCESS_KEY": "ENVKEY",
                "LAKEFS_SECRET_KEY": "ENVSECRET",
            },
            clear=True,
        ):
            config = build_notebook_config(lakectl_config_path=yaml_path)
        os.unlink(tmp.name)

        self.assertEqual(config.endpoint, "http://env-override.local:8000")
        self.assertEqual(config.access_key, "ENVKEY")
        self.assertEqual(config.secret_key, "ENVSECRET")

    def test_build_notebook_config_env_vars_skip_yaml(self) -> None:
        """When all three required env vars are set, yaml file is not needed."""
        with patch.dict(
            os.environ,
            {
                "LAKEFS_ENDPOINT": "http://lakefs.local:8000",
                "LAKEFS_ACCESS_KEY": "key",
                "LAKEFS_SECRET_KEY": "secret",
                "LAKEFS_REPO": "repo-x",
                "LAKEFS_STORAGE_NAMESPACE": "s3://bucket/prefix",
                "RAW_CSV_PATH": "data/raw/custom.csv",
                "SAMPLE_SIZE_CAP": "not-an-int",
                "RANDOM_SEED": "123",
            },
            clear=True,
        ):
            config = build_notebook_config(lakectl_config_path=self._NO_YAML)

        self.assertEqual(config.endpoint, "http://lakefs.local:8000")
        self.assertEqual(config.access_key, "key")
        self.assertEqual(config.secret_key, "secret")
        self.assertEqual(config.repo_name, "repo-x")
        self.assertEqual(config.storage_namespace, "s3://bucket/prefix")
        self.assertEqual(config.raw_csv_path, "data/raw/custom.csv")
        self.assertEqual(config.sample_size_cap, 500_000)
        self.assertEqual(config.random_seed, 123)

    @patch("notebook_setup.importlib.import_module")
    def test_initialize_lakefs_repository_with_creation(self, import_module_mock: MagicMock) -> None:
        fake_lakefs = MagicMock()
        fake_client = object()
        fake_repository = MagicMock()

        fake_lakefs.Client.return_value = fake_client
        fake_lakefs.Repository.return_value = fake_repository
        fake_repository.branches.return_value = [MagicMock(id="main")]
        import_module_mock.return_value = fake_lakefs

        with patch.dict(
            os.environ,
            {
                "LAKEFS_ENDPOINT": "http://lakefs.local:8000",
                "LAKEFS_ACCESS_KEY": "key",
                "LAKEFS_SECRET_KEY": "secret",
                "LAKEFS_REPO": "repo-x",
                "LAKEFS_STORAGE_NAMESPACE": "s3://bucket/prefix",
            },
            clear=True,
        ):
            config = build_notebook_config(lakectl_config_path=self._NO_YAML)

        result = initialize_lakefs_repository(config)
        self.assertTrue(result.connected)
        self.assertEqual(result.repository, "repo-x")
        self.assertTrue(result.repository_created)
        self.assertTrue(result.default_branch_exists)

        fake_lakefs.Client.assert_called_once_with(
            host="http://lakefs.local:8000",
            username="key",
            password="secret",
        )
        fake_lakefs.Repository.assert_called_once_with("repo-x", client=fake_client)
        fake_repository.create.assert_called_once()

    @patch("notebook_setup.importlib.import_module", side_effect=ImportError("no module"))
    def test_initialize_lakefs_repository_import_failure(self, _: MagicMock) -> None:
        with patch.dict(
            os.environ,
            {
                "LAKEFS_ENDPOINT": "http://lakefs.local:8000",
                "LAKEFS_ACCESS_KEY": "key",
                "LAKEFS_SECRET_KEY": "secret",
            },
            clear=True,
        ):
            config = build_notebook_config(lakectl_config_path=self._NO_YAML)

        with self.assertRaises(RuntimeError):
            initialize_lakefs_repository(config)


if __name__ == "__main__":
    unittest.main()
