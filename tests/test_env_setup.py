from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from env_setup import (
    all_imports_ok,
    parse_requirement_name,
    parse_requirements_file,
    requirement_to_module_name,
    validate_imports,
)


class EnvSetupTests(unittest.TestCase):
    def test_parse_requirement_name(self) -> None:
        cases: list[tuple[str, str | None]] = [
            ("", None),
            ("   ", None),
            ("# comment", None),
            ("--index-url https://example.invalid/simple", None),
            ("numpy", "numpy"),
            ("pandas>=2.0", "pandas"),
            ("scikit-learn>=1.3", "scikit-learn"),
            ("matplotlib[dev]>=3.7", "matplotlib"),
            ("seaborn>=0.12 # inline comment", "seaborn"),
            ("xgboost>=2.0; python_version >= '3.11'", "xgboost"),
        ]
        for line, expected in cases:
            with self.subTest(line=line):
                self.assertEqual(parse_requirement_name(line), expected)

    def test_parse_requirements_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            req = Path(tmp) / "requirements.txt"
            req.write_text(
                """
# base deps
lakefs>=0.5.0
numpy
scikit-learn>=1.3
--index-url https://example.invalid/simple
                """.strip()
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_requirements_file(req), ["lakefs", "numpy", "scikit-learn"])

    def test_requirement_to_module_name(self) -> None:
        cases = [
            ("scikit-learn", "sklearn"),
            ("my-package", "my_package"),
            ("numpy", "numpy"),
        ]
        for package, module in cases:
            with self.subTest(package=package):
                self.assertEqual(requirement_to_module_name(package), module)

    def test_validate_imports_success_and_failure(self) -> None:
        results = validate_imports(["json", "this-package-should-not-exist"])
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].ok)
        self.assertFalse(results[1].ok)
        self.assertFalse(all_imports_ok(results))


if __name__ == "__main__":
    unittest.main()
