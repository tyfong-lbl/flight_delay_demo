"""Environment bootstrap and dependency validation utilities.

This module provides small, testable building blocks for:
- parsing package names from ``requirements.txt``;
- mapping package names to import module names;
- validating imports with structured debug logging.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
import re
from pathlib import Path


LOGGER = logging.getLogger(__name__)


IMPORT_NAME_OVERRIDES = {
    "scikit-learn": "sklearn",
}


_VERSION_SPLIT_RE = re.compile(r"[<>=!~]")


@dataclass(frozen=True)
class ImportCheckResult:
    """Result for a single package import validation check."""

    package_name: str
    module_name: str
    ok: bool
    error: str | None = None


def parse_requirement_name(requirement_line: str) -> str | None:
    """Extract a normalized package name from one requirements line.

    Returns ``None`` for empty lines, comments, and unsupported pip directives.
    """

    line = requirement_line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith(("-", "--")):
        LOGGER.debug("Skipping requirement directive line: %s", line)
        return None

    no_inline_comment = line.split("#", 1)[0].strip()
    if not no_inline_comment:
        return None

    # Strip environment markers (e.g. '; python_version > "3.10"').
    no_marker = no_inline_comment.split(";", 1)[0].strip()
    if not no_marker:
        return None

    # Remove version constraints.
    base = _VERSION_SPLIT_RE.split(no_marker, maxsplit=1)[0].strip()
    if not base:
        return None

    # Remove extras (e.g. package[extra]).
    package = base.split("[", 1)[0].strip().lower()
    return package or None


def parse_requirements_file(path: str | Path) -> list[str]:
    """Parse and return normalized package names from a requirements file."""

    requirements_path = Path(path)
    packages: list[str] = []
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_requirement_name(line)
        if parsed:
            packages.append(parsed)

    LOGGER.debug("Parsed %d packages from %s", len(packages), requirements_path)
    return packages


def requirement_to_module_name(package_name: str) -> str:
    """Map a package requirement name to its importable module name."""

    normalized = package_name.strip().lower()
    if normalized in IMPORT_NAME_OVERRIDES:
        return IMPORT_NAME_OVERRIDES[normalized]
    return normalized.replace("-", "_")


def validate_imports(package_names: list[str]) -> list[ImportCheckResult]:
    """Validate imports for package names and return per-package results."""

    results: list[ImportCheckResult] = []

    for package_name in package_names:
        module_name = requirement_to_module_name(package_name)
        LOGGER.debug(
            "Validating dependency import", extra={"package": package_name, "module": module_name}
        )
        try:
            importlib.import_module(module_name)
            results.append(
                ImportCheckResult(
                    package_name=package_name,
                    module_name=module_name,
                    ok=True,
                )
            )
        except Exception as exc:  # pragma: no cover - exception shape is runtime-dependent
            LOGGER.debug(
                "Dependency import failed",
                extra={"package": package_name, "module": module_name, "error": str(exc)},
            )
            results.append(
                ImportCheckResult(
                    package_name=package_name,
                    module_name=module_name,
                    ok=False,
                    error=str(exc),
                )
            )

    return results


def all_imports_ok(results: list[ImportCheckResult]) -> bool:
    """Return ``True`` if all import checks succeeded."""

    return all(result.ok for result in results)
