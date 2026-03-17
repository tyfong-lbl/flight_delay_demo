#!/usr/bin/env python3
"""Create project virtual environment and validate dependency imports."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys
import venv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env_setup import all_imports_ok, parse_requirements_file, requirement_to_module_name, validate_imports


LOGGER = logging.getLogger("bootstrap_env")


def _venv_python_path(venv_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def create_venv(venv_dir: Path) -> None:
    LOGGER.debug("Creating virtual environment at %s", venv_dir)
    builder = venv.EnvBuilder(with_pip=True, clear=False)
    builder.create(str(venv_dir))


def install_requirements(venv_dir: Path, requirements_path: Path) -> None:
    python_path = _venv_python_path(venv_dir)
    LOGGER.debug("Installing requirements from %s", requirements_path)
    subprocess.run(
        [str(python_path), "-m", "pip", "install", "-r", str(requirements_path)],
        check=True,
    )


def validate_imports_in_venv(venv_dir: Path, package_names: list[str]) -> int:
    python_path = _venv_python_path(venv_dir)
    module_names = [requirement_to_module_name(name) for name in package_names]
    command = [
        str(python_path),
        "-c",
        (
            "import importlib, json, sys; "
            "mods=json.loads(sys.argv[1]); "
            "fails=[]; "
            "\nfor mod in mods:\n"
            "    try:\n"
            "        importlib.import_module(mod)\n"
            "    except Exception as exc:\n"
            "        fails.append((mod, str(exc)))\n"
            "\n"
            "print(json.dumps(fails)); "
            "sys.exit(1 if fails else 0)"
        ),
        json.dumps(module_names),
    ]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        LOGGER.debug("Venv validation stderr: %s", proc.stderr.strip())
        LOGGER.debug("Venv validation failures: %s", proc.stdout.strip())
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", default=".venv", help="Virtual environment directory")
    parser.add_argument(
        "--requirements",
        default="requirements.txt",
        help="Path to requirements file",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip pip install step and only validate imports from current interpreter",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logger verbosity",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")

    venv_dir = Path(args.venv)
    requirements_path = Path(args.requirements)

    if not requirements_path.exists():
        LOGGER.error("requirements file does not exist: %s", requirements_path)
        return 2

    create_venv(venv_dir)

    if not args.skip_install:
        try:
            install_requirements(venv_dir, requirements_path)
        except subprocess.CalledProcessError as exc:
            LOGGER.error("pip install failed with return code %s", exc.returncode)
            return exc.returncode

    package_names = parse_requirements_file(requirements_path)

    if args.skip_install:
        results = validate_imports(package_names)
        failures = [result for result in results if not result.ok]
        if failures:
            for failure in failures:
                LOGGER.error(
                    "Import check failed for package=%s module=%s error=%s",
                    failure.package_name,
                    failure.module_name,
                    failure.error,
                )
            return 1

        if all_imports_ok(results):
            LOGGER.info("Validated %d dependency imports in current interpreter", len(results))

    # Also validate imports against the created virtual environment interpreter.
    venv_rc = validate_imports_in_venv(venv_dir, package_names)
    if venv_rc != 0:
        LOGGER.error("Import validation failed under virtual environment interpreter")
        return venv_rc

    LOGGER.info("Virtual environment import validation succeeded for %d packages", len(package_names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
