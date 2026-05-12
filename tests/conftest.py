from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Give each test run an isolated base temp directory on Windows.

    Reusing a single ``--basetemp`` directory made repeat runs flaky when
    SQLite / Chroma files were still being released by the OS.
    """

    if getattr(config.option, "basetemp", None):
        return

    project_root = Path(__file__).resolve().parents[1]
    base_root = project_root / "data" / "pytest_tmp"
    base_root.mkdir(parents=True, exist_ok=True)

    existing_runs = sorted(
        (path for path in base_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for old_run in existing_runs[5:]:
        shutil.rmtree(old_run, ignore_errors=True)

    run_dir = base_root / f"run-{uuid.uuid4().hex}"
    run_dir.mkdir()
    config.option.basetemp = str(run_dir)
