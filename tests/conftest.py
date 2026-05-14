from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """在 Windows 上为每次测试运行提供独立的临时基础目录。

    重用单个 ``--basetemp`` 目录会导致重复运行时出现不稳定，
    因为 SQLite / Chroma 文件可能仍被操作系统占用。
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
