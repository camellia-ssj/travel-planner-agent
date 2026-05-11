"""Development-time shim for the src-layout package.

This makes `travel_agent.*` importable from the repository root before the
package is installed into the active environment.
"""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path


__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

SRC_PACKAGE_DIR = Path(__file__).resolve().parent.parent / "src" / "travel_agent"
if SRC_PACKAGE_DIR.is_dir():
    src_package_text = str(SRC_PACKAGE_DIR)
    if src_package_text not in __path__:
        __path__.append(src_package_text)
