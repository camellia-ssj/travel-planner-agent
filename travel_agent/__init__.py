"""Development-time shim for the src-layout package.

This makes ``travel_agent.*`` importable from the repository root before the
package is installed into the active environment.

**IMPORTANT**: the repo's ``src/travel_agent`` is inserted at the **front** of
``__path__`` so it always wins over any site-packages copy (e.g. from an older
``pip install -e .``).  Without this guarantee, ``python -m travel_agent...``
can silently run stale installed code instead of the current working-tree.
"""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

SRC_PACKAGE_DIR = Path(__file__).resolve().parent.parent / "src" / "travel_agent"
if SRC_PACKAGE_DIR.is_dir():
    src_package_text = str(SRC_PACKAGE_DIR)
    # Remove any existing entry first, then insert at front so it wins.
    if src_package_text in __path__:
        __path__.remove(src_package_text)
    __path__.insert(0, src_package_text)
