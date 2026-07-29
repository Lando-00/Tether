"""Phase 8 RD Fix 3 — console-script entry-point gating.

The ``tether-server`` and ``tether-cli`` console scripts ship in the
**base** wheel but their underlying implementations require the optional
``server`` / ``cli`` extras. The :mod:`tether.app._entry` and
:mod:`tether.cli._entry` shims must:

1. Defer importing the heavy entry-point module to call-time (so a missing
   extra surfaces only when the user actually invokes the script).
2. Catch ``ImportError`` and print a clear ``pip install tether[<extra>]``
   remediation message to stderr.
3. ``sys.exit(1)`` instead of letting the traceback escape.

The simplest reliable verification across environments is **static**:

* The shim modules are import-clean even when the heavy deps are absent.
* The ``import`` of the heavy entry point lives inside :func:`main`
  (verified via :func:`inspect.getsource`), not at module top level.

We then dynamically simulate the missing extra by patching ``sys.modules``
+ shadowing the heavy entry-point modules with a sentinel that raises
``ImportError`` on attribute access, and assert the shim emits the
expected remediation text.
"""
from __future__ import annotations

import builtins
import inspect
import sys

import pytest

import tether.app._entry as server_entry
import tether.cli._entry as cli_entry

# ---------------------------------------------------------------------------
# Static checks: heavy import lives INSIDE main()
# ---------------------------------------------------------------------------


def test_server_entry_main_is_callable() -> None:
    """Sanity: the shim exposes a zero-arg ``main`` callable."""
    assert callable(server_entry.main)


def test_cli_entry_main_is_callable() -> None:
    """Sanity: the shim exposes a zero-arg ``main`` callable."""
    assert callable(cli_entry.main)


def test_server_entry_imports_heavy_dep_inside_main() -> None:
    """``tether.app.__main__`` must NOT be a module-level import.

    If the heavy import is at the top of the shim, the shim itself becomes
    unimportable when the ``server`` extra is missing — defeating the
    gating's whole point.
    """
    src = inspect.getsource(server_entry)
    # Module-level imports look like "from tether.app.__main__ import" with
    # no leading whitespace. Inside main() the import is indented.
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("from tether.app.__main__", "import tether.app.__main__")):
            assert line != stripped, (
                "Heavy import 'tether.app.__main__' is at module level in "
                "_entry.py — must be inside main() so missing extras don't "
                "break the shim itself.\nLine: " + repr(line)
            )


def test_cli_entry_imports_heavy_dep_inside_main() -> None:
    """``tether.cli.main`` must NOT be a module-level import (see server twin)."""
    src = inspect.getsource(cli_entry)
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("from tether.cli.main", "import tether.cli.main")):
            assert line != stripped, (
                "Heavy import 'tether.cli.main' is at module level in "
                "_entry.py — must be inside main() so missing extras don't "
                "break the shim itself.\nLine: " + repr(line)
            )


# ---------------------------------------------------------------------------
# Dynamic checks: simulate ImportError, assert remediation message + exit
# ---------------------------------------------------------------------------


def _install_failing_import(
    monkeypatch: pytest.MonkeyPatch, target_modname: str, missing_pkg: str
) -> None:
    """Force ``import <target_modname>`` to raise ``ImportError(missing_pkg)``.

    Wraps ``builtins.__import__`` so the failure is observable from inside
    ``_entry.main``'s ``try: from <target> import ...`` block. Other
    imports pass through untouched.
    """
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == target_modname:
            raise ImportError(f"No module named {missing_pkg!r}")
        return real_import(name, globals, locals, fromlist, level)

    # Drop any cached version of the target so the import machinery has to
    # re-execute the import (and hit our failing __import__).
    monkeypatch.delitem(sys.modules, target_modname, raising=False)
    monkeypatch.setattr(builtins, "__import__", _fake_import)


def test_server_entry_prints_remediation_when_extra_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing ``server`` extra → stderr says ``pip install tether[server]``."""
    _install_failing_import(monkeypatch, "tether.app.__main__", "fastapi")

    with pytest.raises(SystemExit) as ei:
        server_entry.main()
    assert ei.value.code == 1

    captured = capsys.readouterr()
    assert "pip install tether[server]" in captured.err
    assert "tether-server requires the 'server' optional dependencies" in captured.err


def test_cli_entry_prints_remediation_when_extra_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing ``cli`` extra → stderr says ``pip install tether[cli]``."""
    _install_failing_import(monkeypatch, "tether.cli.main", "typer")

    with pytest.raises(SystemExit) as ei:
        cli_entry.main()
    assert ei.value.code == 1

    captured = capsys.readouterr()
    assert "pip install tether[cli]" in captured.err
    assert "tether-cli requires the 'cli' optional dependencies" in captured.err
