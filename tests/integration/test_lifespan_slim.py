"""Phase 3 step 35: verify the slim lifespan in ``app/http/api.py``.

After ``p3-lifespan-slim``, ``api.py`` is ~60 LOC: lifespan does
``yield`` then ``await engine.aclose()``; the legacy
``_install_force_exit_handler``, ``handle_thread_exception``,
``_asyncio_exc_handler``, and ``shutdown_provider_with_timeout`` are gone.

Synthesis §4 Phase 3 step 35.
"""
from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import tether_service.app.http.api as api_mod
from tether_service.app.http.api import create_app
from tether_service.runtime.watchdog_mode import WatchdogMode


def test_lifespan_calls_engine_aclose():
    """TestClient app start/stop: engine.aclose must be awaited exactly once."""
    app = create_app()
    # Patch aclose on the actual engine instance the app holds.
    aclose_mock = AsyncMock()
    app.state.gen_svc.aclose = aclose_mock  # type: ignore[method-assign]

    with TestClient(app):
        # Just enter + exit the lifespan.
        pass

    aclose_mock.assert_awaited_once()


def test_create_app_uses_server_mode():
    """create_app() returns app with watchdog_mode=WatchdogMode.SERVER.
    SERVER mode is the contract for the HTTP entry point."""
    app = create_app()
    engine = app.state.gen_svc
    assert engine.watchdog_mode is WatchdogMode.SERVER


def test_create_app_no_force_exit_handler():
    """The legacy helpers must be deleted from api.py source.

    Each of these names was a free-floating helper in the pre-Phase 3
    api.py; SignalSupervisor + HardwareWatchdog supersede all of them.
    A grep returning ZERO matches for each name is the contract.
    """
    src = inspect.getsource(api_mod)
    forbidden = [
        "_install_force_exit_handler",
        "handle_thread_exception",
        "_asyncio_exc_handler",
        "shutdown_provider_with_timeout",
        "_force_cleanup_provider_sync",
    ]
    for needle in forbidden:
        assert needle not in src, (
            f"api.py still contains legacy symbol {needle!r}; expected to be "
            f"removed in Phase 3 step 35."
        )


def test_api_py_loc():
    """api.py size cap: target ~50 lines, allow up to 160.

    The pre-Phase 3 api.py was 214 LOC (most of it dead helpers we just
    deleted). If this drifts back over the cap, the slim contract has
    been violated.

    The cap is a regression fence against legacy helper re-introduction
    (see :func:`test_create_app_no_force_exit_handler`), NOT a freeze on
    legitimate evolution. Each new HTTP router that ships in v1 adds two
    lines (one import + one ``include_router`` call); the cap is sized to
    accommodate that growth without losing the regression signal. The
    Phase 5 ``p5-event-types-schema-endpoint`` step added the
    ``protocol`` router (introspection endpoints), bumping the cap from
    100 → 110. Phase 7 step 79 added three conditional middleware blocks
    (CSRF / CORS / TrustedHost), bumping from 110 → 160.
    """
    src = inspect.getsource(api_mod)
    line_count = len(src.splitlines())
    assert line_count <= 160, (
        f"api.py is {line_count} lines; expected <=160 (target ~50). "
        "The lifespan slim contract has drifted."
    )


def test_api_py_no_dead_imports():
    """The legacy imports (gc, signal, threading, os, asyncio for
    set_exception_handler, cast, ModelProvider, StreamParser, SessionStore)
    were only used by deleted helpers. They must be gone."""
    src = inspect.getsource(api_mod)
    forbidden = [
        "import gc",
        "import signal",
        "import threading",
        "import os",
        "from typing import cast",
        "from tether_service.core.interfaces import ModelProvider",
        "from tether_service.core.interfaces import StreamParser",
        "from tether_service.core.interfaces import SessionStore",
    ]
    for needle in forbidden:
        assert needle not in src, (
            f"api.py still imports legacy symbol: {needle!r}"
        )
