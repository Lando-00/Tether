"""Phase 3 follow-up: SignalSupervisor install lives in FastAPI lifespan.

Pre-fix: ``__main__.py`` called ``SignalSupervisor.install()`` BEFORE
``uvicorn.run``. ``uvicorn.run`` calls ``Server.serve``, which calls
``capture_signals`` — which REPLACES whatever signal handlers the
process had. So our force-exit timer never fired.

Fix: install from inside the FastAPI lifespan ``__aenter__``, which
runs AFTER uvicorn's ``capture_signals``. Our handler wins.

Library-mode users (no FastAPI) do NOT install SignalSupervisor —
they manage signals themselves per ``WatchdogMode.LIBRARY`` contract.

Synthesis §4 Phase 3 step 35 + Phase 3 follow-up (rubber-duck consensus,
gpt-5.5 BLOCKING + xhigh CONCERN).
"""
from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import tether.app.__main__ as main_mod
import tether.app.http.api as api_mod
from tether.app.http.api import create_app
from tether.runtime.signal_supervisor import SignalSupervisor


@pytest.fixture(autouse=True)
def _reset_signal_supervisor_installed_flag():
    """SignalSupervisor uses a class-level ``_installed`` flag so multiple
    instances don't double-install. Tests need a clean slate so we can
    observe the install fire (or not). Reset both before AND after each
    test to avoid pollution from / to other tests in the suite.
    """
    SignalSupervisor._installed = False
    yield
    SignalSupervisor._installed = False


# ---------------------------------------------------------------------------
# A4.1 — Lifespan startup installs SignalSupervisor.
# ---------------------------------------------------------------------------


def test_lifespan_installs_signal_supervisor():
    """Starting the app via TestClient enters the lifespan, which calls
    ``SignalSupervisor(...).install()``. We monkeypatch the install
    method with a flag-setting Mock so the test doesn't actually mutate
    process-level signal handlers.
    """
    install_calls: list[bool] = []

    def fake_install(self):
        install_calls.append(True)

    # Avoid awaiting the real engine.aclose during teardown — the test
    # only cares about the install call.
    aclose_mock = AsyncMock()

    with patch.object(SignalSupervisor, "install", fake_install):
        app = create_app()
        app.state.gen_svc.aclose = aclose_mock  # type: ignore[method-assign]

        with TestClient(app):
            # Lifespan startup has run → install must have been called.
            assert len(install_calls) == 1, (
                "SignalSupervisor.install was not called during lifespan startup; "
                "uvicorn capture_signals would have replaced our handlers"
            )

    # Lifespan shutdown should have awaited engine.aclose exactly once.
    aclose_mock.assert_awaited_once()


# ---------------------------------------------------------------------------
# A4.2 — __main__.py no longer installs SignalSupervisor.
# ---------------------------------------------------------------------------


def test_main_does_not_install_signal_supervisor():
    """``__main__.py``'s ``main()`` must NOT install SignalSupervisor —
    that was the original bug (uvicorn's capture_signals replaced it).
    Source-level grep is sufficient: even an import-level reference
    would mean the symbol is somehow used here.
    """
    src = inspect.getsource(main_mod)
    assert "SignalSupervisor" not in src, (
        "tether/app/__main__.py still references SignalSupervisor; "
        "the install must live ONLY in the FastAPI lifespan startup so it "
        "runs AFTER uvicorn's capture_signals."
    )


def test_api_py_installs_signal_supervisor():
    """The other half of the contract: ``app/http/api.py`` now contains
    the SignalSupervisor install (lifespan startup). Source-level grep
    catches accidental regressions where someone moves the install back
    out of the lifespan.
    """
    src = inspect.getsource(api_mod)
    assert "SignalSupervisor" in src
    # The install call itself must be present (not just an import).
    assert "SignalSupervisor(max_shutdown_sec=5.0).install()" in src or (
        "SignalSupervisor(" in src and ").install()" in src
    ), (
        "api.py imports SignalSupervisor but doesn't appear to call install; "
        "the lifespan must install it explicitly."
    )


# ---------------------------------------------------------------------------
# A4.3 — Idempotent install across multiple lifespan starts.
# ---------------------------------------------------------------------------


def test_lifespan_idempotent_install():
    """``SignalSupervisor.install`` is idempotent (class-level
    ``_installed`` flag). Two TestClient sessions in sequence trigger
    the lifespan twice; install is *called* once per session, but only
    the first call actually mutates the flag — subsequent calls
    short-circuit at the ``if SignalSupervisor._installed`` guard.

    We patch ``install`` with a fake that mirrors the real idempotency
    contract WITHOUT calling ``signal.signal`` (which raises
    ``ValueError`` from non-main threads — TestClient runs the
    lifespan in an anyio worker). The fake counts both total
    invocations and the subset that actually mutated the flag.
    """
    aclose_mock = AsyncMock()

    install_invocations: list[bool] = []
    actual_mutations: list[bool] = []

    def fake_install(self):
        install_invocations.append(True)
        if SignalSupervisor._installed:
            return  # mirror real idempotency short-circuit
        SignalSupervisor._installed = True
        actual_mutations.append(True)

    with patch.object(SignalSupervisor, "install", fake_install):
        app1 = create_app()
        app1.state.gen_svc.aclose = aclose_mock  # type: ignore[method-assign]
        app2 = create_app()
        app2.state.gen_svc.aclose = aclose_mock  # type: ignore[method-assign]

        # First session: install called, flag set.
        with TestClient(app1):
            assert SignalSupervisor._installed is True
            assert len(install_invocations) == 1
            assert len(actual_mutations) == 1

        # Second session: install called again, but short-circuits.
        with TestClient(app2):
            assert SignalSupervisor._installed is True
            assert len(install_invocations) == 2, (
                "Lifespan should call install on every session start"
            )
            assert len(actual_mutations) == 1, (
                "SignalSupervisor.install must be idempotent — second "
                "call must not re-install (class-level flag)."
            )
