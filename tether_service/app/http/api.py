"""FastAPI app factory + lifespan.

Phase 3 step 35 (``p3-lifespan-slim``): the lifespan body is now just
``yield`` then ``await engine.aclose()`` — bounded shutdown lives in
:meth:`tether_service.runtime.hw_watchdog.HardwareWatchdog.shutdown_all`,
signal handling lives in
:class:`tether_service.runtime.signal_supervisor.SignalSupervisor`.

Phase 3 follow-up (rubber-duck consensus, gpt-5.5 BLOCKING + xhigh
CONCERN): the SignalSupervisor install moved from ``__main__.py`` INTO
the lifespan startup. Why: ``uvicorn.run`` calls ``Server.serve``,
which calls ``capture_signals`` that REPLACES whatever signal handlers
the process had. If we install in ``__main__.py`` BEFORE
``uvicorn.run``, uvicorn overwrites our handlers and the force-exit
timer never fires. Installing from inside the lifespan ``__aenter__``
runs AFTER ``capture_signals``, so our ``signal.signal(...)`` runs
LAST and wins.

Library-mode users (no FastAPI) do NOT install SignalSupervisor — they
manage signals themselves per ``WatchdogMode.LIBRARY`` contract.

Synthesis §4 Phase 3 step 35; Phase 3 follow-up.
"""
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from tether_service.app.http.routers.chat import router as chat_router
from tether_service.app.http.routers.connectors import router as connectors_router
from tether_service.app.http.routers.health import router as health_router
from tether_service.app.http.routers.models import router as models_router
from tether_service.app.http.routers.sessions import router as sessions_router
from tether_service.config.settings import load_settings
from tether_service.engine import Engine
from tether_service.runtime.signal_supervisor import SignalSupervisor
from tether_service.runtime.watchdog_mode import WatchdogMode


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup installs :class:`SignalSupervisor` (post-uvicorn-capture
    so our force-exit timer can actually fire); shutdown delegates to
    :meth:`Engine.aclose` which routes through :class:`HardwareWatchdog`.

    The install must run from HERE (not ``__main__.py``) because
    ``uvicorn.run`` calls ``capture_signals`` which replaces any
    handlers installed before it. Our lifespan ``__aenter__`` runs
    AFTER that, so ``signal.signal(...)`` wins. Synthesis §4 Phase 3
    step 35; Phase 3 follow-up rubber-duck consensus.

    :class:`SignalSupervisor.install` is idempotent (class-level
    ``_installed`` flag), so multiple lifespan starts in tests don't
    double-install.

    Note: ``signal.signal`` only works in the main thread of the main
    interpreter. Production uvicorn runs the lifespan in the main
    thread (where ``capture_signals`` ran), so the install always
    succeeds. ``starlette.testclient.TestClient`` runs the lifespan in
    an ``anyio`` worker thread, which causes ``signal.signal`` to raise
    ``ValueError``. We swallow that specific case so existing
    integration tests using ``TestClient`` don't have to patch the
    install themselves; tests that need to *verify* the install fires
    should patch it explicitly (see ``test_lifespan_signal_install.py``).
    """
    try:
        SignalSupervisor(max_shutdown_sec=5.0).install()
    except ValueError:
        # Non-main thread (TestClient anyio worker, embedded interpreter,
        # etc.) — signal handlers can't be installed here; in production
        # this branch is unreachable. Phase 3 follow-up.
        pass
    yield
    engine: Engine = app.state.gen_svc
    await engine.aclose()


def create_app():
    """Create and configure the FastAPI application.

    Builds the Engine with :attr:`WatchdogMode.SERVER` because the HTTP
    entry point is the canonical SERVER-mode caller; library users go
    through ``Engine.from_settings`` directly with the LIBRARY default.
    Per _synthesis.md §4 Phase 3 step 35.
    """
    settings_v2 = load_settings()
    gen_service = Engine.from_settings(settings_v2, watchdog_mode=WatchdogMode.SERVER)

    app = FastAPI(lifespan=lifespan)
    app.state.gen_svc = gen_service
    app.state.settings = settings_v2

    v1_router = APIRouter(prefix="/api/v1")
    v1_router.include_router(chat_router)
    v1_router.include_router(connectors_router)
    v1_router.include_router(health_router)
    v1_router.include_router(models_router)
    v1_router.include_router(sessions_router)

    app.include_router(v1_router)
    return app
