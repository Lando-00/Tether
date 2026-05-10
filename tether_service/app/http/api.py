"""FastAPI app factory + lifespan.

Phase 3 step 35 (``p3-lifespan-slim``): bounded shutdown lives in
:class:`HardwareWatchdog`; signal handling lives in
:class:`SignalSupervisor`.

Phase 3 follow-up: SignalSupervisor installs from inside the lifespan
``__aenter__`` (post ``uvicorn.capture_signals``) so our handlers win.

Phase 4.5 follow-up (rubber-duck consensus, gpt-5.5 BLOCKING #1): the
lifespan now enters the :class:`Engine` async context before ``yield``
and exits it on shutdown. The previous body skipped ``__aenter__``, so
``WebSearchTool.startup`` never ran (every search returned
``"web_search not initialised"``) and READY connectors never
auto-started after a server restart.

Library-mode users (no FastAPI) do NOT install SignalSupervisor — they
manage signals themselves per ``WatchdogMode.LIBRARY`` contract.

Synthesis §4 Phase 3 step 35; Phase 3 + Phase 4.5 follow-ups.
"""
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from tether_service.app.http.middleware import RequestIdMiddleware
from tether_service.app.http.routers.chat import router as chat_router
from tether_service.app.http.routers.connectors import router as connectors_router
from tether_service.app.http.routers.health import router as health_router
from tether_service.app.http.routers.models import router as models_router
from tether_service.app.http.routers.protocol import router as protocol_router
from tether_service.app.http.routers.sessions import router as sessions_router
from tether_service.config.settings import load_settings
from tether_service.engine import Engine
from tether_service.runtime.signal_supervisor import SignalSupervisor
from tether_service.runtime.watchdog_mode import WatchdogMode


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: install :class:`SignalSupervisor` (post-uvicorn-capture)
    and enter the :class:`Engine` async context — runs ``startup_all``
    on tools and schedules ``start_connector`` for every READY
    connector (Phase 4 step 41 + Phase 4.5 step 47d).

    Shutdown: exit the Engine async context, which runs
    ``shutdown_all`` on tools, ``stop_all`` on connectors, then
    ``aclose`` via :class:`HardwareWatchdog`.

    Phase 4.5 follow-up (rubber-duck consensus, gpt-5.5 BLOCKING #1):
    previously the body was ``yield`` + ``aclose`` only; tools and
    connectors never started in production HTTP serving.

    The SignalSupervisor install must run from HERE (not
    ``__main__.py``) because ``uvicorn.run`` calls ``capture_signals``
    which replaces any handlers installed before it; our lifespan
    ``__aenter__`` runs AFTER that, so ``signal.signal(...)`` wins
    (Synthesis §4 Phase 3 step 35; Phase 3 follow-up).

    ``signal.signal`` only works in the main thread; production uvicorn
    is fine, but ``TestClient`` runs the lifespan in an ``anyio`` worker
    thread which raises ``ValueError`` — we swallow that case so route
    tests don't have to patch the install themselves.
    """
    try:
        SignalSupervisor(max_shutdown_sec=5.0).install()
    except ValueError:
        # Non-main thread (TestClient anyio worker, embedded interpreter).
        pass

    engine: Engine = app.state.gen_svc
    await engine.__aenter__()
    try:
        yield
    finally:
        await engine.__aexit__(None, None, None)


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

    app.add_middleware(RequestIdMiddleware)

    v1_router = APIRouter(prefix="/api/v1")
    v1_router.include_router(chat_router)
    v1_router.include_router(connectors_router)
    v1_router.include_router(health_router)
    v1_router.include_router(models_router)
    v1_router.include_router(protocol_router)
    v1_router.include_router(sessions_router)

    app.include_router(v1_router)
    return app
