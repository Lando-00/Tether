"""FastAPI app factory + lifespan.

Phase 3 step 35 (this PR, ``p3-lifespan-slim``): the lifespan body is now
just ``yield`` then ``await engine.aclose()`` — bounded shutdown lives in
:meth:`tether_service.runtime.hw_watchdog.HardwareWatchdog.shutdown_all`,
signal handling lives in
:class:`tether_service.runtime.signal_supervisor.SignalSupervisor` (installed
from ``__main__.py`` so library users opt out by construction). Synthesis
§4 Phase 3 step 35.
"""
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from tether_service.app.http.routers.chat import router as chat_router
from tether_service.app.http.routers.health import router as health_router
from tether_service.app.http.routers.models import router as models_router
from tether_service.app.http.routers.sessions import router as sessions_router
from tether_service.config.settings import load_settings
from tether_service.engine import Engine
from tether_service.runtime.watchdog_mode import WatchdogMode


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup is a no-op (Engine.from_settings already wired in
    :func:`create_app`); shutdown delegates to :meth:`Engine.aclose` which
    routes through :class:`HardwareWatchdog`. Synthesis §4 Phase 3 step 35.
    """
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
    v1_router.include_router(health_router)
    v1_router.include_router(models_router)
    v1_router.include_router(sessions_router)

    app.include_router(v1_router)
    return app
