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
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, FastAPI

from tether.app.http.middleware import RequestIdMiddleware
from tether.app.http.routers.chat import router as chat_router
from tether.app.http.routers.connectors import router as connectors_router
from tether.app.http.routers.debug import router as debug_router
from tether.app.http.routers.health import router as health_router
from tether.app.http.routers.models import router as models_router
from tether.app.http.routers.protocol import router as protocol_router
from tether.app.http.routers.sessions import router as sessions_router
from tether.app.http.routers.tools import router as tools_router
from tether.config.settings import Settings, load_settings
from tether.engine import Engine
from tether.runtime.signal_supervisor import SignalSupervisor
from tether.runtime.watchdog_mode import WatchdogMode


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


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Builds the Engine with :attr:`WatchdogMode.SERVER` because the HTTP
    entry point is the canonical SERVER-mode caller; library users go
    through ``Engine.from_settings`` directly with the LIBRARY default.
    Per _synthesis.md §4 Phase 3 step 35.

    Args:
        settings: Optional pre-built Settings object. When None (default),
            :func:`load_settings` is called to load from YAML + env.
            Pass an explicit instance in tests to avoid file I/O and to
            configure specific security policies (Phase 7 step 79).
    """
    settings_v2 = settings if settings is not None else load_settings()
    gen_service = Engine.from_settings(settings_v2, watchdog_mode=WatchdogMode.SERVER)

    app = FastAPI(lifespan=lifespan)
    app.state.gen_svc = gen_service
    app.state.settings = settings_v2

    # Middleware order: last-added = outermost in Starlette's execution model.
    # Desired runtime order:
    #   RequestId (outermost) → TrustedHost → CORS → RequireJsonContentType
    #   → CSRF → handler
    # So we add: CSRF first (innermost), then RequireJsonContentType,
    # then CORS, then TrustedHost, then RequestId LAST (outermost overall).
    # This means TrustedHost rejects bad-Host requests with 400 BEFORE other
    # security middlewares run; the Content-Type 415 reject (Phase 9 P0-B2,
    # Tribunal §3 P0-04) fires before CSRF so a missing/wrong CT never
    # reaches the CSRF check; and EVERY response — 400 (TrustedHost),
    # 415 (Content-Type), 403 (CSRF), CORS preflight, 200 — passes back
    # through RequestIdMiddleware and carries X-Request-ID for correlation.
    # Phase 7 step 68 + step 79 + RD followup (FIX 1) + Phase 9 P0-B2.
    if settings_v2.security.csrf_token.enabled:
        from tether.app.http.csrf_middleware import CSRFTokenMiddleware
        app.add_middleware(CSRFTokenMiddleware, settings=settings_v2.security.csrf_token)

    from tether.app.http.content_type_middleware import RequireJsonContentTypeMiddleware
    app.add_middleware(RequireJsonContentTypeMiddleware)

    if settings_v2.security.cors.enabled:
        from fastapi.middleware.cors import CORSMiddleware
        cors_cfg = settings_v2.security.cors
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_cfg.allow_origins,
            allow_methods=cors_cfg.allow_methods,
            allow_headers=cors_cfg.allow_headers,
            allow_credentials=cors_cfg.allow_credentials,
        )

    if settings_v2.security.trusted_host.enabled:
        from fastapi.middleware.trustedhost import TrustedHostMiddleware
        th_cfg = settings_v2.security.trusted_host
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(th_cfg.allowed_hosts))

    app.add_middleware(RequestIdMiddleware)

    v1_router = APIRouter(prefix="/api/v1")
    v1_router.include_router(chat_router)
    v1_router.include_router(connectors_router)
    v1_router.include_router(debug_router)
    v1_router.include_router(health_router)
    v1_router.include_router(models_router)
    v1_router.include_router(protocol_router)
    v1_router.include_router(sessions_router)
    v1_router.include_router(tools_router)

    app.include_router(v1_router)
    return app
