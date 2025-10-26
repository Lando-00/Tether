from typing import cast

from fastapi import APIRouter, FastAPI

from tether_service.app.http.routers.chat import router as chat_router
from tether_service.app.http.routers.health import router as health_router
from tether_service.app.http.routers.models import router as models_router
from tether_service.app.http.routers.sessions import router as sessions_router
from tether_service.core.interfaces import ModelProvider
from tether_service.core.interfaces import StreamParser, SessionStore


def create_app():
    """Create and configure the FastAPI application with DI"""
    from tether_service.core.config import load_settings
    from tether_service.core.factory import load
    from tether_service.protocol.service.generation_service import GenerationService

    settings = load_settings()
    # instantiate model provider
    model_cfg = settings.get('providers', {}).get('model', {})
    provider = cast(ModelProvider, load(model_cfg.get('impl', ''), **model_cfg.get('args', {}) or {}))
    # instantiate parser (could be function)
    parser_cfg = settings.get('providers', {}).get('parser', {})
    parser = cast(StreamParser, load(parser_cfg.get('impl', ''), **parser_cfg.get('args', {}) or {}))
    # instantiate session store
    store_cfg = settings.get('providers', {}).get('session_store', {})
    session_store = cast(SessionStore, load(store_cfg.get('impl', ''), **store_cfg.get('args', {}) or {}))
    # build tools registry
    from tether_service.core.tool_registry import ToolRegistry
    tools_cfg = settings.get('tools', {})
    registry_cfg = tools_cfg.get('registry', [])
    enabled_tools = tools_cfg.get('enabled', [])
    tool_registry = ToolRegistry(registry_cfg, enabled_tools)
    tools = tool_registry.all()

    # Get system prompt
    system_prompt = settings.get("system", {}).get("prompt", "")

    # create service
    gen_service = GenerationService(
        provider,
        parser=parser,
        session_store=session_store,
        tools=tools,
        system_prompt=system_prompt,
    )
    app = FastAPI()
    # store service on app state
    app.state.gen_svc = gen_service

    # Create a new APIRouter for versioning
    v1_router = APIRouter(prefix="/api/v1")

    # include routers
    v1_router.include_router(chat_router)
    v1_router.include_router(health_router)
    v1_router.include_router(models_router)
    v1_router.include_router(sessions_router)

    app.include_router(v1_router)
    return app