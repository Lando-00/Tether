"""
Router modules for the API component.
"""
from llm_service.protocol.api.routers.models import get_models_router
from llm_service.protocol.api.routers.sessions import get_sessions_router
from llm_service.protocol.api.routers.generations import get_generations_router
from llm_service.protocol.api.routers.tools import get_tools_router