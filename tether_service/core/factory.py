from importlib import import_module
from typing import Any, cast
import inspect

from tether_service.core.config import load_settings
from tether_service.core.interfaces import ModelProvider, StreamParser, SessionStore
from tether_service.protocol.service.generation_service import GenerationService


def load(dotted: str, **kwargs: Any) -> Any:
    """Import a dotted path and instantiate the class if callable.
    Filters kwargs to match the constructor signature (unless **kwargs is accepted)."""
    module, cls = dotted.rsplit(".", 1)
    mod = import_module(module)
    obj = getattr(mod, cls)

    if isinstance(obj, type):
        # class: inspect __init__ signature
        sig = inspect.signature(obj.__init__)
        params = list(sig.parameters.values())
        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in params)
        if accepts_kwargs:
            return obj(**kwargs)
        # Filter only accepted params (skip 'self')
        allowed = {p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and p.name != "self"}
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return obj(**filtered)

    # callable or object (rare)
    return obj


class ServiceFactory:
    def __init__(self):
        self.config = load_settings()
        self._provider: ModelProvider | None = None
        self._parser: StreamParser | None = None
        self._store: SessionStore | None = None

    def get_provider(self) -> ModelProvider:
        if not self._provider:
            model_cfg = self.config.get("providers", {}).get("model", {})
            impl = model_cfg.get("impl")
            args = model_cfg.get("args", {}) or {}
            self._provider = cast(ModelProvider, load(impl, **args))
        return self._provider

    def get_parser(self) -> StreamParser:
        if not self._parser:
            parser_cfg = self.config.get("providers", {}).get("parser", {})
            impl = parser_cfg.get("impl")
            args = parser_cfg.get("args", {}) or {}
            self._parser = cast(StreamParser, load(impl, **args))
        return self._parser

    def get_store(self) -> SessionStore:
        if not self._store:
            store_cfg = self.config.get("providers", {}).get("session_store", {})
            impl = store_cfg.get("impl")
            args = store_cfg.get("args", {}) or {}
            self._store = cast(SessionStore, load(impl, **args))
        return self._store

    def get_generation_service(self) -> GenerationService:
        # Tools registry
        from tether_service.core.tool_registry import ToolRegistry

        tools_cfg = self.config.get("tools", {}) or {}
        registry_cfg = tools_cfg.get("registry", []) or []
        enabled = tools_cfg.get("enabled", []) or []
        registry = ToolRegistry(registry_cfg, enabled)
        tools = registry.all()

        system_prompt = self.config.get("system", {}).get("prompt", "")

        return GenerationService(
            provider=self.get_provider(),
            parser=self.get_parser(),
            session_store=self.get_store(),
            tools=tools,
            system_prompt=system_prompt,
        )
