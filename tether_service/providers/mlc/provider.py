import asyncio
import json
import os
import platform
import re
from pathlib import Path
from threading import Lock
from typing import Any, AsyncGenerator, Dict, List, Optional

from mlc_llm import AsyncMLCEngine
from tether_service.core.interfaces import ModelProvider


def base_key_from_model_name(model_name: str) -> str:
    """Extract the base key from a model name for DLL matching."""
    s = model_name
    s = re.sub(r"([_-]mlc)$", "", s, flags=re.IGNORECASE)
    s = re.split(r"-q\d+f?\d*[_-]?\d*-?mlc", model_name, flags=re.IGNORECASE)
    return s[0] if s else model_name


def match_model_dlls(model_name: str, dlls):
    """Match model name with available DLLs using pattern matching."""
    base_key = base_key_from_model_name(model_name).lower()
    matches = []
    for dll in dlls:
        name = dll.name.lower()
        if base_key and base_key in name:
            matches.append(dll)
    if not matches:
        loose_key = model_name.split("-q")[0].lower()
        for dll in dlls:
            if loose_key in dll.name.lower():
                matches.append(dll)
    return matches


def find_models(dist_root: Path = Path("dist")) -> List[Dict[str, str]]:
    """Find available models in the dist directory."""
    models = []
    for cfg in dist_root.rglob("mlc-chat-config.json"):
        model_dir = cfg.parent
        models.append(
            {"model_name": model_dir.name, "model_dir": str(model_dir), "config_path": str(cfg)}
        )
    return models


def resolve_model_lib(model_name: str, libs_dir: Path) -> str:
    """Resolve the appropriate library file for a given model."""
    ext = {"Windows": ".dll", "Darwin": ".dylib"}.get(platform.system(), ".so")
    if not libs_dir.exists():
        raise ValueError(f"Libraries directory not found: {libs_dir}")

    dlls = list(libs_dir.glob(f"*{ext}"))
    matches = match_model_dlls(model_name, dlls)
    if matches:
        return str(matches[0])

    # fallback: try generic mlc_llm lib
    generic = libs_dir / f"mlc_llm{ext}"
    if generic.exists():
        return str(generic)

    raise ValueError(f"No matching model library found for {model_name} in {libs_dir}")


class MLCProvider(ModelProvider):
    _engine_cache: Dict[str, AsyncMLCEngine] = {}
    _cache_lock = Lock()

    def __init__(
        self,
        dist_root: str = "dist",
        device: str = "auto",
        max_tokens: int = 1024,
    ):
        """Initialize MLC provider with general config, not a specific model."""
        self.dist_root = Path(dist_root)
        self.libs_dir = self.dist_root / "libs"
        self.device = device
        self.max_tokens = max_tokens
        self._init_locks: Dict[str, asyncio.Lock] = {}

    def list_models(self) -> List[str]:
        """List available models."""
        models = find_models(self.dist_root)
        return [m["model_name"] for m in models]

    def unload_model(self, model_name: str) -> bool:
        """Unload a model from the cache."""
        # In this simplified implementation, we find the cache key and pop it.
        # A more robust implementation might need to map model_name to cache_key.
        with self._cache_lock:
            key_to_delete = None
            for key, engine in self._engine_cache.items():
                if model_name in key:
                    key_to_delete = key
                    break
            if key_to_delete:
                # This will call the deconstructor of the engine
                self._engine_cache.pop(key_to_delete)
                print(f"==== MODEL UNLOADED: {model_name} ====")
                return True
        return False

    def _get_engine(self, model_name: str) -> AsyncMLCEngine:
        """Get a cached engine instance or create a new one for a specific model."""
        model_dir = self.dist_root / model_name
        if not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_dir}")

        model_lib_path = resolve_model_lib(model_name, self.libs_dir)
        cache_key = f"{model_dir}:{self.device}:{model_lib_path}"

        with self._cache_lock:
            if cache_key in self._engine_cache:
                return self._engine_cache[cache_key]

            print(f"==== LOADING MODEL: {model_name} on {self.device} ====")
            engine = AsyncMLCEngine(
                model=str(model_dir),
                model_lib=model_lib_path,
                device=self.device,
                mode="server",  # Use server mode for async
            )
            print(f"==== MODEL LOADED: {model_name} ====")
            self._engine_cache[cache_key] = engine
            return engine

    async def _ensure_engine(self, model_name: str) -> AsyncMLCEngine:
        """Ensure the engine for a specific model is initialized."""
        # Check if engine is already cached (fast path)
        model_dir = self.dist_root / model_name
        model_lib_path = resolve_model_lib(model_name, self.libs_dir)
        cache_key = f"{model_dir}:{self.device}:{model_lib_path}"
        with self._cache_lock:
            if cache_key in self._engine_cache:
                return self._engine_cache[cache_key]

        # Get or create a lock for this specific model initialization
        with self._cache_lock:
            if model_name not in self._init_locks:
                self._init_locks[model_name] = asyncio.Lock()
        
        init_lock = self._init_locks[model_name]

        async with init_lock:
            # Double-check after acquiring lock
            with self._cache_lock:
                if cache_key in self._engine_cache:
                    return self._engine_cache[cache_key]
            
            # This is a blocking call, so we run it in an executor
            loop = asyncio.get_running_loop()
            engine = await loop.run_in_executor(None, self._get_engine, model_name)
            return engine

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[str | List[Dict[str, Any]], None]:
        """Stream raw text chunks from the MLC engine for a specific model."""
        engine = await self._ensure_engine(model_name)
        if not engine:
            raise RuntimeError(f"MLC Engine for model '{model_name}' could not be initialized.")

        # Pre-process tool_call args in messages
        for m in messages:
            call_list = m.get("tool_calls")
            if call_list:
                for tc in call_list:
                    func = tc.get("function", {})
                    args = func.get("arguments")
                    if isinstance(args, str):
                        try:
                            func["arguments"] = json.loads(args)
                        except json.JSONDecodeError:
                            # Ignore if it's not valid JSON
                            pass

        tool_choice = "auto" if tools else "none"

        print(f"==== STARTING MODEL STREAM: {model_name} ====")
        stream_generator = await engine.chat.completions.create(
            messages=messages,
            max_tokens=self.max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
        )
        async for response in stream_generator:
            delta = response.choices[0].delta
            if delta.content:
                yield delta.content
            if delta.tool_calls:
                # The parser expects a list of dicts for tool calls
                tool_calls_data = [tc.model_dump() for tc in delta.tool_calls]
                yield tool_calls_data
