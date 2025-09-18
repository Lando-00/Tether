"""
mlc_engine.py - Model Component for MCP Architecture

This module represents the Model component in a Model-Context-Protocol architecture.
It handles all interactions with the MLC-LLM engine, including:
1. Model loading/unloading
2. Inference logic
3. Engine cache management
4. DLL resolution

TODO: Add model version tracking
TODO: Implement model performance metrics
TODO: Add fallback mechanisms for model loading failures
TODO: Optimize memory management for large models

The ModelComponent is designed to be independent of any specific context or protocol.
"""

import re
import os
from pathlib import Path
import platform
from threading import Lock
from typing import List, Dict, Any, Optional, Tuple

from fastapi import HTTPException
from mlc_llm import MLCEngine

# Monkey-patch mlc_llm to support JSON-based tool_call outputs
import json
from mlc_llm.serve import engine_base as _engine_base

# TODO: Remove After TESTING
# _original_convert = _engine_base.convert_function_str_to_json

# def convert_function_str_to_json(stringified_calls: str):
#     try:
#         obj = json.loads(stringified_calls)
#     except json.JSONDecodeError:
#         return _original_convert(stringified_calls)
#     calls = obj if isinstance(obj, list) else [obj]
#     function_calls = []
#     for c in calls:
#         tc = c.get("tool_call") or c.get("function_call") or {}
#         name = tc.get("name")
#         params = tc.get("parameters") or {}
#         if name:
#             function_calls.append({"name": name, "arguments": params})
#         else:
#             function_calls.append(None)
#     return function_calls

# _engine_base.convert_function_str_to_json = convert_function_str_to_json

# --- DLL resolution functions ---

def base_key_from_model_name(model_name: str) -> str:
    """Extract the base key from a model name for DLL matching."""
    s = model_name
    s = re.sub(r'([_-]mlc)$', '', s, flags=re.IGNORECASE)
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
        models.append({
            "model_name": model_dir.name,
            "model_dir": str(model_dir),
            "config_path": str(cfg)
        })
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


# --- ModelComponent class ---

class ModelComponent:
    """Model component for MCP architecture handling MLC-LLM model operations."""
    
    def __init__(self, dist_path: Path = Path("dist")):
        """Initialize the model component with the dist path."""
        self.dist_path = dist_path
        self.libs_path = dist_path / "libs"
        self._engine_cache = {}
        self._cache_lock = Lock()
        self.max_cache_size = 2  # Maximum number of models to keep in cache
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """Get a list of available models."""
        return find_models(self.dist_path)
    
    def _get_model_path(self, model_name: str) -> str:
        """Get the full path to a model directory."""
        model_path = os.path.join(str(self.dist_path), model_name)
        if not os.path.isdir(model_path):
            raise ValueError(f"Model directory not found: {model_path}")
        return model_path
    
    def _get_model_lib_path(self, model_name: str, custom_lib_path: Optional[str] = None) -> str:
        """Get the appropriate library path for a model."""
        if custom_lib_path:
            lib_path = Path(custom_lib_path)
            if not lib_path.is_file():
                raise ValueError(f"Model library not found: {lib_path}")
            return str(lib_path)
        
        try:
            return resolve_model_lib(model_name, self.libs_path)
        except Exception as e:
            raise ValueError(f"Failed to resolve model library: {e}")
    
    def _get_engine(self, model_dir: str, device: str, dll_path: str) -> MLCEngine:
        """Get a cached engine instance or create a new one."""
        cache_key = (model_dir, device, dll_path)
        
        with self._cache_lock:
            if cache_key in self._engine_cache:
                return self._engine_cache[cache_key]
            
            # Cache eviction if full
            if len(self._engine_cache) >= self.max_cache_size:
                for engine in self._engine_cache.values():
                    engine.terminate()
                self._engine_cache.clear()
            
            # Create new engine
            engine = MLCEngine(
                model=model_dir,
                model_lib=dll_path,
                device=device,
                # TODO: Remove After TESTING
                mode="interactive"
            )
            self._engine_cache[cache_key] = engine
            return engine
    
    def generate(self, 
                 model_name: str, 
                 messages: List[Dict[str, Any]], 
                 device: str = "auto", 
                 dll_path: Optional[str] = None,
                 max_tokens: int = 512, 
                 temperature: float = 0.2, 
                 top_p: float = 0.95,
                 tools: Optional[List[Dict]] = None,
                 stream: bool = False) -> Any:
        """
        Generate text using a model.
        
        Args:
            model_name: Name of the model to use
            messages: List of message dictionaries in chat format
            device: Device to run inference on (auto, cpu, cuda, opencl)
            dll_path: Optional custom DLL path
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            tools: Optional list of tools for function calling
            stream: Whether to stream the output
            
        Returns:
            Model response or generator for streaming responses
        """
        try:
            model_dir = self._get_model_path(model_name)
            lib_path = self._get_model_lib_path(model_name, dll_path)
            engine = self._get_engine(model_dir, device, lib_path)
            
            tool_choice = "auto" if tools else "none"
            
            # Pre-process tool_call args in messages
            import json
            for m in messages:
                call_list = getattr(m, 'tool_calls', None) or m.get('tool_calls')
                if call_list:
                    for tc in call_list:
                        func = tc.get('function') or {}
                        args = func.get('arguments')
                        if isinstance(args, str):
                            func['arguments'] = json.loads(args)
            return engine.chat.completions.create(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                tool_choice=tool_choice,
                stream=stream
            )
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")
    
    def unload_model(self, model_name: Optional[str] = None, device: Optional[str] = None) -> None:
        """Unload a specific model or all models from the cache."""
        with self._cache_lock:
            if model_name is None and device is None:
                # Unload all models
                for engine in self._engine_cache.values():
                    engine.terminate()
                self._engine_cache.clear()
            else:
                # Try to find matching engines
                keys_to_remove = []
                for (dir_path, dev, lib_path), engine in self._engine_cache.items():
                    model_matches = model_name is None or model_name in dir_path
                    device_matches = device is None or device == dev
                    if model_matches and device_matches:
                        engine.terminate()
                        keys_to_remove.append((dir_path, dev, lib_path))
                
                # Remove terminated engines from cache
                for key in keys_to_remove:
                    self._engine_cache.pop(key, None)
    
    def clear_engine_cache(self) -> None:
        """Clear the entire engine cache."""
        with self._cache_lock:
            for engine in self._engine_cache.values():
                engine.terminate()
            self._engine_cache.clear()
