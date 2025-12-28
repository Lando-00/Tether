import asyncio
import json
import os
import platform
import re
import sys
import threading
import traceback
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, AsyncGenerator, Dict, List, Optional

from mlc_llm import AsyncMLCEngine
from tether_service.core.interfaces import ModelProvider


def _abort_all_requests(engine) -> int:
    """
    Abort all in-flight requests for this engine; returns count aborted.
    This ensures background loops aren't waiting on abandoned requests.
    """
    try:
        # Grab a stable list of IDs first
        inflight = list(getattr(engine.state, "async_streamers", {}).keys())
    except Exception:
        inflight = []
    
    for rid in inflight:
        try:
            # AsyncMLCEngine.abort is async, but _abort is sync & safe here
            engine._abort(rid)
        except Exception:
            pass
    
    return len(inflight)


def _terminate_bounded(engine, timeout: float = 0.75):
    """
    Run engine.terminate() on a daemon thread with timeout.
    Raises TimeoutError if termination doesn't complete in time.
    """
    done = threading.Event()
    err: list[BaseException] = []

    def runner():
        try:
            engine.terminate()
        except BaseException as e:
            err.append(e)
        finally:
            done.set()

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    
    if not done.wait(timeout):
        raise TimeoutError("engine.terminate() timed out")
    
    if err:
        raise err[0]


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
        
        # Instance-level cache and locks (not shared across providers)
        self._engine_cache: Dict[str, AsyncMLCEngine] = {}
        self._cache_lock = Lock()
        self._init_locks: Dict[str, asyncio.Lock] = {}

    def list_models(self) -> List[str]:
        """List available models."""
        models = find_models(self.dist_root)
        return [m["model_name"] for m in models]

    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from the cache.
        Detaches under lock, then terminates outside lock with timeout.
        """
        to_terminate = None
        
        # Find and detach under lock
        with self._cache_lock:
            key_to_delete = None
            for key, engine in self._engine_cache.items():
                if model_name in key:  # Consider stricter matching if needed
                    key_to_delete = key
                    break
            if key_to_delete:
                to_terminate = self._engine_cache.pop(key_to_delete)
        
        if not to_terminate:
            return False
        
        # Terminate outside lock with timeout
        try:
            _abort_all_requests(to_terminate)
            _terminate_bounded(to_terminate, timeout=0.75)
        except TimeoutError:
            print(f"Timeout terminating engine for {model_name} — abandoning")
        except Exception as e:
            print(f"Warning: Error terminating engine for {model_name}: {e}")
        
        print(f"==== MODEL UNLOADED: {model_name} ====")
        return True

    def shutdown_all(self, per_engine_timeout: float = 0.75) -> None:
        """
        Bounded, lock-light shutdown that won't hang the process.
        
        Phase A: Detach cache under lock (fast)
        Phase B: Abort requests and terminate engines outside lock (bounded)
        Phase C: Clear references immediately to avoid destructor hangs
        
        If termination times out, we abandon (drop refs) rather than hang.
        
        CRITICAL: For Qwen2.5-7B (prefill_chunk_size=256) vs Qwen3-4B (2048),
        the smaller chunks mean different OpenCL resource states. We must
        clear engine references immediately after terminate() to avoid
        hanging in destructors when function scope exits.
        """
        import gc
        
        # 1) Snapshot and detach cache quickly
        with self._cache_lock:
            if not self._engine_cache:
                print("==== NO MODELS TO UNLOAD ====")
                return
            items = list(self._engine_cache.items())
            self._engine_cache = {}  # detach in O(1)
        
        print(f"==== SHUTTING DOWN: Unloading {len(items)} model(s) ====")
        
        # 2) Abort in-flight requests and terminate engines outside the lock
        # CRITICAL: Process items one at a time and clear references immediately
        # to prevent holding references that trigger destructors on scope exit
        for key, engine in items:
            # Phase A: Abort outstanding requests
            try:
                aborted = _abort_all_requests(engine)
                if aborted:
                    print(f"Aborted {aborted} in-flight request(s) for {key}")
            except Exception:
                pass
            
            # Phase B: Terminate with timeout
            try:
                print(f"Terminating engine: {key}")
                _terminate_bounded(engine, timeout=per_engine_timeout)
                print(f"Engine terminated: {key}")
            except TimeoutError:
                print(f"Timeout terminating engine: {key} — abandoning")
            except Exception as e:
                print(f"Warning: Error terminating engine {key}: {e}")
                traceback.print_exc(file=sys.stderr)
            
            # Phase C: Immediately delete reference to this engine
            # This prevents accumulation of terminated engines in the items list
            # which would all have destructors called at once on scope exit
            try:
                del engine
            except Exception:
                pass
        
        # 3) Clear the items list and force GC while we're still in controlled context
        try:
            items.clear()  # Remove all (key, engine) tuples
            del items
        except Exception:
            pass
        
        print(f"==== ALL MODELS UNLOADED ====")

    def unload_model_by_cache_key(self, cache_key: str) -> bool:
        """
        Unload a specific model by its cache key.
        Useful for recovery from fatal errors.
        Detaches under lock, then terminates outside lock with timeout.
        
        Args:
            cache_key: The cache key of the model to unload
        
        Returns:
            True if model was found and unloaded, False otherwise
        """
        to_terminate = None
        
        # Detach under lock
        with self._cache_lock:
            if cache_key in self._engine_cache:
                to_terminate = self._engine_cache.pop(cache_key)
        
        if not to_terminate:
            return False
        
        # Terminate outside lock with timeout
        try:
            _abort_all_requests(to_terminate)
            _terminate_bounded(to_terminate, timeout=0.75)
        except TimeoutError:
            print(f"Timeout terminating engine {cache_key} — abandoning")
        except Exception as e:
            print(f"Warning: Error terminating engine {cache_key}: {e}")
        
        print(f"==== MODEL UNLOADED BY CACHE KEY: {cache_key} ====")
        return True

    def get_context_window(self, model_name: str) -> int:
        """
        Get the context window size for a specific model from its config.
        
        Args:
            model_name: Name of the model (e.g., "Qwen3-4B-q4f16_0-MLC")
        
        Returns:
            Context window size in tokens (defaults to 4096 if not found)
        """
        config_path = self.dist_root / model_name / "mlc-chat-config.json"
        
        if not config_path.exists():
            # Fallback to conservative default
            return 4096
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            # Try top-level first, then nested in model_config
            context_window = config.get("context_window_size")
            if context_window is None:
                context_window = config.get("model_config", {}).get("context_window_size")
            
            return context_window if context_window is not None else 4096
            
        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"Warning: Failed to read context window from {config_path}: {e}")
            return 4096

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
                mode="interactive",  # Use server mode for async
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
        
        # Generate request_id so we can abort deterministically
        request_id = f"tether-{uuid.uuid4().hex}"

        print(f"==== STARTING MODEL STREAM: {model_name} (request_id={request_id}) ====")
        stream_generator = None
        try:
            stream_generator = await engine.chat.completions.create(
                messages=messages,
                max_tokens=self.max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                stream=True,
                request_id=request_id,
            )
            
            # Iterate with explicit exception handling
            try:
                async for response in stream_generator:
                    delta = response.choices[0].delta
                    if delta.content:
                        yield delta.content
                    if delta.tool_calls:
                        # The parser expects a list of dicts for tool calls
                        tool_calls_data = [tc.model_dump() for tc in delta.tool_calls]
                        yield tool_calls_data
            except GeneratorExit:
                # Client disconnected - abort the engine-side request
                print(f"==== MODEL STREAM GENERATOR EXIT (client disconnect): {model_name} ====")
                try:
                    engine._abort(request_id)
                except Exception:
                    pass
                raise
            except Exception as iter_error:
                # Exception during iteration
                print(f"==== MODEL STREAM ITERATION ERROR: {model_name} ====")
                print(f"Iteration error: {type(iter_error).__name__}: {iter_error}")
                traceback.print_exc(file=sys.stderr)
                raise
                
        except GeneratorExit:
            # Re-raise GeneratorExit (client disconnect)
            raise
        except Exception as e:
            # Catch all exceptions from the MLC engine (including TVM/OpenCL errors)
            error_type = type(e).__name__
            error_msg = str(e)
            print(f"==== MODEL STREAM ERROR: {model_name} ====")
            print(f"Error type: {error_type}")
            print(f"Error message: {error_msg}")
            traceback.print_exc(file=sys.stderr)
            
            # Check if this is a fatal OpenCL/TVM error that requires model reload
            is_fatal = "TVMError" in error_type or "CLML Error" in error_msg or "CL_" in error_msg
            
            if is_fatal:
                print(f"==== FATAL ERROR DETECTED - Model may need to be reloaded ====")
                # Mark engine as potentially corrupted (future enhancement: auto-reload)
            
            # Re-raise with more context to allow orchestrator to handle
            raise RuntimeError(
                f"Model streaming failed for '{model_name}': {error_type} - {error_msg}"
            ) from e
        finally:
            # Proactively close the async generator to release native handles
            if stream_generator is not None:
                try:
                    await stream_generator.aclose()
                except Exception:
                    pass
            
            # Defensive: abort again if it's still registered
            try:
                engine._abort(request_id)
            except Exception:
                pass
