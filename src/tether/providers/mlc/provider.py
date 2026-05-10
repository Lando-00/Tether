import asyncio
import json
import os
import platform
import re
import threading
import uuid

import structlog

_log = structlog.get_logger(__name__)
from pathlib import Path
from threading import Lock
from typing import Any, AsyncGenerator, Dict, List, Optional

from mlc_llm import AsyncMLCEngine
from tether.core.interfaces import ModelProvider
from tether.providers.hw import HwErrorClass, HwHealth

# §security R-pathtraversal: model_name must be a plain directory component.
# This pattern deliberately excludes path separators, colons, and any other
# characters that could be used to escape the models_root directory.
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


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
    """Build a precise match key for DLL lookup.

    Strips ONLY the trailing ``-MLC`` marker (case-insensitive) so that the
    quantization suffix is preserved. ``Qwen3-4B-q4f16_1-MLC`` becomes
    ``Qwen3-4B-q4f16_1`` — distinct from the q4f16_0 sibling. Two model
    variants that share a family but differ in quant produce different
    keys, so the resolver can pick the right compiled library.

    Pre-fix this function dropped the quant via a regex split and both
    quant variants collapsed to ``Qwen3-4B``; loading then picked the
    alphabetically-first DLL and TVM raised a weight-shape mismatch.
    """
    return re.sub(r"[-_]mlc$", "", model_name, flags=re.IGNORECASE)


def match_model_dlls(model_name: str, dlls):
    """Match a model name against available compiled libraries.

    Two-pass match:

    1. Precise: DLL filename contains the full match key (family + quant).
       Catches the common case where compiled DLLs encode the quant in
       their filename (``Qwen3-4B-q4f16_1-adreno.dll``).
    2. Loose family fallback: if the precise pass finds nothing, drop the
       quant and match on the family alone. Supports the legacy case where
       a single DLL covers multiple quant variants of the same family
       (e.g. ``mymodel.dll`` without quant in the name).
    """
    precise_key = base_key_from_model_name(model_name).lower()
    matches = [d for d in dlls if precise_key in d.name.lower()]
    if matches:
        return matches

    loose_key = model_name.split("-q")[0].lower()
    return [d for d in dlls if loose_key in d.name.lower()]


def find_models(models_root: Path = Path("models")) -> List[Dict[str, str]]:
    """Find available models in the models directory."""
    models = []
    for cfg in models_root.rglob("mlc-chat-config.json"):
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
        models_root: str = "models",
        device: str = "auto",
        max_tokens: int = 1024,
    ):
        """Initialize MLC provider with general config, not a specific model.

        The ``models_root`` directory holds downloaded MLC model directories
        (each containing an ``mlc-chat-config.json``) plus a ``libs/``
        subdirectory of compiled DLLs/shared objects. The ``TETHER_MODELS_DIR``
        environment variable, when set, overrides both the explicit constructor
        argument and the YAML default — this lets operators relocate large
        model artifacts off the project tree without editing config (synthesis
        §4 Phase 8 step 85).
        """
        env_override = os.environ.get("TETHER_MODELS_DIR")
        if env_override:
            models_root = env_override
        self.models_root = Path(models_root)
        self.libs_dir = self.models_root / "libs"
        self.device = device
        self.max_tokens = max_tokens
        
        # Instance-level cache and locks (not shared across providers)
        self._engine_cache: Dict[str, AsyncMLCEngine] = {}
        self._cache_lock = Lock()
        self._init_locks: Dict[str, asyncio.Lock] = {}

    def _validate_model_name(self, name: str) -> None:
        """
        Validate that model_name is a safe, plain directory name.

        §security R-pathtraversal (§4 Phase 0A): any model_name that contains
        path separators, colons, or other escape characters must be rejected
        before any filesystem access. We also resolve the resulting path to
        confirm it remains within models_root, catching encoded traversal
        strings such as '..%2F'.

        Raises:
            ValueError: if name is unsafe or would escape models_root.
        """
        if not isinstance(name, str) or not _MODEL_NAME_RE.match(name):
            raise ValueError(f"invalid model_name: {name!r}")
        # Resolve to absolute paths and verify containment (Python 3.9+).
        # The resolved path must be a strict child of models_root, not
        # models_root itself (which "." would produce).
        resolved = (self.models_root / name).resolve()
        models_root_resolved = self.models_root.resolve()
        if not resolved.is_relative_to(models_root_resolved) or resolved == models_root_resolved:
            raise ValueError(f"invalid model_name: {name!r}")

    def list_models(self) -> List[str]:
        """List available models."""
        models = find_models(self.models_root)
        return [m["model_name"] for m in models]

    def unload_model(self, model_name: str) -> bool:
        """Unload a specific cached engine by EXACT model_name match.

        Phase 3 follow-up (rubber-duck consensus, gpt-5.5 BLOCKING +
        xhigh BLOCKING-2): replaces the previous substring match
        (``if model_name in key``) with canonical-key construction. The
        substring match could unload the WRONG engine when one model
        name is a prefix of another (e.g. ``Qwen3-4B`` vs
        ``Qwen3-4B-Instruct``); ``hw_reset`` calls this directly, so the
        recovery path inherited the bug. Synthesis §6 bug #15; R10/R21.

        Returns:
            ``True`` if a matching engine was found and torn down;
            ``False`` if not in the cache, or the model_name doesn't
            resolve to a real model dir / library (no engine to unload).
        """
        self._validate_model_name(model_name)

        # Compute the canonical cache key for this model_name. Mirrors
        # the construction in ``_get_engine`` / ``_ensure_engine`` so we
        # match the exact key that was inserted on load.
        try:
            model_dir = self.models_root / model_name
            model_lib_path = resolve_model_lib(model_name, self.libs_dir)
            canonical_key = f"{model_dir}:{self.device}:{model_lib_path}"
        except (ValueError, FileNotFoundError):
            # Model name doesn't resolve to a real model dir / library
            # (e.g. models dir was wiped after load). Nothing to unload;
            # return False rather than raising so callers can decide.
            return False

        # Atomically pop the engine under the lock.
        with self._cache_lock:
            if canonical_key not in self._engine_cache:
                return False
            to_terminate = self._engine_cache.pop(canonical_key)

        # Terminate outside the lock with bounded timeout.
        try:
            _abort_all_requests(to_terminate)
            _terminate_bounded(to_terminate, timeout=0.75)
        except TimeoutError:
            _log.warning("provider.engine.terminate_timeout", model_name=model_name)
        except Exception as e:
            _log.warning("provider.engine.terminate_error", model_name=model_name, error=str(e))

        _log.info("provider.engine.unloaded", model_name=model_name)
        return True

    def shutdown_all(self, per_engine_timeout: float = 0.75) -> None:
        """
        Bounded, lock-light, PARALLEL shutdown that won't hang the process.

        Phase A: Detach cache under lock (fast; serial)
        Phase B: PARALLEL — abort requests and terminate engines via
                 ``ThreadPoolExecutor(max_workers=min(N, 4))``. Each worker
                 calls ``_abort_all_requests`` then ``_terminate_bounded`` and
                 drops its local engine reference before returning.
        Phase C: Per-worker references are released by Phase B; the items
                 list is drained into futures and explicitly cleared before
                 we wait, so the only remaining references live inside the
                 worker thread frames.

        Worst case is bounded by ``per_engine_timeout`` (≈0.75s) instead of
        ``N × per_engine_timeout`` because the workers run concurrently.
        Total wall time is additionally bounded by
        ``hw_shutdown_budget_sec`` (the watchdog's outer cap); engines that
        miss that budget are abandoned — the watchdog's ``daemon_thread_call``
        wrapper will let the process exit without waiting on them.

        CRITICAL: For Qwen2.5-7B (prefill_chunk_size=256) vs Qwen3-4B (2048),
        the smaller chunks mean different OpenCL resource states. Each
        worker drops its engine reference immediately after ``terminate()``
        so destructors don't fire after the worker thread exits and start
        chaining cleanup work on the main thread. Synthesis §4 Phase 3
        step 38; B6 step 9.
        """
        import concurrent.futures

        # 1) Snapshot and detach cache quickly (Phase A)
        with self._cache_lock:
            if not self._engine_cache:
                _log.debug("provider.shutdown.no_models")
                return
            items = list(self._engine_cache.items())
            self._engine_cache = {}  # detach in O(1)

        n = len(items)
        max_workers = min(n, 4)
        _log.info("provider.shutdown.started", model_count=n, max_workers=max_workers)

        # 2) Parallel teardown (Phase B). Each worker owns its engine ref
        # locally; when the worker function returns, the ref dies before
        # the executor records the result, so destructors don't accumulate.
        def _terminate_one(key: str, engine):
            try:
                aborted = _abort_all_requests(engine)
                if aborted:
                    _log.debug("provider.engine.requests_aborted", cache_key=key, aborted_count=aborted)
            except Exception:
                pass

            try:
                _log.debug("provider.engine.terminating", cache_key=key)
                _terminate_bounded(engine, timeout=per_engine_timeout)
                _log.debug("provider.engine.terminated", cache_key=key)
                return (key, "ok")
            except TimeoutError:
                _log.warning("provider.engine.terminate_timeout", cache_key=key)
                return (key, "timeout")
            except Exception as e:
                _log.warning("provider.engine.terminate_error", cache_key=key, error=str(e), exc_info=True)
                return (key, f"error: {e}")
            # `engine` ref dies on return — no scope-exit destructor pile-up.

        # Use shutdown(wait=False) on abandon so a stuck future doesn't
        # block our caller. The watchdog (daemon_thread_call) will let the
        # process exit even if a worker is wedged inside native code.
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="mlc-shutdown",
        )
        results: list[tuple[str, str]] = []
        try:
            futures = [executor.submit(_terminate_one, k, e) for k, e in items]
            # Drop the list-level engine refs immediately so only worker
            # frames keep the engines alive. Without this, `items` would
            # pin every engine until after the as_completed loop.
            items.clear()
            items = None

            # Bound total wall time by the provider's shutdown budget.
            # Per-engine timeout already caps each worker; the outer cap
            # protects against worst-case contention (e.g., several engines
            # all timing out at once).
            total_budget = max(per_engine_timeout * 1.5, self.hw_shutdown_budget_sec)

            try:
                for fut in concurrent.futures.as_completed(futures, timeout=total_budget):
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        results.append(("?", f"error: {e}"))
            except concurrent.futures.TimeoutError:
                _log.warning(
                    "provider.shutdown.budget_exceeded",
                    budget_sec=round(total_budget, 2),
                )
        finally:
            # Don't wait for stuck futures; abandoned worker threads are
            # daemons (default for ThreadPoolExecutor) so they won't block
            # interpreter shutdown. Synthesis §4 Phase 3 step 38.
            executor.shutdown(wait=False)

        # 3) Phase C — references already dropped by workers; just log.
        _log.info("provider.shutdown.complete", results=results)
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
            _log.warning("provider.engine.terminate_timeout", cache_key=cache_key)
        except Exception as e:
            _log.warning("provider.engine.terminate_error", cache_key=cache_key, error=str(e))
        
        _log.info("provider.engine.unloaded_by_key", cache_key=cache_key)
        return True

    # ------------------------------------------------------------------
    # HardwareLifecycle Protocol implementation.
    #
    # MLCProvider owns native hardware resources (OpenCL command queues
    # via TVM, the Adreno X1 GPU through Qualcomm CLML). The watchdog
    # (``HardwareWatchdog``, Phase 3 step 30) detects that we own such
    # resources via ``isinstance(provider, HardwareLifecycle)`` and routes
    # shutdown / recovery / health probes through these 5 members.
    #
    # DummyProvider deliberately does NOT implement this Protocol so the
    # watchdog skips it (DummyProvider has no native handles to release).
    #
    # Synthesis §4 Phase 3 step 34; B6 step 5; A3 step 6.
    # ------------------------------------------------------------------

    def hw_classify(self, exc: BaseException) -> HwErrorClass:
        """Classify an MLC-emitted exception.

        Delegates to the canonical ``classify_mlc_error`` so all consumers
        (orchestrator, watchdog, signal_supervisor's thread exception
        handler) agree on what's fatal. The import is lazy to keep the
        ``providers.mlc.errors`` module load off the import-time path of
        anything that just touches ``MLCProvider``'s class object.
        """
        from tether.providers.mlc.errors import classify_mlc_error
        return classify_mlc_error(exc)

    async def hw_reset(self, model_name: str) -> None:
        """Tear down + EAGER reload the named model.

        Called by ``HardwareWatchdog.reset_after`` when ``hw_classify``
        returns ``FATAL_RECOVERABLE`` (synthesis §4 Phase 3 step 36).

        Implementation:

            1. ``_validate_model_name`` — path-traversal guard from
               Phase 0A (security R-pathtraversal). hw_reset is reachable
               from anywhere a FatalProviderError surfaces, so we must
               re-validate the name even though stream() already did.
            2. ``unload_model`` — releases OpenCL/TVM resources for the
               cache entry whose key contains ``model_name``.
            3. ``_ensure_engine`` — eagerly recreates the engine.

        The eager reload (vs. defer-to-next-request) is intentional: the
        next request may immediately re-trigger the corrupted state if we
        don't proactively prove the engine works. Synthesis A3 step 6.

        NOTE: this method does NOT manipulate Python GC. The GC-disable
        invariant (R5) only applies to ``HardwareWatchdog.shutdown_all``'s
        daemon thread, not to runtime recovery.
        """
        self._validate_model_name(model_name)
        self.unload_model(model_name)
        # Eager reload: load the engine again so the next request doesn't
        # pay the cold-start cost AND we surface any reload failure now
        # (not later, in the middle of a user request).
        await self._ensure_engine(model_name)

    async def hw_health(self) -> HwHealth:
        """Cheap health probe — does NOT touch the model graph.

        Used by the ``/readyz`` handler (synthesis §4 Phase 3 step 37
        replaces the legacy ``list_models()`` probe with this).

        Returns:
            - ``"healthy"`` if the engine cache has at least one entry
              (engines are pre-warmed and ready)
            - ``"degraded"`` if no engines loaded yet (cold cache; not
              an error — readyz consumers may choose to treat this as
              "not ready" without flagging it as a failure)

        We do not currently track recent reset failures here; if
        ``hw_reset`` raised, the orchestrator surfaces the error event
        directly. Future iterations may add a "last reset failed at"
        timestamp and downgrade the status accordingly.
        """
        with self._cache_lock:
            engine_count = len(self._engine_cache)

        if engine_count == 0:
            return HwHealth(
                status="degraded",
                details={"loaded_models": 0, "note": "cold cache"},
            )
        return HwHealth(
            status="healthy",
            details={"loaded_models": engine_count},
        )

    @property
    def hw_shutdown_budget_sec(self) -> float:
        """Total budget for shutting down all engines.

        Matches the legacy ``api.py:189`` hardcoded ``3.0`` seconds. The
        watchdog's daemon thread waits at most this long before
        abandoning cleanup. Synthesis §4 Phase 3 step 30.
        """
        return 3.0

    @property
    def hw_per_engine_terminate_sec(self) -> float:
        """Per-engine cap when shutting down in parallel.

        Default: ``hw_shutdown_budget_sec / 4``. Matches the existing
        ``shutdown_all(per_engine_timeout=0.75)`` and
        ``_terminate_bounded(timeout=0.75)`` constants in this file.
        Synthesis §4 Phase 3 step 38.
        """
        return self.hw_shutdown_budget_sec / 4

    # ------------------------------------------------------------------
    # Provider v2 contract (Phase 3 step 39).
    # Synthesis §4 Phase 3 step 39, §11.3 R21, §6 bug #12 (typed path
    # emits ProviderToolCall instead of dropping native tool_calls).
    # ------------------------------------------------------------------

    @property
    def kind(self) -> str:
        return "mlc"

    @property
    def capabilities(self):
        # Lazy-import to keep ``providers/types`` off the import-time path
        # of any code that just touches MLCProvider's class object.
        from tether.providers.types import ProviderCapabilities
        return ProviderCapabilities(
            streaming=True,
            tools_native=True,         # MLC supports OpenAI tools API.
            tools_marker=True,         # Tether also supports <<function_call>>.
            thinking_channel=False,    # MLC has no separate thinking channel.
            cancel_inflight=True,      # engine._abort(request_id) supported.
            multi_model=True,          # engine cache holds multiple.
            warm_up_required=True,     # _ensure_engine takes seconds first time.
        )

    async def warm_up(self, model_name: str) -> None:
        """Pre-create the engine for ``model_name``.

        Validates path-traversal first (security R-pathtraversal) and
        then triggers the cold-start path so the next inference is fast.
        Synthesis §4 Phase 3 step 39.
        """
        self._validate_model_name(model_name)
        await self._ensure_engine(model_name)

    async def aclose(self) -> None:
        """Provider-level shutdown — calls :meth:`shutdown_all`.

        :class:`HardwareWatchdog` wraps this from outside in
        ``Engine.aclose`` (Phase 3 step 35) when the provider implements
        :class:`HardwareLifecycle`. The wrapping watchdog applies the
        daemon-thread + GC-disable invariant (R5); this method itself
        just delegates to the existing parallel teardown
        (Phase 3 step 38).
        """
        self.shutdown_all()

    async def stream_typed(
        self,
        *,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        cancel_token: Optional[Any] = None,
    ):
        """v2 typed stream — adapts from the legacy :meth:`stream` chunks.

        Native MLC ``delta.tool_calls`` (when the legacy ``stream()``
        yields a list-of-dicts) become :class:`ProviderToolCall` events
        instead of being silently dropped. Synthesis §6 bug #12.

        This PR does NOT migrate the orchestrator. Phase 5 step 52 will.
        For now, this method is reachable only by tests and any library
        user who calls it directly.

        The kwargs ``request_id``, ``max_output_tokens``, and
        ``cancel_token`` are accepted for forward compatibility (the v2
        contract documents them) but are not threaded through to the
        legacy ``stream()`` yet — Phase 5 will add the wiring.
        """
        from tether.providers.types import (
            ProviderText,
            ProviderToolCall,
        )

        # Adapt the existing legacy stream's chunks to typed events. The
        # legacy generator already handles abort/cancel/error classification
        # (synthesis §4 Phase 3 step 34); we just translate its output.
        async for chunk in self.stream(
            model_name=model_name, messages=messages, tools=tools
        ):
            if isinstance(chunk, str):
                yield ProviderText(text=chunk)
            elif isinstance(chunk, list):
                # Legacy native-tool_calls path: list of dicts (each tc
                # is ``delta.tool_calls[i].model_dump()``). Adapt to
                # ProviderToolCall events. Synthesis §6 bug #12.
                for tc in chunk:
                    func = tc.get("function", {}) if isinstance(tc, dict) else {}
                    args = func.get("arguments")
                    # MLC sometimes returns ``arguments`` as a JSON string;
                    # normalize to a dict so consumers don't re-parse.
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"_raw": args}
                    if not isinstance(args, dict):
                        args = {}
                    yield ProviderToolCall(
                        tool_call_id=(tc.get("id") if isinstance(tc, dict) else None)
                            or f"tc-{id(tc):x}",
                        name=(func.get("name") or ""),
                        arguments=args,
                    )
            else:
                # Unknown chunk type — surface as text for safety so the
                # consumer doesn't silently lose data.
                yield ProviderText(text=str(chunk))

    def get_context_window(self, model_name: str) -> int:
        """
        Get the context window size for a specific model from its config.
        
        Args:
            model_name: Name of the model (e.g., "Qwen3-4B-q4f16_0-MLC")
        
        Returns:
            Context window size in tokens (defaults to 4096 if not found)
        """
        self._validate_model_name(model_name)
        config_path = self.models_root / model_name / "mlc-chat-config.json"
        
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
            _log.warning(
                "provider.config.context_window_read_error",
                config_path=str(config_path),
                error=str(e),
            )
            return 4096

    def _get_engine(self, model_name: str) -> AsyncMLCEngine:
        """Get a cached engine instance or create a new one for a specific model."""
        self._validate_model_name(model_name)
        model_dir = self.models_root / model_name
        if not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_dir}")

        model_lib_path = resolve_model_lib(model_name, self.libs_dir)
        cache_key = f"{model_dir}:{self.device}:{model_lib_path}"

        with self._cache_lock:
            if cache_key in self._engine_cache:
                return self._engine_cache[cache_key]

            _log.info("provider.engine.loading", model_name=model_name, device=self.device)
            engine = AsyncMLCEngine(
                model=str(model_dir),
                model_lib=model_lib_path,
                device=self.device,
                mode="interactive",  # Use server mode for async
            )
            _log.info("provider.engine.loaded", model_name=model_name)
            self._engine_cache[cache_key] = engine
            return engine

    async def _ensure_engine(self, model_name: str) -> AsyncMLCEngine:
        """Ensure the engine for a specific model is initialized."""
        self._validate_model_name(model_name)
        # Check if engine is already cached (fast path)
        model_dir = self.models_root / model_name
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
        *,
        request_id: Optional[str] = None,
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
        
        # Generate mlc_request_id so we can abort the engine request deterministically.
        # Renamed from `request_id` (the caller's correlation ID) to avoid shadowing.
        # Phase 7 step 72: caller request_id available for internal log correlation.
        mlc_request_id = f"tether-{uuid.uuid4().hex}"

        # Phase 7 step 70: demoted to DEBUG (no user content emitted).
        # request_id is the caller's correlation ID; mlc_request_id is the
        # internal per-request abort handle — log only the internal one.
        _log.debug(
            "provider.stream.starting",
            model_name=model_name,
            mlc_request_id=mlc_request_id,
        )
        stream_generator = None
        try:
            stream_generator = await engine.chat.completions.create(
                messages=messages,
                max_tokens=self.max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                stream=True,
                request_id=mlc_request_id,
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
                _log.info(
                    "provider.stream.client_disconnect",
                    model_name=model_name,
                    mlc_request_id=mlc_request_id,
                )
                try:
                    engine._abort(mlc_request_id)
                except Exception:
                    pass
                raise
            except Exception as iter_error:
                # Exception during iteration
                _log.error(
                    "provider.stream.iteration_error",
                    model_name=model_name,
                    error_type=type(iter_error).__name__,
                    exc_info=True,
                )
                raise
                
        except GeneratorExit:
            # Re-raise GeneratorExit (client disconnect)
            raise
        except Exception as e:
            # Catch all exceptions from the MLC engine (including TVM/OpenCL errors).
            #
            # Synthesis §4 Phase 3 step 34 / §6 row 13: classify via the
            # canonical ``classify_mlc_error`` (single source of truth) and
            # raise typed errors so the orchestrator and watchdog can branch
            # on class, not on substring grep. Lazy-imported here so import
            # of this module doesn't pull errors.py.
            from tether.core.errors import (
                FatalProviderError,
                TransientProviderError,
            )
            from tether.providers.mlc.errors import classify_mlc_error

            error_type = type(e).__name__
            error_msg = str(e)
            # Phase 7 step 70: structured error event replaces 3 print+traceback.
            # Event name "provider.stream.engine_error" (not "provider.stream.error"
            # which chatty.py orchestrator already uses for its span). exc_info=True
            # captures the full traceback via structlog's format_exc_info processor.
            _log.error(
                "provider.stream.engine_error",
                model_name=model_name,
                error_type=error_type,
                exc_info=True,
            )

            # Classify using the canonical function — replaces the legacy
            # 3-line substring grep that was here before. Kept as a local
            # bool so the inline log message below stays readable.
            err_class = classify_mlc_error(e)
            is_fatal = err_class == HwErrorClass.FATAL_RECOVERABLE

            if is_fatal:
                _log.error(
                    "provider.stream.fatal_classified",
                    model_name=model_name,
                    error_type=error_type,
                )
                # Watchdog (Phase 3 step 30) handles the reload via hw_reset.

            # Re-raise as a typed TetherError so callers can branch on
            # class. ``from e`` preserves the original cause for diagnostics.
            wrapped_msg = (
                f"Model streaming failed for '{model_name}': "
                f"{error_type} - {error_msg}"
            )
            if is_fatal:
                raise FatalProviderError(wrapped_msg) from e
            else:
                raise TransientProviderError(wrapped_msg) from e
        finally:
            # Proactively close the async generator to release native handles
            if stream_generator is not None:
                try:
                    await stream_generator.aclose()
                except Exception:
                    pass
            
            # Defensive: abort again if it's still registered.
            # NOTE: use mlc_request_id (the engine's per-request abort ID),
            # NOT the caller's `request_id` correlation arg added in Phase 7
            # step 72 — they are different concepts and shadowing was the
            # whole reason for the rename.
            try:
                engine._abort(mlc_request_id)
            except Exception:
                pass
