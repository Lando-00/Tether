from contextlib import asynccontextmanager
from typing import cast
import asyncio
import os
import signal
import sys
import threading

from fastapi import APIRouter, FastAPI

from tether_service.app.http.routers.chat import router as chat_router
from tether_service.app.http.routers.health import router as health_router
from tether_service.app.http.routers.models import router as models_router
from tether_service.app.http.routers.sessions import router as sessions_router
from tether_service.core.interfaces import ModelProvider
from tether_service.core.interfaces import StreamParser, SessionStore
from tether_service.core.logging import logger


# Global flag to track if we're in shutdown
_shutdown_in_progress = False
_shutdown_start_time = None


def _install_force_exit_handler(max_shutdown_sec: float = 5.0):
    """
    Install a signal handler that will force exit if shutdown takes too long.
    This prevents indefinite hangs from stuck native code.
    """
    import time
    
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    
    def force_exit_handler(signum, frame):
        global _shutdown_in_progress, _shutdown_start_time
        
        if not _shutdown_in_progress:
            # First signal - start graceful shutdown
            _shutdown_in_progress = True
            _shutdown_start_time = time.time()
            print("\n==== SHUTDOWN INITIATED (Ctrl+C) ====")
            
            # Call original handler to trigger uvicorn shutdown
            if signum == signal.SIGINT and callable(original_sigint):
                original_sigint(signum, frame)
            elif signum == signal.SIGTERM and callable(original_sigterm):
                original_sigterm(signum, frame)
        else:
            # Second signal OR timeout - force exit
            elapsed = time.time() - (_shutdown_start_time or 0)
            if elapsed > max_shutdown_sec:
                print(f"\n==== FORCE EXIT: Shutdown exceeded {max_shutdown_sec}s ====")
            else:
                print("\n==== FORCE EXIT: Second interrupt signal received ====")
            
            print("Forcing immediate exit...")
            os._exit(1)  # Hard exit, no cleanup
    
    # Install handler for both SIGINT (Ctrl+C) and SIGTERM
    signal.signal(signal.SIGINT, force_exit_handler)
    signal.signal(signal.SIGTERM, force_exit_handler)
    
    print(f"Installed force-exit handler (max shutdown: {max_shutdown_sec}s)")


def handle_thread_exception(args):
    """
    Global handler for uncaught exceptions in threads (Python 3.8+).
    This catches TVM/OpenCL errors that occur in background threads.
    """
    exc_type, exc_value, exc_tb, thread = args.exc_type, args.exc_value, args.exc_traceback, args.thread
    if exc_type is SystemExit:
        # Allow clean exit
        return
    
    logger.error(
        f"Uncaught exception in thread {thread.name}: {exc_type.__name__}: {exc_value}",
        exc_info=(exc_type, exc_value, exc_tb)
    )
    
    # Check if this is a fatal TVM/OpenCL error
    error_msg = str(exc_value)
    is_fatal = "TVMError" in exc_type.__name__ or "CLML" in error_msg or "CL_" in error_msg
    
    if is_fatal:
        logger.critical(
            f"FATAL ERROR in background thread: {exc_type.__name__} - {error_msg}\n"
            "This is likely an OpenCL/TVM driver error. The model may be corrupted.\n"
            "Consider restarting the service or unloading/reloading the model."
        )


def _asyncio_exc_handler(loop, context):
    """Handler for asyncio exceptions during shutdown."""
    msg = context.get("message", "")
    exc = context.get("exception")
    logger.error(f"Asyncio exception during shutdown: {msg}", exc_info=exc)


async def shutdown_provider_with_timeout(provider, timeout_sec: float = 3.0):
    """
    Shutdown the provider with timeout, using a daemon thread to avoid hanging on exit.
    
    Critical: We use a daemon thread directly instead of asyncio.to_thread() because:
    - asyncio.to_thread() uses the default ThreadPoolExecutor with non-daemon threads
    - Even if we cancel the task, the executor thread keeps running
    - Python waits for non-daemon threads on exit -> hang if terminate() blocks
    
    With a daemon thread, Python will exit immediately even if terminate() is stuck.
    
    IMPORTANT: For models like Qwen2.5-7B with smaller prefill chunks (256 vs 2048),
    OpenCL/CLML destructors may hang. We protect against this by:
    1. Running shutdown in a daemon thread (can be abandoned)
    2. Explicit timeout on the entire shutdown operation
    3. Disabling GC in the daemon thread to prevent destructor hangs
    """
    import threading
    import time
    import gc
    
    done = threading.Event()
    
    def do_shutdown():
        # CRITICAL: Disable GC in this thread to prevent hanging in destructors
        # This is especially important for Qwen2.5-7B which has different OpenCL
        # resource states than Qwen3-4B due to smaller prefill chunks (256 vs 2048).
        # We disable GC and NEVER re-enable it in this thread - it's a daemon thread
        # that will be killed on process exit anyway, so accumulated garbage is fine.
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
            print("==== SHUTDOWN: GC disabled in daemon thread ====")
        
        try:
            if hasattr(provider, 'shutdown_all'):
                print("==== SHUTTING DOWN: Cleaning up models ====")
                provider.shutdown_all()
                print("==== SHUTDOWN COMPLETE ====")
        except Exception as e:
            logger.exception(f"Exception during provider shutdown: {e}")
        finally:
            # DO NOT re-enable GC here! That would trigger collection which
            # calls destructors on MLC engine objects, causing OpenCL hangs
            # for models like Qwen2.5-7B. This is a daemon thread, so any
            # accumulated garbage will be cleaned up when process exits.
            done.set()
    
    # Use daemon thread so Python won't wait for it on exit
    shutdown_thread = threading.Thread(target=do_shutdown, daemon=True, name="ProviderShutdown")
    shutdown_thread.start()
    
    start = time.time()
    # Wait for shutdown with timeout, but don't block forever
    if not done.wait(timeout=timeout_sec):
        elapsed = time.time() - start
        print(f"==== SHUTDOWN TIMEOUT after {elapsed:.2f}s ====")
        logger.warning(
            f"Provider shutdown timed out after {elapsed:.2f}s. "
            "Engine may be stuck (e.g., OpenCL/TVM). Abandoning cleanup - daemon thread will be killed on exit."
        )
        # Don't wait - the daemon thread will be killed on process exit
        return
    
    elapsed = time.time() - start
    print(f"==== SHUTDOWN COMPLETED IN {elapsed:.2f}s ====")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup - install exception handlers
    threading.excepthook = handle_thread_exception
    logger.info("Installed global thread exception handler")
    
    # Install asyncio exception handler
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(_asyncio_exc_handler)
    logger.info("Installed asyncio exception handler")
    
    # Install force-exit handler to prevent indefinite hangs
    _install_force_exit_handler(max_shutdown_sec=5.0)
    logger.info("Installed force-exit handler for shutdown protection")
    
    yield
    
    # Shutdown - cleanup all cached model engines with timeout and cancellation
    provider = app.state.gen_svc.provider
    await shutdown_provider_with_timeout(provider, timeout_sec=3.0)


def create_app():
    """Create and configure the FastAPI application with DI"""
    from tether_service.core.config import load_settings_legacy
    from tether_service.core.factory import load
    from tether_service.protocol.service.generation_service import GenerationService
    from tether_service.config.settings import load_settings as load_settings_v2
    from tether_service.engine import Engine

    settings = load_settings_legacy()
    # NOTE: legacy dict-based provider/parser/store/tool_registry construction
    # below is preserved for one cycle for diff readability and to avoid mixing
    # this PR's scope with p2-cleanup. It is no longer referenced after the
    # Engine.from_settings call further down. p2-cleanup will delete this block.
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

    # Phase 2 (p2-engine-class): build the Engine from typed Settings. The
    # Engine class has the same method surface as the old GenerationService,
    # so existing routers (chat.py, sessions.py, models.py, health.py) work
    # unchanged. Per _synthesis.md §4 Phase 2 step 22.
    settings_v2 = load_settings_v2()
    gen_service = Engine.from_settings(settings_v2)
    app = FastAPI(lifespan=lifespan)
    # store service on app state
    app.state.gen_svc = gen_service
    # Phase 2 (p2-settings): also expose typed Settings on app.state. Existing
    # dict-based wiring above is unchanged; p2-cleanup migrates consumers.
    app.state.settings = settings_v2

    # Create a new APIRouter for versioning
    v1_router = APIRouter(prefix="/api/v1")

    # include routers
    v1_router.include_router(chat_router)
    v1_router.include_router(health_router)
    v1_router.include_router(models_router)
    v1_router.include_router(sessions_router)

    app.include_router(v1_router)
    return app