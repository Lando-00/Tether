"""Concurrent startup/shutdown helpers for the tool registry.

Phase 4 step 41 (synthesis §4) ships these alongside the
:class:`tether.tools.base.BaseTool` lifecycle hooks. Both helpers
run in a regular asyncio event loop — they are NOT the daemon-thread
shutdown path used for hardware-backed providers (see
:mod:`tether.runtime.hw_watchdog`), so the GC-disable rule
(synthesis §13.2 R5) does not apply here.

Two functions:

* :func:`startup_all` — concurrent ``startup()`` calls via
  :func:`asyncio.gather`. Required-tool failures raise
  :class:`RuntimeError`; optional-tool failures are returned in a dict
  for the caller to log and drop.
* :func:`shutdown_all` — concurrent best-effort ``shutdown()`` via
  ``asyncio.gather(..., return_exceptions=True)``. Failures are logged
  and returned but never raised.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, Mapping, Optional

from tether.core.interfaces import Tool

logger = logging.getLogger(__name__)


async def startup_all(
    tools: Mapping[str, Tool],
    *,
    fail_fast_required: bool = True,
) -> Dict[str, Exception]:
    """Run ``startup()`` on every tool concurrently.

    Concurrency: a single :func:`asyncio.gather` call lets long-running
    network ``startup()`` methods overlap (e.g., a Brave-API health
    ping). The whole gather waits for all coroutines before returning,
    even on a required-tool failure — that's what synthesis §13.2 R5
    means by "after gather() completes": no required-tool error
    short-circuits the still-running optional tools.

    Args:
        tools: Mapping of registry-name → tool instance.
        fail_fast_required: When ``True`` (default), raises
            :class:`RuntimeError` after the gather if any tool whose
            class declares ``REQUIRED = True`` raised during startup.
            Set to ``False`` for diagnostic introspection (e.g., a
            ``/admin`` endpoint that wants to inspect failures without
            aborting). Per synthesis §4 Phase 4 step 41.

    Returns:
        Dict mapping ``tool_name → exception`` for every tool whose
        ``startup()`` raised. Tools that succeed are not in the dict.
        Callers (Engine.__aenter__) use this to drop optional failures
        from the registry while keeping the engine alive.

    Raises:
        RuntimeError: When a REQUIRED tool fails and
            ``fail_fast_required`` is ``True``. The chained exception
            is the first required-tool failure.
    """
    if not tools:
        return {}

    names = list(tools.keys())
    coros = [tools[n].startup() for n in names]
    results = await asyncio.gather(*coros, return_exceptions=True)

    failures: Dict[str, Exception] = {}
    for name, result in zip(names, results):
        if isinstance(result, BaseException):
            if isinstance(result, Exception):
                failures[name] = result
            else:
                # Re-raise non-Exception BaseException (KeyboardInterrupt,
                # SystemExit) immediately — we're not catching those.
                raise result

    if fail_fast_required:
        for name, exc in failures.items():
            tool = tools[name]
            if getattr(type(tool), "REQUIRED", False):
                raise RuntimeError(
                    f"Required tool {name!r} failed during startup: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

    for name, exc in failures.items():
        if not getattr(type(tools[name]), "REQUIRED", False):
            logger.warning(
                "Optional tool %r failed during startup: %s: %s",
                name,
                type(exc).__name__,
                exc,
            )

    return failures


async def shutdown_all(tools: Mapping[str, Tool]) -> Dict[str, Exception]:
    """Best-effort concurrent ``shutdown()`` on every tool.

    Failures are logged and returned in the dict, NEVER raised. Partial
    cleanup is always preferable to letting a tool's bad shutdown abort
    a shutdown that should also clean up the provider/store.

    Args:
        tools: Mapping of registry-name → tool instance.

    Returns:
        Dict mapping ``tool_name → exception`` for every tool whose
        ``shutdown()`` raised.
    """
    if not tools:
        return {}

    names = list(tools.keys())
    coros = [tools[n].shutdown() for n in names]
    results = await asyncio.gather(*coros, return_exceptions=True)

    failures: Dict[str, Exception] = {}
    for name, result in zip(names, results):
        if isinstance(result, BaseException):
            if isinstance(result, Exception):
                failures[name] = result
                logger.warning(
                    "Tool %r shutdown failed: %s: %s",
                    name,
                    type(result).__name__,
                    result,
                )
            # Non-Exception BaseException during shutdown is unusual but
            # we still don't raise — letting subsequent cleanup proceed.
    return failures


__all__ = ["startup_all", "shutdown_all"]
