"""ConnectorRegistry.

Boot-time validation of connector tool names per connector spec §3.3:

* Each connector's tool name MUST start with ``f"{connector.id}_"``.
* No two connectors expose the same tool name.
* No connector tool collides with the in-tree ``tools.registry``.

Validation reuses :func:`tether.core.registry_validator.validate_unique_names`
(synthesis §13.4 M5) so the wording of error messages and the contract for
"a valid set of registered names" stays in one place.

Lifecycle: :meth:`start_connector` / :meth:`stop_connector` /
:meth:`start_all` / :meth:`stop_all`. ``stop_connector`` enforces a 2 s
cooperative budget (connector spec §3.3 step 6); blocking native cleanup
remains the connector's responsibility — concrete connectors with native
cleanup needs are expected to mirror the daemon-thread + force-exit pattern
documented on :class:`tether.connectors.base.Connector` (Phase 3
``HardwareWatchdog``).

OAuth state for the spec §3.8 callback handshake is held in a small
in-memory TTL cache (``maxsize=8``, ``ttl=300 s``); stdlib only — no
``cachetools`` dependency (R6 anti-overengineering).

Citations:
    - Synthesis §4 Phase 4.5 steps 47b-47c, §13.4 M5, §13.5 M5 callers,
      R6.
    - Connector spec §3.3 (registry validation + lifecycle), §3.6 (data
      layout), §3.8 (OAuth callback).
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tether.connectors.base import Connector
from tether.connectors.types import ConnectorState
from tether.core.interfaces import Tool
from tether.core.registry_validator import validate_unique_names

logger = logging.getLogger(__name__)


class _OAuthStateCache:
    """Tiny TTL cache for OAuth ``state`` tokens (connector spec §3.8).

    ``maxsize`` bounds concurrent in-flight logins (default 8 — generous
    for a single-user device); ``ttl`` bounds how long a callback can
    arrive after ``begin_login`` (default 300 s — matches typical OAuth
    consent-screen timeouts). Stdlib-only per R6: a 30-line dict is more
    obviously correct than pulling in ``cachetools`` for one cache.

    Eviction policy:

    * ``set``: drop entries whose ``ts`` is older than ``ttl``, then if at
      capacity drop the oldest remaining entry, then insert the new one.
    * ``get`` / ``pop``: return ``None`` for unknown OR expired keys, and
      delete expired keys lazily on read.
    """

    def __init__(self, maxsize: int = 8, ttl: float = 300.0) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        if ttl <= 0:
            raise ValueError(f"ttl must be > 0, got {ttl}")
        self._maxsize = maxsize
        self._ttl = ttl
        self._store: Dict[str, "tuple[float, Any]"] = {}

    def set(self, key: str, value: Any) -> None:
        now = time.monotonic()
        # 1. Evict expired entries.
        expired = [k for k, (ts, _) in self._store.items() if now - ts > self._ttl]
        for k in expired:
            del self._store[k]
        # 2. If updating an existing key, just overwrite (no capacity change).
        if key in self._store:
            self._store[key] = (now, value)
            return
        # 3. If at capacity, evict the oldest entry.
        if len(self._store) >= self._maxsize:
            oldest_key = min(self._store.items(), key=lambda kv: kv[1][0])[0]
            del self._store[oldest_key]
        self._store[key] = (now, value)

    def get(self, key: str) -> Optional[Any]:
        item = self._store.get(key)
        if item is None:
            return None
        ts, value = item
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def pop(self, key: str) -> Optional[Any]:
        value = self.get(key)
        if value is None:
            return None
        # ``get`` already deleted on expiry; safe to delete here.
        self._store.pop(key, None)
        return value

    def __len__(self) -> int:
        return len(self._store)


class ConnectorRegistry:
    """Connector lifecycle + tool aggregation. Boot-time validation.

    Args:
        connectors: List of :class:`Connector` instances. Each must have a
            unique ``id``.
        tool_names: Set of in-tree tool names (typically
            ``ToolRegistry.all().keys()``). Used as the initial ``forbidden``
            set for the M5 prefix-collision check so connector tools cannot
            shadow an in-tree tool.
        data_dir: Optional override for ``data/connectors/<id>/`` paths. If
            ``None``, defaults to ``platformdirs.user_data_dir("Tether") /
            "connectors"`` when ``platformdirs`` is importable; otherwise
            ``./data/connectors``. Directories are lazy-created on first
            :meth:`start_connector` call (connector spec §3.6).

    Validation at construction (connector spec §3.3):
        * ``connectors`` ids are unique.
        * Every connector's ``tools().keys()`` start with ``f"{cid}_"``.
        * No two connectors expose the same tool name.
        * No connector tool collides with ``tool_names``.

    Synthesis §13.4 M5; connector spec §3.3.
    """

    def __init__(
        self,
        connectors: List[Connector],
        tool_names: Optional[Set[str]] = None,
        *,
        data_dir: Optional[Path] = None,
    ) -> None:
        if tool_names is None:
            tool_names = set()

        # 1. connector_id -> Connector; reject duplicate ids and empty ids.
        self._connectors: Dict[str, Connector] = {}
        for c in connectors:
            if not c.id:
                raise ValueError(
                    f"Connector {type(c).__name__} has empty id; concrete "
                    f"connectors must override the id ClassVar."
                )
            if c.id in self._connectors:
                raise ValueError(f"Duplicate connector id: {c.id!r}")
            self._connectors[c.id] = c

        # 2. Validate each connector's tool names per M5. Forbidden set
        #    grows as we accept connectors: this catches A's tool name
        #    colliding with B's even when neither violates ``tool_names``.
        #
        # Phase 4.5 follow-up (rubber-duck consensus, gpt-5.5 CONCERN):
        # cache the per-connector ``tools()`` result so we call it
        # exactly ONCE per connector (the previous code called it twice
        # — here for validation, and again below for aggregation). A
        # non-idempotent or eventually-failing ``tools()`` could pass
        # validation and then yield a different dict during aggregation,
        # producing inconsistent registry state silently. ``tools()``
        # raising during construction is also wrapped here with a clear
        # ``Connector '<cid>' tools() raised: ...`` message so the
        # boot-time failure points at the offending connector.
        accumulated_forbidden: Set[str] = set(tool_names)
        cached_tools_per_connector: Dict[str, Dict[str, Tool]] = {}
        for cid, conn in self._connectors.items():
            try:
                conn_tools = conn.tools()
            except Exception as exc:
                raise ValueError(
                    f"Connector {cid!r} tools() raised: {exc}"
                ) from exc
            cached_tools_per_connector[cid] = conn_tools
            try:
                validate_unique_names(
                    conn_tools,
                    require_prefix=f"{cid}_",
                    forbidden=accumulated_forbidden,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Connector {cid!r} tool registration failed: {exc}"
                ) from exc
            accumulated_forbidden.update(conn_tools.keys())

        # 3. Aggregate from the cached dicts (single-pass; no second
        #    ``tools()`` call). Safe to merge naively because step 2
        #    proved no cross-connector collisions exist.
        self._all_tools: Dict[str, Tool] = {}
        for cid in self._connectors:
            self._all_tools.update(cached_tools_per_connector[cid])

        # 4. Resolve data_dir lazily; per spec §3.6 the directories are
        #    only materialised on first ``start_connector``.
        if data_dir is None:
            try:
                from platformdirs import user_data_dir  # type: ignore[import-not-found]

                data_dir = Path(user_data_dir("Tether")) / "connectors"
            except ImportError:
                data_dir = Path("data") / "connectors"
        self._data_dir = data_dir

        # 5. OAuth state cache for spec §3.8 callback (state token -> any
        #    connector-defined payload). Owned by the registry so the
        #    future HTTP route can look it up without depending on
        #    individual connector instances.
        self._oauth_state = _OAuthStateCache(maxsize=8, ttl=300.0)

        logger.info(
            "ConnectorRegistry: %d connector(s), %d aggregated tool(s)",
            len(self._connectors),
            len(self._all_tools),
        )

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    def get(self, connector_id: str) -> Connector:
        """Return the connector for ``connector_id`` or raise ``KeyError``."""
        if connector_id not in self._connectors:
            raise KeyError(f"connector not registered: {connector_id!r}")
        return self._connectors[connector_id]

    def all(self) -> List[Connector]:
        """Return all registered connectors as a list."""
        return list(self._connectors.values())

    def aggregate_tools(self) -> Dict[str, Tool]:
        """Return a fresh dict of every connector tool, keyed by name.

        Validated unique at construction; callers may freely mutate the
        returned dict without affecting the registry.
        """
        return dict(self._all_tools)

    def names(self) -> Set[str]:
        """Return the set of registered connector tool names.

        Useful for downstream validators that want to assert no other
        registry will collide with a connector tool.
        """
        return set(self._all_tools.keys())

    @property
    def oauth_state(self) -> _OAuthStateCache:
        """Return the OAuth-state TTL cache (connector spec §3.8)."""
        return self._oauth_state

    @property
    def data_dir(self) -> Path:
        """Return the configured data root for connector data dirs.

        ``data/connectors/<id>/`` is materialised on first
        :meth:`start_connector` call.
        """
        return self._data_dir

    def _ensure_connector_data_dir(self, connector_id: str) -> Path:
        """Lazy-create ``data/connectors/<id>/`` (connector spec §3.6)."""
        path = self._data_dir / connector_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start_connector(self, connector_id: str) -> None:
        """Start a single connector. Lazy-creates its data directory.

        ``Connector.start()`` is contractually idempotent (connector spec
        §3.1) so this method is safe to call repeatedly; the data dir
        ``mkdir(..., exist_ok=True)`` mirrors that.
        """
        conn = self.get(connector_id)
        self._ensure_connector_data_dir(connector_id)
        await conn.start()
        logger.info("Started connector: %s", connector_id)

    async def stop_connector(
        self, connector_id: str, *, timeout_sec: float = 2.0
    ) -> None:
        """Stop a single connector with a bounded cooperative budget.

        Per connector spec §3.3 step 6: ``stop()`` must complete within
        ``timeout_sec`` (default 2 s) cooperatively; this method abandons
        the coroutine on timeout and logs a warning. Connectors with
        potentially blocking native cleanup MUST themselves use the
        daemon-thread + force-exit pattern that
        ``shutdown_provider_with_timeout()`` uses for the MLC provider
        (see :class:`tether.connectors.base.Connector` docstring).

        Exceptions raised by ``stop()`` are logged but never re-raised so
        one failing connector cannot block shutdown of the others when
        called via :meth:`stop_all`.
        """
        conn = self.get(connector_id)
        try:
            await asyncio.wait_for(conn.stop(), timeout=timeout_sec)
            logger.info(
                "Stopped connector: %s (within %.1fs budget)",
                connector_id,
                timeout_sec,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Connector %s stop() exceeded %.1fs cooperative budget; "
                "abandoning. (Connector responsible for any blocking "
                "native cleanup per spec §3.3.)",
                connector_id,
                timeout_sec,
            )
        except Exception as exc:  # noqa: BLE001 - logged + swallowed by design
            logger.exception("Connector %s stop() raised: %s", connector_id, exc)

    async def start_all(self) -> Dict[str, Optional[BaseException]]:
        """Start every currently-READY connector concurrently.

        Connectors in ``UNCONFIGURED``, ``AUTHENTICATING``,
        ``LOGGED_OUT``, ``DEGRADED``, or ``ERROR`` are skipped — they
        need login (or recovery) first, so calling ``start()`` on them
        would either no-op or raise ``ConnectorNotConfiguredError``.

        Phase 4.5 follow-up (rubber-duck consensus, xhigh OBSERVATION):
        previously this method started ALL connectors regardless of
        state, which contradicted spec §3.3 step 4 (and Engine.__aenter__
        already filtered to READY). Aligned with Engine.__aenter__ so
        library users calling ``start_all`` directly get the same
        contract as the production HTTP path.

        Returns a ``{connector_id: None | exception}`` dict for the
        connectors that were *attempted* — i.e. those whose
        ``auth_status`` reported READY. Connectors filtered out (not
        READY) do NOT appear in the result. Failures from
        ``start_connector`` are logged but never re-raised; failures
        from ``auth_status`` itself are logged and the connector is
        recorded with the exception in the dict.

        To force-start every connector regardless of state (rare;
        useful for diagnostics or after a manual credential restore)::

            for conn in registry.all():
                await registry.start_connector(conn.id)
        """
        if not self._connectors:
            return {}

        results: Dict[str, Optional[BaseException]] = {}
        for cid, conn in self._connectors.items():
            try:
                status = await conn.auth_status()
            except Exception as exc:  # noqa: BLE001 - defensive
                logger.exception(
                    "start_all: auth_status(%s) failed: %s", cid, exc
                )
                results[cid] = exc
                continue
            if status.state is not ConnectorState.READY:
                logger.debug(
                    "start_all: skipping %s (auth_status=%s, not READY)",
                    cid,
                    status.state.value,
                )
                continue
            try:
                await self.start_connector(cid)
                results[cid] = None
            except Exception as exc:  # noqa: BLE001 - logged + recorded
                logger.exception(
                    "start_all: start_connector(%s) failed: %s", cid, exc
                )
                results[cid] = exc
        return results

    async def stop_all(self, *, timeout_sec: float = 2.0) -> None:
        """Stop every registered connector concurrently.

        Each connector gets the same ``timeout_sec`` budget; failures are
        already logged + swallowed inside :meth:`stop_connector`, so this
        call always completes within ``~timeout_sec`` regardless of how
        many connectors hang.
        """
        if not self._connectors:
            return
        await asyncio.gather(
            *(
                self.stop_connector(cid, timeout_sec=timeout_sec)
                for cid in self._connectors
            ),
            return_exceptions=True,
        )


__all__ = ["ConnectorRegistry"]
