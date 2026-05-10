"""Typed runtime configuration for Tether.

This module ships the Pydantic v2 ``Settings`` model (the new typed loader)
alongside its sub-models. The legacy ``dict``-returning loader at
``tether_service.core.config.load_settings_legacy`` remains available for one
deprecation cycle while call sites migrate (see ``p2-cleanup``).

Citations:
    - _synthesis.md §13.4 M6 — StrictModel base for sub-models.
    - _synthesis.md §10.8 #2 — connectors / inbox placeholder sections.
    - A1 DI/Config investigation — sub-model field designs.
"""
from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from tether_service.config._strict import StrictModel


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class HttpSettings(StrictModel):
    """Top-level host/port for the HTTP server (``http:`` section)."""

    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)


class SystemSettings(StrictModel):
    """``system:`` section — system prompt prefix."""

    prompt: str = ""


class ProviderSpec(StrictModel):
    """``impl: ..., args: {...}`` shape used in ``providers.*`` sections."""

    impl: str
    args: dict[str, Any] = Field(default_factory=dict)


class ProvidersSettings(StrictModel):
    """``providers:`` section — model / parser / session_store implementations."""

    model: ProviderSpec
    parser: ProviderSpec
    session_store: ProviderSpec


class ToolSpec(StrictModel):
    """One entry in ``tools.registry`` (name + impl + optional args)."""

    name: str
    impl: str
    args: dict[str, Any] = Field(default_factory=dict)


class WebSearchTimeouts(StrictModel):
    connect_sec: float = 2
    read_sec: float = 6
    total_sec: float = 15


class WebSearchRetries(StrictModel):
    max_attempts: int = 3
    backoff_base_sec: float = 0.5


class WebSearchDefaults(StrictModel):
    count: int = 5
    max_count: int = 20
    country: str = "us"
    search_lang: str = "en"
    freshness: Optional[str] = None


class WebSearchSettings(StrictModel):
    """``tools.web_search`` subtree."""

    provider: str = "brave"
    timeouts: WebSearchTimeouts = Field(default_factory=WebSearchTimeouts)
    retries: WebSearchRetries = Field(default_factory=WebSearchRetries)
    defaults: WebSearchDefaults = Field(default_factory=WebSearchDefaults)


class ToolsSettings(StrictModel):
    """``tools:`` section."""

    registry: List[ToolSpec] = Field(default_factory=list)
    enabled: List[str] = Field(default_factory=list)
    disabled: List[str] = Field(default_factory=list)
    """Phase 4 (synthesis §4 step 42): tools opt-out list for the
    discover path. When ``registry`` is empty, ToolRegistry auto-
    discovers @tool-decorated classes; this list filters them out by
    name. Has no effect when ``registry`` is non-empty (legacy path).
    """
    call_contract: Literal["marker_json"] = "marker_json"
    continue_after_tool: bool = True
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)


class LimitsSettings(StrictModel):
    """``limits:`` section."""

    tool_timeout_sec: int = Field(default=15, ge=1)
    max_tool_loops: int = Field(default=5, ge=1, le=50)
    auto_reload_on_fatal_error: bool = True
    # Phase 5 step 52 (synthesis §3.5): policy fields. Stored as plain
    # strings here so the YAML schema stays a flat key-value mapping;
    # ``OrchestratorConfig.from_settings`` coerces to the typed enums.
    # User-ratified defaults: emit_limit_event + feed_back_to_model.
    loop_limit_policy: Literal["raise", "emit_limit_event"] = Field(
        default="emit_limit_event",
        description="LoopLimitPolicy enum value.",
    )
    tool_error_policy: Literal["break_loop", "feed_back_to_model"] = Field(
        default="feed_back_to_model",
        description="ToolErrorPolicy enum value.",
    )


class ContextSettings(StrictModel):
    """``context:`` section — thinking persistence flags."""

    save_thinking: bool = True
    include_thinking_in_history: bool = False


class StreamParserSettings(StrictModel):
    strict_balancing: bool = True


class StreamSettings(StrictModel):
    """``stream:`` section."""

    parser: StreamParserSettings = Field(default_factory=StreamParserSettings)


class AuditLogSettings(StrictModel):
    """``security.audit_log:`` sub-model.

    Phase 7 step 73: tool_audit table receives one row per tool call.
    By default args are stored as SHA-256 hash (privacy-preserving).
    Set ``store_args=True`` to also persist raw args_json (debug-only;
    contains potentially-PII tool input).

    Synthesis §3.6 + B5 step 7.
    """

    store_args: bool = False


class SecuritySettings(StrictModel):
    """``security:`` section. Phase 4 will populate ``capability_allowlist``."""

    capability_allowlist: Optional[List[str]] = None
    audit_log: AuditLogSettings = Field(default_factory=AuditLogSettings)
    tool_result_max_bytes: int = Field(
        default=256 * 1024,  # 256 KB
        description=(
            "Reject tool results exceeding this byte size; replaces with a "
            "structured error so the model can retry with smaller args. "
            "Phase 7 step 77."
        ),
        ge=1024,  # at least 1 KB
    )


class LogFileSettings(StrictModel):
    """``observability.logs.file:`` sub-model.

    ``path`` (default None) resolves via platformdirs.user_log_dir to a
    cross-platform default (e.g., ``%LOCALAPPDATA%\\Tether\\Logs\\tether.jsonl``
    on Windows, ``~/.cache/Tether/log/tether.jsonl`` on Linux).

    Synthesis §3 (observability), §4 Phase 7 step 67.
    """

    enabled: bool = True
    path: Optional[str] = None  # None -> platformdirs default


class LogsSettings(StrictModel):
    """``observability.logs:`` sub-model.

    Synthesis §3 (observability), §4 Phase 7 step 67.
    """

    level: str = "INFO"
    file: LogFileSettings = Field(default_factory=LogFileSettings)
    console: bool = True
    format: str = "json"  # "json" | "console"


class OTelSettings(StrictModel):
    """``observability.otel:`` sub-model.

    OpenTelemetry tracing adapter (Phase 7 step 76). Off by default so that
    the OTel packages are never imported at runtime unless explicitly enabled.

    Requires ``pip install tether[otel]`` when ``enabled=True``.

    Synthesis §3 (observability), B3 step 10.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Enable OpenTelemetry tracing adapter. Off by default. "
            "Requires `tether[otel]` extra."
        ),
    )
    service_name: str = Field(
        default="tether",
        description="OTel resource service.name attribute.",
    )
    exporter: Literal["console", "otlp_http", "otlp_grpc"] = Field(
        default="console",
        description=(
            "OTel span exporter. 'console' prints spans to stdout for local "
            "debugging; 'otlp_http'/'otlp_grpc' send to an OTel collector at "
            "exporter_endpoint."
        ),
    )
    exporter_endpoint: Optional[str] = Field(
        default=None,
        description=(
            "OTLP collector endpoint when exporter is 'otlp_http' or 'otlp_grpc'. "
            "If None and exporter requires it, raises at startup."
        ),
    )
    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Span sampling rate. 1.0 = sample all, 0.0 = sample none."
        ),
    )


class ObservabilitySettings(StrictModel):
    """``observability:`` section. Phase 7 structured logging foundation."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    logs: LogsSettings = Field(default_factory=LogsSettings)
    otel: OTelSettings = Field(
        default_factory=OTelSettings,
        description="OpenTelemetry tracing adapter. See OTelSettings.",
    )
    provider_chunk_log_sample: int = Field(
        default=50,
        description=(
            "Sample rate for provider.stream.chunk log events. "
            "1 = log every chunk, 50 = log first + every 50th chunk, "
            "0 = disable chunk logging entirely. "
            "Synthesis §3 (observability), §4 Phase 7 step 72."
        ),
        ge=0,
    )


class SqliteSettings(StrictModel):
    """``storage.sqlite:`` sub-model.

    ``dsn`` (default ``None``) resolves to
    ``platformdirs.user_data_dir('Tether')/tether.db`` via
    :meth:`StorageSettings.resolved_dsn`. Library callers can override
    by passing ``dsn="sqlite:///path/to/db"`` at config load time.

    Synthesis §3.6, §4 Phase 6 step 60.
    """

    dsn: Optional[str] = None


class StorageSettings(StrictModel):
    """``storage:`` section.

    Phase 6 step 60: replaces the previous hardcoded
    ``sqlite:///./data/tether.db`` with platformdirs-resolved default
    so library users importing tether_service don't write into the
    consumer's CWD. Synthesis §3.6, §10.8 #2.
    """

    sqlite: SqliteSettings = Field(default_factory=SqliteSettings)

    def resolved_dsn(self) -> str:
        """Return the effective SQLite DSN (configured or defaulted).

        Resolves SqliteSettings.dsn ``None`` to
        ``f"sqlite:///{platformdirs.user_data_dir('Tether')}/tether.db"``.
        Cross-platform: on Windows ``%LOCALAPPDATA%\\Tether``; on
        Linux/macOS XDG-compliant.

        ``platformdirs`` is lazy-imported here to preserve the library-first
        invariant: ``import tether_service`` must not trigger it.
        """
        if self.sqlite.dsn is not None:
            return self.sqlite.dsn

        from pathlib import Path

        import platformdirs

        data_dir = Path(platformdirs.user_data_dir("Tether", appauthor=False))
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path = data_dir / "tether.db"
        return f"sqlite:///{db_path.as_posix()}"


class ConnectorSpec(StrictModel):
    """One entry in ``connectors.registry`` (per connector spec §3.7).

    Mirrors :class:`ProviderSpec` + :class:`ToolSpec` shape: a dotted-path
    ``impl`` plus optional ``args`` filtered against the target ``__init__``
    by ``tether_service.core.factory.load`` (so adding a new arg doesn't
    crash older configs). ``enabled`` defaults to True so YAML entries are
    live unless explicitly disabled.

    Phase 4.5 ships the schema; concrete ``connectors.registry`` entries
    arrive in Phase 2a (WhatsApp) / 2b (Gmail) sessions.
    """

    impl: str
    args: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class ConnectorsSettings(StrictModel):
    """``connectors:`` section (per connector spec §3.7).

    ``registry`` maps stable connector ids (``"whatsapp"``, ``"gmail"``,
    ``"echo"``) to their :class:`ConnectorSpec`. The future
    ``ConnectorRegistry`` (``p4_5-connector-registry``) consumes this map
    at app construction.

    Phase 4.5 ships the schema; concrete entries arrive in Phase 2a / 2b.
    """

    registry: dict[str, ConnectorSpec] = Field(default_factory=dict)


class OrchestratorSettings(StrictModel):
    """``orchestrator:`` section — strategy registry.

    Mirrors the ``tools.registry`` pattern. The registry maps mode
    names (e.g., "chat", "research") to dotted impl paths. Engine.chat
    resolves the requested mode at call time via
    ``protocol.orchestration.registry.resolve_orchestrator_class``.

    Briefing §2 Seam B item 4; §5 anti-pattern (no auto-routing).
    """

    default: str = Field(
        default="chat",
        description="Mode used when the request omits it.",
    )
    registry: Dict[str, str] = Field(
        default_factory=lambda: {
            "chat": "tether_service.protocol.orchestration.chatty.ChattyAgentOrchestrator",
            "research": "tether_service.protocol.orchestration.notebook.NotebookOrchestrator",
        },
        description="Mode -> dotted impl path (e.g., 'pkg.module.Class').",
    )


class InboxSettings(StrictModel):
    """``inbox:`` section (per connector spec §3.7 + §3.4).

    Phase 4.5 ships the schema; the SqliteInbox impl lands in Phase 6.5.
    The defaults match the connector spec recommendations: 30-day
    retention, 64 KiB payload cap, 512-char summary cap. Connectors are
    responsible for clamping to ``max_payload_bytes`` /
    ``max_summary_chars`` when constructing :class:`InboundEvent` values
    — the inbox layer validates as defense in depth (Phase 6.5).
    """

    enabled: bool = True
    retention_days: int = Field(default=30, ge=1, le=3650)
    max_payload_bytes: int = Field(default=64_000, ge=1024)
    max_summary_chars: int = Field(default=512, ge=64)


# ---------------------------------------------------------------------------
# Top-level Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Tether's typed runtime configuration.

    Loaded from (in order, deep-merged):
      1. ``tether_service/config/default.yml`` (package data; immutable).
      2. ``tether_service/config/dev.yml`` (if present; deep-merged or
         full-replaced via the ``_replaces_default: true`` flag).
      3. Environment variables ``TETHER__SECTION__SUB__KEY`` (parsed as YAML
         for type coercion).

    ``extra="forbid"`` on every sub-model catches typos in YAML; ``frozen=True``
    makes the constructed Settings immutable so callers cannot mutate it as a
    global side effect.

    Citations: _synthesis.md §13.4 M6, §10.8 #2.
    """

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=True,
        env_prefix="TETHER__",
        env_nested_delimiter="__",
    )

    http: HttpSettings = Field(default_factory=HttpSettings)
    system: SystemSettings = Field(default_factory=SystemSettings)
    providers: ProvidersSettings
    tools: ToolsSettings = Field(default_factory=ToolsSettings)
    limits: LimitsSettings = Field(default_factory=LimitsSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    stream: StreamSettings = Field(default_factory=StreamSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    connectors: ConnectorsSettings = Field(default_factory=ConnectorsSettings)
    inbox: InboxSettings = Field(default_factory=InboxSettings)
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _apply_env_overrides(cfg: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    """Mirror legacy semantics: TETHER__a__b=val -> cfg['a']['b']=parsed(val).

    Values are parsed via ``yaml.safe_load`` so numbers / bools / lists work.
    Keys are lowercased to match YAML conventions.
    """
    prefix = "TETHER__"
    out = cfg
    for key, val in env.items():
        if not key.startswith(prefix):
            continue
        parts = [p.strip().lower() for p in key[len(prefix):].split("__") if p.strip()]
        if not parts:
            continue
        sub = out
        for p in parts[:-1]:
            sub = sub.setdefault(p, {})
        try:
            parsed = yaml.safe_load(val)
        except Exception:
            parsed = val
        sub[parts[-1]] = parsed
    return out


def load_settings(
    *,
    default_yaml: Optional[Path] = None,
    overlay_yaml: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
) -> Settings:
    """Pure loader returning a validated ``Settings`` object.

    Args:
        default_yaml: Path to base YAML. Defaults to packaged ``default.yml``.
        overlay_yaml: Optional dev overlay. Defaults to ``dev.yml`` next to the
            packaged default if it exists, else ``None``. Honors
            ``_replaces_default: true`` flag.
        env: Mapping of environment variables. Defaults to ``os.environ``.
            Pass an explicit dict in tests to avoid global env coupling.

    Tests may pass explicit ``default_yaml`` / ``overlay_yaml`` paths; production
    callers invoke with no arguments.
    """
    pkg_root = resources.files("tether_service.config")

    if default_yaml is None:
        cfg = _read_yaml(Path(str(pkg_root / "default.yml")))
    else:
        cfg = _read_yaml(default_yaml)

    if overlay_yaml is None:
        dev_file = pkg_root / "dev.yml"
        if dev_file.is_file():
            overlay_yaml = Path(str(dev_file))

    if overlay_yaml is not None and overlay_yaml.is_file():
        dev_cfg = _read_yaml(overlay_yaml)
        if dev_cfg.pop("_replaces_default", False):
            cfg = dev_cfg
        else:
            cfg = _deep_merge(cfg, dev_cfg)

    env_map = env if env is not None else dict(os.environ)
    cfg = _apply_env_overrides(cfg, env_map)

    return Settings.model_validate(cfg)
