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
from typing import Any, List, Literal, Optional

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


class ContextSettings(StrictModel):
    """``context:`` section — thinking persistence flags."""

    save_thinking: bool = True
    include_thinking_in_history: bool = False


class StreamParserSettings(StrictModel):
    strict_balancing: bool = True


class StreamSettings(StrictModel):
    """``stream:`` section."""

    parser: StreamParserSettings = Field(default_factory=StreamParserSettings)


class SecuritySettings(StrictModel):
    """``security:`` section. Phase 4 will populate ``capability_allowlist``."""

    capability_allowlist: Optional[List[str]] = None


class ObservabilitySettings(StrictModel):
    """``observability:`` section. Phase 7 will populate."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class StorageSettings(StrictModel):
    """``storage:`` section. Phase 6 will populate (DB path, retention, etc.).

    Currently ``providers.session_store.args.dsn`` is the source of truth; this
    placeholder anticipates the move per _synthesis.md §10.8 #2.
    """


class ConnectorsSettings(StrictModel):
    """``connectors:`` section. Phase 4.5 will populate (connector spec §3.7)."""


class InboxSettings(StrictModel):
    """``inbox:`` section. Phase 6.5 will populate."""


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
