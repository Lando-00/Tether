"""Verify ``create_app()`` no longer constructs ``gen_svc`` via the legacy
dict-based wiring.

Per _synthesis.md §4 Phase 2 step 24 (typed Settings + Engine.from_settings
are the sole composition path).
"""
from __future__ import annotations

import inspect

import tether_service.app.http.api as api_mod
from tether_service.app.http.api import create_app
from tether_service.engine import Engine


def test_create_app_uses_engine_from_settings():
    app = create_app()
    assert isinstance(app.state.gen_svc, Engine)


def test_create_app_no_legacy_dict_construction():
    """Source of api.py must not contain ``load_settings_legacy`` (the
    legacy dict-based loader) anywhere."""
    src = inspect.getsource(api_mod)
    assert "load_settings_legacy" not in src, (
        "api.py still references load_settings_legacy — the legacy dict "
        "wiring must be deleted (Phase 2 step 24)."
    )


def test_create_app_imports():
    """The dead imports from the legacy wiring (cast, ModelProvider,
    StreamParser, SessionStore, GenerationService, load) must be gone."""
    src = inspect.getsource(api_mod)
    forbidden = [
        "from typing import cast",
        "from tether_service.core.interfaces import ModelProvider",
        "from tether_service.core.interfaces import StreamParser",
        "from tether_service.core.interfaces import SessionStore",
        "from tether_service.protocol.service.generation_service import GenerationService",
        "from tether_service.core.factory import load",
    ]
    for needle in forbidden:
        assert needle not in src, f"api.py still imports legacy symbol: {needle!r}"
