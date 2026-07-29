"""Phase 3 follow-up: ``MLCProvider.unload_model`` uses EXACT-key match.

Replaces the previous substring match (``if model_name in key``) which
could unload the WRONG engine when one model name is a prefix of another
(e.g. ``Qwen3-4B`` vs ``Qwen3-4B-Instruct``). ``hw_reset`` calls
``unload_model`` directly, so the recovery path inherited the bug.

Synthesis §6 bug #15; rubber-duck consensus (gpt-5.5 BLOCKING + xhigh
BLOCKING-2 + 1m OBSERVATION).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tether.providers.mlc.provider import MLCProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(tmp_path: Path) -> MLCProvider:
    """Construct an MLCProvider rooted at tmp_path. Mirrors the helper
    in ``tests/unit/test_mlc_provider_hwlifecycle.py`` — ``__init__``
    only stores paths, no MLC native libs are touched."""
    models_root = tmp_path / "models"
    models_root.mkdir()
    (models_root / "libs").mkdir()
    return MLCProvider(models_root=str(models_root), device="auto", max_tokens=1024)


class _FakeEngine:
    """Sentinel for the engine cache — only ``state.async_streamers`` and
    ``terminate`` are touched by the unload path. We track which engine
    was terminated by recording its identity in a shared list."""

    class _State:
        async_streamers: dict = {}

    def __init__(self, label: str):
        self.label = label
        self.state = _FakeEngine._State()
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True

    def _abort(self, _rid):  # pragma: no cover — _abort_all_requests fast-path
        return None


def _patch_resolve_model_lib(monkeypatch, mapping: dict[str, str]):
    """Mock ``resolve_model_lib`` so unload_model can construct the
    canonical key without real .dll files. ``mapping`` maps ``model_name``
    to the fake lib path the provider will see.

    A model_name not in ``mapping`` raises ``ValueError`` (mirrors the
    real resolver's behavior when no DLL is found).
    """
    def fake_resolve(model_name: str, libs_dir: Path) -> str:
        if model_name not in mapping:
            raise ValueError(f"No matching model library found for {model_name}")
        return mapping[model_name]

    monkeypatch.setattr(
        "tether.providers.mlc.provider.resolve_model_lib",
        fake_resolve,
    )


# ---------------------------------------------------------------------------
# A2.1 — Exact-key match: substring siblings are NOT touched.
# ---------------------------------------------------------------------------


def test_unload_model_exact_match_only(tmp_path: Path, monkeypatch):
    """Pre-populate the cache with two engines whose canonical keys share
    a prefix. Unloading ``Qwen3-4B`` must terminate ONLY the exact match;
    ``Qwen3-4B-Instruct`` must remain in the cache untouched.

    Pre-fix behavior (substring match): ``unload_model("Qwen3-4B")``
    would iterate the dict and the first key containing ``Qwen3-4B`` as
    a substring (which is BOTH keys) would be popped. Depending on dict
    insertion order this could unload the wrong engine.
    """
    provider = _make_provider(tmp_path)

    fake_libs = {
        "Qwen3-4B": str(tmp_path / "models" / "libs" / "qwen3-4b.dll"),
        "Qwen3-4B-Instruct": str(tmp_path / "models" / "libs" / "qwen3-4b-instruct.dll"),
    }
    _patch_resolve_model_lib(monkeypatch, fake_libs)

    # Build canonical keys exactly as MLCProvider does.
    key_4b = (
        f"{provider.models_root / 'Qwen3-4B'}:{provider.device}:{fake_libs['Qwen3-4B']}"
    )
    key_4b_instruct = (
        f"{provider.models_root / 'Qwen3-4B-Instruct'}:{provider.device}:"
        f"{fake_libs['Qwen3-4B-Instruct']}"
    )

    engine_4b = _FakeEngine(label="4b")
    engine_4b_instruct = _FakeEngine(label="4b-instruct")
    with provider._cache_lock:
        provider._engine_cache[key_4b] = engine_4b
        provider._engine_cache[key_4b_instruct] = engine_4b_instruct

    # Act — unload exact match.
    result = provider.unload_model("Qwen3-4B")

    # Assert — exact match unloaded; sibling untouched.
    assert result is True
    assert engine_4b.terminated is True
    assert engine_4b_instruct.terminated is False
    # Cache: instruct still present, 4b removed.
    assert key_4b not in provider._engine_cache
    assert key_4b_instruct in provider._engine_cache
    assert provider._engine_cache[key_4b_instruct] is engine_4b_instruct


# ---------------------------------------------------------------------------
# A2.2 — Unknown model (not in cache) returns False, doesn't raise.
# ---------------------------------------------------------------------------


def test_unload_model_returns_false_for_unknown(tmp_path: Path, monkeypatch):
    """An unknown but well-formed model_name (resolves successfully but
    isn't in the cache) returns False, doesn't raise. Matches the legacy
    contract for callers that probe-then-unload."""
    provider = _make_provider(tmp_path)
    _patch_resolve_model_lib(
        monkeypatch,
        {"NotLoaded-Model": str(tmp_path / "models" / "libs" / "notloaded.dll")},
    )

    assert provider.unload_model("NotLoaded-Model") is False


# ---------------------------------------------------------------------------
# A2.3 — Path-traversal guard preserved (Phase 0A).
# ---------------------------------------------------------------------------


def test_unload_model_validates_path_traversal(tmp_path: Path):
    """An attacker-influenced model_name must be rejected by the
    ``_validate_model_name`` check before any cache lookup or filesystem
    access. Phase 0A security R-pathtraversal — preserved across the
    refactor."""
    provider = _make_provider(tmp_path)
    with pytest.raises(ValueError):
        provider.unload_model("../../../etc/passwd")


def test_unload_model_validates_path_separator_windows(tmp_path: Path):
    """Cross-OS coverage: backslash path separators rejected."""
    provider = _make_provider(tmp_path)
    with pytest.raises(ValueError):
        provider.unload_model("..\\windows\\system32")


# ---------------------------------------------------------------------------
# A2.4 — resolve_model_lib failure → False (don't crash callers).
# ---------------------------------------------------------------------------


def test_unload_model_resolve_lib_failure_returns_false(tmp_path: Path, monkeypatch):
    """If the library resolver fails (e.g. dist directory was wiped after
    load), ``unload_model`` returns False instead of raising. This matches
    the recovery contract — callers like ``hw_reset`` handle False
    gracefully (eager reload would resolve normally; the unload step is
    best-effort).
    """
    provider = _make_provider(tmp_path)
    # Empty mapping → resolve_model_lib raises ValueError for everything.
    _patch_resolve_model_lib(monkeypatch, {})

    assert provider.unload_model("Qwen3-4B") is False


def test_unload_model_resolve_lib_filenotfound_returns_false(
    tmp_path: Path, monkeypatch
):
    """``resolve_model_lib`` may also raise ``FileNotFoundError`` (less
    common but possible if the libs dir is removed). Caught by the same
    except clause; returns False."""
    provider = _make_provider(tmp_path)

    def fake_resolve(model_name: str, libs_dir: Path) -> str:
        raise FileNotFoundError("libs dir vanished")

    monkeypatch.setattr(
        "tether.providers.mlc.provider.resolve_model_lib",
        fake_resolve,
    )

    assert provider.unload_model("Qwen3-4B") is False


# ---------------------------------------------------------------------------
# A2.5 — Cache state preserved on failures.
# ---------------------------------------------------------------------------


def test_unload_model_leaves_cache_intact_on_unknown(tmp_path: Path, monkeypatch):
    """Unloading an unknown model must NOT mutate the cache — other
    engines remain in place. Defensive check in case a future regression
    accidentally pops on resolver failure.
    """
    provider = _make_provider(tmp_path)
    fake_libs = {
        "Loaded-Model": str(tmp_path / "models" / "libs" / "loaded.dll"),
    }
    _patch_resolve_model_lib(monkeypatch, fake_libs)

    key = f"{provider.models_root / 'Loaded-Model'}:{provider.device}:{fake_libs['Loaded-Model']}"
    engine = _FakeEngine(label="loaded")
    with provider._cache_lock:
        provider._engine_cache[key] = engine

    # Try to unload a model that resolves to a different lib path — even
    # if it happened to be a prefix of "Loaded-Model"'s key, we must NOT
    # touch the loaded engine. We verify by passing a name whose canonical
    # key differs from the only loaded key.
    fake_libs["Loaded"] = str(tmp_path / "models" / "libs" / "different.dll")
    _patch_resolve_model_lib(monkeypatch, fake_libs)
    result = provider.unload_model("Loaded")

    assert result is False  # not in cache (different canonical key)
    assert engine.terminated is False
    assert key in provider._engine_cache  # loaded engine untouched
