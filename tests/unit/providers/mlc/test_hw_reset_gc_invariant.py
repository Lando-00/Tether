"""P0-D regression: hw_reset / unload / shutdown must route through
``daemon_thread_call(gc_disable=True)``.

Tribunal §3 P0-08 (A6-F1, A2-F4, B6-F11). ADR-0003.

The previous inline ``_terminate_bounded`` helper used a daemon thread
but did NOT disable GC. ADR-0003 says GC-disable is load-bearing for
the OpenCL/TVM destructor hang on ``prefill_chunk_size <= 256`` models
(Qwen2.5-7B). ``hw_reset`` calls into the unload path, so the recovery
path could deadlock on exactly the models the invariant exists for.
These tests guard against silent regression.
"""
from __future__ import annotations

import platform
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _ext() -> str:
    return {"Windows": ".dll", "Darwin": ".dylib"}.get(platform.system(), ".so")


def _build_provider_with_one_engine(tmp_path: Path):
    """Create an :class:`MLCProvider` whose cache contains a single fake
    engine keyed under the canonical ``model_dir:device:lib_path`` form
    that ``unload_model`` recomputes from ``model_name``.
    """
    from tether.providers.mlc.provider import MLCProvider

    model_name = "FakeModel"
    model_dir = tmp_path / model_name
    model_dir.mkdir()
    (model_dir / "mlc-chat-config.json").write_text("{}")
    libs = tmp_path / "libs"
    libs.mkdir()
    lib_path = libs / f"{model_name.lower()}{_ext()}"
    lib_path.write_text("")

    provider = MLCProvider(models_root=str(tmp_path))
    canonical_key = f"{model_dir}:{provider.device}:{lib_path}"
    fake_engine = MagicMock(name="AsyncMLCEngine")
    provider._engine_cache[canonical_key] = fake_engine
    return provider, model_name, fake_engine


@pytest.mark.asyncio
async def test_hw_reset_uses_daemon_thread_call_with_gc_disable(tmp_path):
    """hw_reset → unload_model must route through M1 with gc_disable=True."""
    provider, model_name, fake_engine = _build_provider_with_one_engine(tmp_path)

    with patch(
        "tether.providers.mlc.provider.daemon_thread_call"
    ) as mock_dt, patch.object(
        provider, "_ensure_engine", new=AsyncMock()
    ):
        await provider.hw_reset(model_name)

    assert mock_dt.called, "hw_reset did not route through daemon_thread_call"
    call_kwargs = mock_dt.call_args.kwargs
    assert call_kwargs.get("gc_disable") is True, (
        f"daemon_thread_call invoked without gc_disable=True: kwargs={call_kwargs}"
    )
    # The bound method passed in must be the engine's terminate.
    fn_arg = mock_dt.call_args.args[0]
    assert fn_arg == fake_engine.terminate, (
        "daemon_thread_call must wrap engine.terminate"
    )


def test_shutdown_all_uses_daemon_thread_call_with_gc_disable(tmp_path):
    """shutdown_all must also route every terminate() through M1."""
    provider, _, fake_engine = _build_provider_with_one_engine(tmp_path)

    with patch("tether.providers.mlc.provider.daemon_thread_call") as mock_dt:
        provider.shutdown_all()

    assert mock_dt.called, "shutdown_all did not route through daemon_thread_call"
    # Every call must carry gc_disable=True.
    for call in mock_dt.call_args_list:
        assert call.kwargs.get("gc_disable") is True, (
            f"shutdown_all invoked daemon_thread_call without gc_disable=True: "
            f"kwargs={call.kwargs}"
        )


def test_no_inline_terminate_bounded_remains():
    """The duplicate ``_terminate_bounded`` helper must be deleted to prevent
    drift (A6-F14). The provider must route through the single M1 primitive
    in ``tether.runtime.daemon_call``.
    """
    import tether.providers.mlc.provider as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "_terminate_bounded" not in src, (
        "_terminate_bounded helper still present in provider.py; route "
        "through tether.runtime.daemon_call.daemon_thread_call instead "
        "(P0-D / A6-F14)."
    )


def test_provider_imports_daemon_thread_call():
    """Static guarantee that the provider module wires the M1 primitive."""
    import tether.providers.mlc.provider as mod

    assert hasattr(mod, "daemon_thread_call"), (
        "MLCProvider module must import daemon_thread_call from "
        "tether.runtime.daemon_call"
    )
