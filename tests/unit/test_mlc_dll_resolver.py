"""Bug regression: DLL resolver must distinguish between quant variants.

When both ``Qwen3-4B-q4f16_0-adreno.dll`` and ``Qwen3-4B-q4f16_1-adreno.dll``
exist in ``libs/``, ``resolve_model_lib("Qwen3-4B-q4f16_1-MLC", libs_dir)``
must pick the q4f16_1 DLL — not the alphabetically-first q4f16_0 one.

The pre-fix resolver stripped the quant from the match key
(``Qwen3-4B-q4f16_1-MLC`` → ``Qwen3-4B``) so any DLL whose name contained
``qwen3-4b`` matched and ``matches[0]`` returned the wrong one. Loading
that DLL caused a TVMError at runtime because q4f16_0 and q4f16_1 have
incompatible weight shapes.

Discovered while running ``scripts/dev/library_smoke.py`` after copying
the model set into ``models/``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tether.providers.mlc.provider import (
    base_key_from_model_name,
    match_model_dlls,
    resolve_model_lib,
)

# ---------------------------------------------------------------------------
# Match-key precision
# ---------------------------------------------------------------------------


def test_base_key_preserves_quantization() -> None:
    """Two models that share a family but differ in quant must produce
    different match keys, otherwise the resolver can't tell them apart."""
    key_0 = base_key_from_model_name("Qwen3-4B-q4f16_0-MLC").lower()
    key_1 = base_key_from_model_name("Qwen3-4B-q4f16_1-MLC").lower()
    assert key_0 != key_1, (
        f"base_key collapsed quant variants: q4f16_0 -> {key_0!r}, "
        f"q4f16_1 -> {key_1!r}"
    )


def test_base_key_strips_only_mlc_suffix() -> None:
    """The trailing ``-MLC`` marker is stripped; the quant suffix is kept."""
    assert base_key_from_model_name("Qwen3-4B-q4f16_1-MLC").lower() == "qwen3-4b-q4f16_1"
    assert base_key_from_model_name("Qwen2.5-7B-q4f16_0-MLC").lower() == "qwen2.5-7b-q4f16_0"


# ---------------------------------------------------------------------------
# Resolver behaviour
# ---------------------------------------------------------------------------


def test_resolve_model_lib_picks_matching_quant(tmp_path: Path) -> None:
    """Both q4f16_0 and q4f16_1 DLLs are present; resolver must pick the
    one matching the requested model's quant."""
    libs_dir = tmp_path / "libs"
    libs_dir.mkdir()
    dll_q0 = libs_dir / "Qwen3-4B-q4f16_0-adreno.dll"
    dll_q1 = libs_dir / "Qwen3-4B-q4f16_1-adreno.dll"
    dll_q0.touch()
    dll_q1.touch()

    assert resolve_model_lib("Qwen3-4B-q4f16_1-MLC", libs_dir) == str(dll_q1)
    assert resolve_model_lib("Qwen3-4B-q4f16_0-MLC", libs_dir) == str(dll_q0)


def test_match_model_dlls_returns_only_quant_matches(tmp_path: Path) -> None:
    """``match_model_dlls`` must NOT return DLLs of sibling quant variants."""
    libs_dir = tmp_path / "libs"
    libs_dir.mkdir()
    dll_q0 = libs_dir / "Qwen3-4B-q4f16_0-adreno.dll"
    dll_q1 = libs_dir / "Qwen3-4B-q4f16_1-adreno.dll"
    dll_q0.touch()
    dll_q1.touch()

    matches_q1 = match_model_dlls("Qwen3-4B-q4f16_1-MLC", [dll_q0, dll_q1])
    assert matches_q1 == [dll_q1], f"expected only q4f16_1, got {matches_q1}"

    matches_q0 = match_model_dlls("Qwen3-4B-q4f16_0-MLC", [dll_q0, dll_q1])
    assert matches_q0 == [dll_q0], f"expected only q4f16_0, got {matches_q0}"


def test_resolve_model_lib_handles_qwen25_mlc_suffix_in_dll_name(tmp_path: Path) -> None:
    """Real-world: ``Qwen2.5-7B-q4f16_0-MLC-adreno.dll`` has ``MLC`` inside
    the DLL name. The resolver must still match it for model
    ``Qwen2.5-7B-q4f16_0-MLC``."""
    libs_dir = tmp_path / "libs"
    libs_dir.mkdir()
    dll = libs_dir / "Qwen2.5-7B-q4f16_0-MLC-adreno.dll"
    dll.touch()
    assert resolve_model_lib("Qwen2.5-7B-q4f16_0-MLC", libs_dir) == str(dll)


def test_resolve_model_lib_falls_back_to_family_match(tmp_path: Path) -> None:
    """If no precise DLL exists but a loose family match does (e.g.
    operator compiled a single DLL named ``mymodel.dll`` for two quants),
    return that loose match as a last resort rather than erroring out."""
    libs_dir = tmp_path / "libs"
    libs_dir.mkdir()
    dll = libs_dir / "Qwen3-4B-adreno.dll"  # no quant in name
    dll.touch()

    # Precise match key would be "qwen3-4b-q4f16_1", which isn't in the DLL.
    # The loose fallback ("qwen3-4b") should still find it.
    assert resolve_model_lib("Qwen3-4B-q4f16_1-MLC", libs_dir) == str(dll)


def test_resolve_model_lib_raises_when_no_match(tmp_path: Path) -> None:
    libs_dir = tmp_path / "libs"
    libs_dir.mkdir()
    (libs_dir / "Mistral-7B-q4f16_1-adreno.dll").touch()

    with pytest.raises(ValueError, match="No matching model library"):
        resolve_model_lib("Qwen3-4B-q4f16_1-MLC", libs_dir)
