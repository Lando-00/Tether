"""
Hardware test: model-specific shutdown behavior.
Converted from root test_model_shutdown.py diagnostic script.
Requires actual Qwen model directories in dist/ on a Snapdragon device.
Run with: pytest -m hardware tests/hardware/test_model_shutdown.py

Original usage: python test_model_shutdown.py <model_name>
Parametrized here over the two models cited in copilot-instructions.md.
"""
import asyncio
import time
from pathlib import Path

import pytest

_MODELS = [
    "Qwen3-4B-q4f16_0-MLC",
    "Qwen2.5-7B-q4f16_0-MLC",
]


@pytest.mark.hardware
@pytest.mark.parametrize("model_name", _MODELS)
async def test_model_shutdown_completes(model_name: str):
    """Load a model and verify shutdown_all() does not hang.

    Preserves the assertion intent of the original test_model_shutdown.py:
    return False (→ sys.exit(1)) when the model is not found or shutdown
    raises. Skips when the model directory is absent (dev machines without
    the Snapdragon Adreno device or downloaded weights).
    """
    from tether_service.providers.mlc.provider import MLCProvider

    dist_root = Path(__file__).parent.parent.parent / "dist"
    model_dir = dist_root / model_name
    if not model_dir.exists():
        pytest.skip(f"Model directory not found: {model_dir}")

    provider = MLCProvider(dist_root=str(dist_root), device="auto", max_tokens=100)
    available = provider.list_models()
    if model_name not in available:
        pytest.skip(f"Model {model_name} not in provider.list_models(): {available}")

    # Load model by streaming a short prompt
    messages = [{"role": "user", "content": "Hi"}]
    chunks = []
    try:
        async for chunk in provider.stream(model_name, messages):
            if isinstance(chunk, str):
                chunks.append(chunk)
            if len(chunks) >= 5:
                break
    except GeneratorExit:
        pass

    # Verify shutdown completes without hanging
    start = time.time()
    provider.shutdown_all(per_engine_timeout=0.75)
    elapsed = time.time() - start

    assert elapsed < 10, (
        f"{model_name} shutdown took {elapsed:.2f}s — "
        "expected < 10s. Check GC-disable fix in api.py."
    )
