"""
Hardware test: MLCProvider initialises and shuts down cleanly.
Converted from root test_shutdown.py diagnostic script.
Requires the MLC native libraries to be importable on the Snapdragon device.
Run with: pytest -m hardware tests/hardware/test_shutdown.py
"""
import pytest


@pytest.mark.hardware
async def test_mlc_provider_shutdown_no_models_loaded():
    """MLCProvider shuts down cleanly when no models have been loaded.

    Preserves the assertion intent of the original test_shutdown.py:
    both shutdown_all() calls must complete without raising.
    """
    from tether_service.providers.mlc.provider import MLCProvider

    provider = MLCProvider(dist_root="dist", device="auto")
    models = provider.list_models()
    assert isinstance(models, list), "list_models() must return a list"

    # Default shutdown
    provider.shutdown_all()

    # Shutdown with explicit per-engine timeout
    provider2 = MLCProvider(dist_root="dist", device="auto")
    provider2.shutdown_all(per_engine_timeout=1.0)
