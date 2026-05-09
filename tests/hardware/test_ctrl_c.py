"""
Hardware test: Ctrl+C shutdown behavior.
Converted from root test_ctrl_c.py diagnostic script.
Requires a running Snapdragon Adreno device with MLC models present.
Run with: pytest -m hardware tests/hardware/test_ctrl_c.py
"""
import asyncio
import time

import pytest


@pytest.mark.hardware
async def test_ctrl_c_shutdown_completes_in_time():
    """Server starts and shuts down within 6 s when Ctrl+C is sent.

    Preserves the assertion intent of the original test_ctrl_c.py
    __main__ block: elapsed > 6 was the sys.exit(1) condition.
    """
    from tether_service.app.http.api import create_app

    app = create_app()

    start = time.time()
    try:
        async with app.router.lifespan_context(app):
            await asyncio.sleep(2)
            raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass

    elapsed = time.time() - start
    assert elapsed <= 6, (
        f"Shutdown took {elapsed:.2f}s — expected ≤ 6s. "
        "See MODEL_DEPENDENT_SHUTDOWN_FIX.md for root cause."
    )
