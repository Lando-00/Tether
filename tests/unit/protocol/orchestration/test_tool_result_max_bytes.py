"""ToolRunner.run rejects oversized tool results.

Phase 7 step 77. Default cap is 256 KB; configurable via
Settings.security.tool_result_max_bytes.
"""
import pytest

from tether.protocol.orchestration.tool_runner import ToolRunner

pytestmark = pytest.mark.anyio


class _OversizedTool:
    """Test tool that returns a result way over the cap."""

    schema = {"name": "oversized", "parameters": {"type": "object"}}

    async def invoke(self, args, context=None):
        # 300 KB of repeated 'x' — will JSON-serialize to ~300 KB
        return {"data": "x" * (300 * 1024)}


class _SmallTool:
    """Test tool that returns a tiny result."""

    schema = {"name": "small", "parameters": {"type": "object"}}

    async def invoke(self, args, context=None):
        return {"data": "ok"}


async def test_runner_accepts_within_cap():
    """Result smaller than cap returns verbatim."""
    runner = ToolRunner({"small": _SmallTool()}, result_max_bytes=256 * 1024)
    result = await runner.run("small", {})
    assert result == {"data": "ok"}


async def test_runner_rejects_oversize():
    """Result larger than cap is replaced with a structured error."""
    runner = ToolRunner({"oversized": _OversizedTool()}, result_max_bytes=10 * 1024)
    result = await runner.run("oversized", {})

    assert isinstance(result, dict)
    assert result.get("error") == "tool_result_oversized"
    assert result.get("tool_name") == "oversized"
    assert result.get("size_bytes", 0) > 10 * 1024
    assert result.get("limit_bytes") == 10 * 1024


async def test_default_cap_256kb():
    """Default cap (no kwarg) is 256 KB."""
    runner = ToolRunner({"small": _SmallTool()})
    assert runner.result_max_bytes == 256 * 1024


async def test_runner_handles_non_serializable_result():
    """Non-JSON-serializable result falls back to repr length; does not raise."""

    class _OddTool:
        schema = {"name": "odd", "parameters": {"type": "object"}}

        async def invoke(self, args, context=None):
            class _NonSerializable:
                pass

            return _NonSerializable()

    runner = ToolRunner({"odd": _OddTool()}, result_max_bytes=256 * 1024)
    # default=str in _measure_result_size handles most types; if it still
    # can't serialise, repr() is used. Either way, should not raise.
    result = await runner.run("odd", {})
    # _NonSerializable repr is tiny; passes the cap and returns verbatim.
    assert result is not None


async def test_size_threshold_boundary():
    """Result exactly at the cap is accepted; result over the cap is rejected."""

    class _AtCapTool:
        schema = {"name": "at_cap", "parameters": {"type": "object"}}

        async def invoke(self, args, context=None):
            # json.dumps({"data": ""}) == '{"data": ""}' == 12 chars overhead.
            # Fill to exactly 5000 total.
            return {"data": "x" * (5000 - 12)}

    class _OverCapTool:
        schema = {"name": "over_cap", "parameters": {"type": "object"}}

        async def invoke(self, args, context=None):
            return {"data": "x" * (5000 - 12 + 1)}  # 1 byte over

    runner = ToolRunner(
        {"at_cap": _AtCapTool(), "over_cap": _OverCapTool()},
        result_max_bytes=5000,
    )

    r1 = await runner.run("at_cap", {})
    assert "error" not in r1

    r2 = await runner.run("over_cap", {})
    assert r2.get("error") == "tool_result_oversized"


async def test_runner_size_check_doesnt_block_timeout_path():
    """Size cap does not interfere with the existing timeout behavior."""

    class _SlowTool:
        schema = {"name": "slow", "parameters": {"type": "object"}}

        async def invoke(self, args, context=None):
            import asyncio

            await asyncio.sleep(2.0)
            return {"data": "ok"}

    runner = ToolRunner({"slow": _SlowTool()}, timeout_sec=1, result_max_bytes=256 * 1024)

    with pytest.raises(Exception):  # asyncio.TimeoutError
        await runner.run("slow", {})
