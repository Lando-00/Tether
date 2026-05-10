"""Decorated fixture tools used by ``discover()`` walk tests."""
from tether.tools.base import BaseTool
from tether.tools.registration import tool


@tool(name="fixture_alpha")
class _AlphaTool(BaseTool):
    @property
    def schema(self):
        return {}

    async def run(self):
        return {}


@tool(name="fixture_beta")
class _BetaTool(BaseTool):
    @property
    def schema(self):
        return {}

    async def run(self):
        return {}
