"""Parametrized SessionStore fixtures for contract tests.

Both Memory and Sqlite stores are exercised through the same test
bodies. The contract is: identical output from get_history() for
identical input sequences. Phase 6 schema migration (yoyo) MUST
preserve these contracts.

Synthesis §11.3 R19; Phase 5 -> Phase 6 gate.

Note: async tests/fixtures are handled by pytest-asyncio (asyncio_mode = "auto"
in pyproject.toml). No @pytest.mark.anyio needed.
"""
import pytest

from tether.context.memory_store import MemoryStore
from tether.context.sqlite_store import SqliteSessionStore


@pytest.fixture
async def memory_store():
    yield MemoryStore()


@pytest.fixture
async def sqlite_store(tmp_path):
    """Fresh SqliteSessionStore per test, isolated by tmp_path."""
    db_path = tmp_path / "contract_test.db"
    dsn = f"sqlite:///{db_path}"
    store = SqliteSessionStore(dsn=dsn)
    await store.connect()
    try:
        yield store
    finally:
        await store.aclose()


@pytest.fixture(params=["memory", "sqlite"])
async def store(request, memory_store, sqlite_store):
    """Parametrized: yields a Memory or Sqlite store per test."""
    if request.param == "memory":
        yield memory_store
    else:
        yield sqlite_store
