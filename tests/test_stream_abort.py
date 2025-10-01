import unittest
import asyncio
from types import SimpleNamespace
from llm_service.protocol.api import ModelTokenSource, TokenDelta

class TestStreamAbort(unittest.IsolatedAsyncioTestCase):
    async def test_abort_current_stream(self):
        # Create a synchronous generator that yields infinite chunks
        def gen():
            class Chunk:
                def __init__(self):
                    delta = SimpleNamespace(content="x")
                    choice = SimpleNamespace(delta=delta, finish_reason=None)
                    self.choices = [choice]
            while True:
                yield Chunk()

        source = ModelTokenSource(gen())
        # Start streaming and read first token
        aiter = source.stream()
        first = await aiter.__anext__()
        self.assertIsInstance(first, TokenDelta)
        self.assertEqual(first.token, "x")
        # Abort the stream
        await source.abort_current_stream()
        # Ensure the stream ends
        with self.assertRaises(StopAsyncIteration):
            await aiter.__anext__()

if __name__ == "__main__":
    unittest.main()
