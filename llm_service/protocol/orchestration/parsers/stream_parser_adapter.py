"""
Adapter for integrating SimpleSlidingToolStreamParser as StreamParser.
"""
from typing import Dict, Any, List

from llm_service.protocol.core.interfaces import StreamParser

# Import the experimental parser from tests
from tests.One_shot_test.simple_sliding_tool_parser import SimpleSlidingToolStreamParser

class SimpleSlidingStreamParser(StreamParser):
    """Adapter implementing StreamParser protocol."""
    def __init__(self, max_tool_chars: int = 32768):
        # Wrap the prototype parser
        self._parser = SimpleSlidingToolStreamParser(max_tool_chars)

    def feed(self, chunk: str) -> List[Dict[str, Any]]:
        """Feed chunk of text and receive list of parser events."""
        return self._parser.feed(chunk)

    def finalize(self) -> List[Dict[str, Any]]:
        """Finalize parser and flush remaining events."""
        return self._parser.finalize()