import re
from typing import Callable, Optional

from llm_service.protocol.core.types import RoutedChunk


class HiddenBlockFilter:
    """
    Strips tokens inside <think>...</think> (and similar) from the stream.
    Can optionally surface them to UI as 'hidden_thought' events.
    """
    START_TAGS = [re.compile(r'<think\b[^>]*>'), re.compile(r'<reflection\b[^>]*>')]
    END_TAGS   = [re.compile(r'</think\s*>'),     re.compile(r'</reflection\s*>')]
    START_HTML_COMMENT = re.compile(r'<!--')
    END_HTML_COMMENT   = re.compile(r'-->')

    def __init__(self, expose_hidden_thoughts: bool = True, on_hidden_chunk: Optional[Callable[[str], None]] = None):
        """Filter that routes hidden blocks/comments away from detector and optionally invokes callback."""
        # expose_hidden_thoughts controls whether hidden tokens are emitted to UI
        self.expose_hidden = expose_hidden_thoughts
        self._on_hidden = on_hidden_chunk
        self._in_hidden = False
        self._in_comment = False
        self._buf = []

    def feed(self, token: str) -> RoutedChunk:
        # Detect HTML comments irrespective of tag state
        if not self._in_hidden and self.START_HTML_COMMENT.search(token):
            self._in_comment = True
        if self._in_comment:
            # Accumulate comments as hidden tokens
            self._buf.append(token)
            if self.END_HTML_COMMENT.search(token):
                self._in_comment = False
                self._buf.clear()
            hidden = token if self.expose_hidden else ""
            if self._on_hidden:
                self._on_hidden(token)
            return RoutedChunk(to_detector="", to_ui_visible="", to_ui_hidden=hidden)
        # Detect think/reflection start tags
        if not self._in_hidden and any(p.search(token) for p in self.START_TAGS):
            self._in_hidden = True
        if self._in_hidden:
            # Accumulate hidden block tokens
            self._buf.append(token)
            if any(p.search(token) for p in self.END_TAGS):
                self._in_hidden = False
                self._buf.clear()
            hidden = token if self.expose_hidden else ""
            if self._on_hidden:
                self._on_hidden(token)
            return RoutedChunk(to_detector="", to_ui_visible="", to_ui_hidden=hidden)
        # Visible token routes to detector and visible UI
        return RoutedChunk(to_detector=token, to_ui_visible=token, to_ui_hidden="")