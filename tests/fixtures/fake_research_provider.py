"""FakeResearchProvider — deterministic test fixture (Phase 9 · rs-D-FAKE).

In-memory :class:`tether.core.interfaces.ModelProvider` that returns
canned phase-aware responses for NotebookOrchestrator tests. Phase
detection inspects the last ``system``-role message in the
``messages`` list (content-based dispatch — rs-D-FAKE.md §2.3).

Canned-response API mirrors rs-D-FAKE.md §3.2:

* :meth:`set_planner_response` — queue ONE planner JSON.
* :meth:`set_extractor_responses` — queue N extractor JSONs (FIFO).
* :meth:`set_synthesizer_response` — queue ONE synth string (or list
  of chunks for streamed delivery).
* :meth:`raise_on_planner` / :meth:`raise_on_extractor` /
  :meth:`raise_on_synthesizer` — inject an exception that fires
  inside the async generator on the next call to that phase.

NOT a production provider. Lives under ``tests/fixtures/`` so it never
ships in the wheel. The orchestrator never sees it; only test code
constructs it.

Wave 1 deliverable: SKELETON. Setters store data; ABC methods return
sensible defaults; :meth:`stream` and :meth:`_detect_phase` raise
``NotImplementedError`` with the Wave-2 owner tag. Wave 2 IMP-D fills
in the dispatch logic per rs-D-FAKE §2.3 / §3.
"""

from __future__ import annotations

from collections import deque
from typing import Any, AsyncGenerator, Deque, Dict, List, Optional, Union

from tether.core.interfaces import ModelProvider
from tether.providers.types import ProviderCapabilities

# Type alias: a single canned response.
# - dict      → serialised to JSON before yielding (normal happy-path)
# - str       → yielded verbatim (allows raw / malformed JSON for failure tests)
# - list[str] → streamed as N separate chunks (synthesizer streaming tests)
# Wave 1 deliberately locks the simpler Union-in-deque model; stream() will
# use isinstance(item, list) rather than a private _ChunkedResponse sentinel.
CannedResponse = Union[Dict[str, Any], str]


class FakeResearchProvider(ModelProvider):
    """Phase-aware deterministic provider for NotebookOrchestrator tests.

    Usage::

        fake = FakeResearchProvider()
        fake.set_planner_response({"key_elements": ["q1", "q2"]})
        fake.set_extractor_responses([
            {"facts": [{"text": "f1", "confidence": "high"}],
             "follow_up_queries": []},
        ])
        fake.set_synthesizer_response("The answer is X [1].")

        orch = NotebookOrchestrator(provider=fake, ...)
        events = [e async for e in orch.run(...)]
    """

    def __init__(self, *, chunk_size: int = 0) -> None:
        # Per-phase canned response deques (FIFO).
        self._planner_responses: Deque[CannedResponse] = deque()
        self._extractor_responses: Deque[CannedResponse] = deque()
        self._synthesizer_responses: Deque[Union[str, List[str]]] = deque()
        self.chunk_size = chunk_size

        # Injected exceptions (raised next time the phase is invoked).
        # Single-slot model: raise_on_*() fails the next call to that
        # phase; subsequent calls consume the response queue normally.
        # To pattern "succeed once, fail once", tests should use a future
        # unified queue that can hold both responses and exceptions.
        self._planner_exc: Optional[BaseException] = None
        self._extractor_exc: Optional[BaseException] = None
        self._synthesizer_exc: Optional[BaseException] = None

        # Call log: every (phase, messages) pair, in order. Tests
        # assert on call ordering / argument values without mocking.
        self.call_log: List[tuple[str, List[Dict[str, Any]]]] = []

    # ------------------------------------------------------------------
    # Canned-response API
    # ------------------------------------------------------------------

    def set_planner_response(self, response: CannedResponse) -> None:
        """Queue ONE planner response (replaces any pending)."""
        self._planner_responses.clear()
        self._planner_responses.append(response)

    def set_extractor_responses(self, responses: List[CannedResponse]) -> None:
        """Queue N extractor responses (replaces any pending). FIFO order."""
        self._extractor_responses.clear()
        self._extractor_responses.extend(responses)

    def set_synthesizer_response(self, response: Union[str, List[str]]) -> None:
        """Queue ONE synthesizer response.

        Pass a ``str`` to yield as one chunk. Pass a ``list[str]`` to
        simulate chunked streamed delivery (each list item becomes one
        chunk yielded from :meth:`stream`).
        """
        self._synthesizer_responses.clear()
        self._synthesizer_responses.append(response)

    def raise_on_planner(self, exc: BaseException) -> None:
        """Inject an exception to be raised on the next planner call."""
        self._planner_exc = exc

    def raise_on_extractor(self, exc: BaseException) -> None:
        """Inject an exception to be raised on the next extractor call."""
        self._extractor_exc = exc

    def raise_on_synthesizer(self, exc: BaseException) -> None:
        """Inject an exception to be raised on the next synthesizer call."""
        self._synthesizer_exc = exc

    # ------------------------------------------------------------------
    # ModelProvider ABC — abstract methods
    # ------------------------------------------------------------------

    async def stream(  # type: ignore[override]
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Phase-aware canned-response dispatch.

        Wave 2 IMP-D will:
            1. Detect phase from system prompt in messages.
            2. Pop next canned response from that phase's queue (or
               raise the injected exception if set).
            3. Serialise dict → JSON / yield str verbatim / yield each
               element of a list[str] as a separate chunk.
            4. Empty queue raises ``RuntimeError`` (fail loud, not
               silent — rs-D-FAKE §3.3).
        """
        raise NotImplementedError(
            "Wave 2 IMP-D will implement phase detection + canned dispatch"
        )
        # Unreachable, but makes the function an async generator so
        # callers can ``async for`` over the result without TypeError.
        if False:  # pragma: no cover
            yield ""  # type: ignore[unreachable]

    def list_models(self) -> List[str]:
        """Single fixed model name for test reproducibility."""
        return ["fake-research-model"]

    def unload_model(self, model_name: str) -> bool:
        """No-op; the fake holds no real model state."""
        return True

    def get_context_window(self, model_name: str) -> int:
        """Matches Qwen3-4B practical ctx (docs/research/06_context_strategies.md)."""
        return 32768

    # ------------------------------------------------------------------
    # ModelProvider v2 overrides (non-abstract)
    # ------------------------------------------------------------------

    @property
    def kind(self) -> str:
        return "fake_research"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(streaming=True)

    # warm_up / aclose → inherit ABC no-ops.
    # stream_typed     → inherit ABC NotImplementedError default;
    #                    NotebookOrchestrator uses the legacy stream() path.

    # ------------------------------------------------------------------
    # Internal helpers (Wave 2 IMP-D will fill)
    # ------------------------------------------------------------------

    def _detect_phase(self, messages: List[Dict[str, Any]]) -> str:
        """Return ``'planner' | 'extractor' | 'synthesizer' | 'unknown'``.

        Wave 2 IMP-D walks ``messages`` in reverse, stops at the first
        ``role == 'system'`` entry, and matches on lowercase content
        substrings:

        * ``'planner'`` or ``'key element'`` → ``'planner'``
        * ``'extract'`` AND ``'fact'``       → ``'extractor'``
        * ``'synthesize'`` or ``'synthesizer'`` → ``'synthesizer'``
        * otherwise                          → ``'unknown'``

        Markers chosen to match the prompts in
        :mod:`tether.protocol.orchestration.notebook_prompts`.
        """
        raise NotImplementedError("Wave 2 IMP-D")


__all__ = ["FakeResearchProvider"]
