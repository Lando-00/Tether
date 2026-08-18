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

import json
from collections import deque
from typing import Any, AsyncGenerator, Deque, Dict, List, Literal, Optional, Union

from tether.core.interfaces import ModelProvider
from tether.providers.types import ProviderCapabilities

# Type alias: a single canned response.
# - dict      → serialised to JSON before yielding (normal happy-path)
# - str       → yielded verbatim (allows raw / malformed JSON for failure tests)
# - list[str] → streamed as N separate chunks (synthesizer streaming tests)
# Wave 1 deliberately locks the simpler Union-in-deque model; stream() will
# use isinstance(item, list) rather than a private _ChunkedResponse sentinel.
CannedResponse = Union[Dict[str, Any], str, List[str]]


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
        self._synthesizer_responses: Deque[CannedResponse] = deque()
        self._direct_responses: Deque[CannedResponse] = deque()
        self._unknown_responses: Deque[CannedResponse] = deque()
        self._chunk_size = chunk_size
        self.chunk_size = chunk_size

        # Injected exceptions (raised next time the phase is invoked).
        # Single-slot model: raise_on_*() fails the next call to that
        # phase; subsequent calls consume the response queue normally.
        # To pattern "succeed once, fail once", tests should use a future
        # unified queue that can hold both responses and exceptions.
        self._planner_exc: Optional[BaseException] = None
        self._extractor_exc: Optional[BaseException] = None
        self._synthesizer_exc: Optional[BaseException] = None
        self._direct_exc: Optional[BaseException] = None

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

    def set_direct_response(self, response: Union[str, List[str]]) -> None:
        """Queue ONE direct-answer response.

        The direct phase is used by
        :class:`~tether.protocol.orchestration.notebook.NotebookOrchestrator`
        when turn triage decides a turn needs no external evidence and the
        orchestrator answers inline. It never plans or searches.
        """
        self._direct_responses.clear()
        self._direct_responses.append(response)

    def set_chat_response(self, response: Union[str, List[str]]) -> None:
        """Queue ONE response for the *chat* path (phase ``unknown``).

        :class:`~tether.protocol.orchestration.notebook.AutoOrchestrator`
        delegates DIRECT turns to
        :class:`~tether.protocol.orchestration.chatty.ChattyAgentOrchestrator`,
        whose prompts carry the application system prompt rather than any
        notebook phase marker — so they land on ``unknown``.
        """
        self._unknown_responses.clear()
        self._unknown_responses.append(response)

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

        1. Detect phase from system prompt in messages.
        2. Raise and clear the phase's injected exception, if set.
        3. Pop next canned response from that phase's queue.
        4. Serialise dict → compact JSON / yield str verbatim / yield each
           element of a list[str] as a separate chunk.
        5. Empty queues yield nothing so parser failure paths can be tested.
        """
        del model_name, tools, request_id

        phase = self._detect_phase(messages)
        self.call_log.append((phase, messages))

        exc_attr = f"_{phase}_exc"
        exc = getattr(self, exc_attr, None)
        if exc is not None:
            setattr(self, exc_attr, None)
            raise exc

        queue_attr = f"_{phase}_responses"
        queue: Deque[CannedResponse] = getattr(self, queue_attr, deque())
        if not queue:
            return

        item = queue.popleft()

        if isinstance(item, dict):
            text = json.dumps(item, separators=(",", ":"))
        elif isinstance(item, list):
            for chunk in item:
                yield chunk
            return
        else:
            text = item

        if self._chunk_size > 0:
            for i in range(0, len(text), self._chunk_size):
                yield text[i : i + self._chunk_size]
        else:
            yield text

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

    def _detect_phase(
        self, messages: List[Dict[str, Any]]
    ) -> Literal["planner", "extractor", "synthesizer", "direct", "unknown"]:
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
        system_msgs = [m for m in messages if m.get("role") == "system"]
        if not system_msgs:
            return "unknown"

        last_system = (system_msgs[-1].get("content") or "").lower()
        if "synthesize" in last_system or "synthesizer" in last_system:
            return "synthesizer"
        # The direct-answer phase (AutoOrchestrator triage → DIRECT) has no
        # tools and no notebook; it is recognised by its explicit no-tools rule.
        if "you have no tools here" in last_system:
            return "direct"
        if "extract" in last_system and "fact" in last_system:
            return "extractor"
        if (
            "planner" in last_system
            or "key_elements" in last_system
            or "key element" in last_system
        ):
            return "planner"
        return "unknown"


__all__ = ["FakeResearchProvider"]
