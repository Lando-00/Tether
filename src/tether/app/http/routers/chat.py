
import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from tether.core.logging import logger

# Compile once at module level for performance.
# (?![\w.]) is a negative lookahead: next char must not be a word character
# or dot, so 'version=0' matches but 'version=00', 'version=0a', 'version=0.5'
# don't. Quoted forms ('version="0"', 'version="1.0"') also work: the closing
# '"' is not in [\w.], satisfying the lookahead. Lowercased input expected.
_VERSION_0_RE = re.compile(r'version="?0"?(?![\w.])')
_VERSION_1_0_RE = re.compile(r'version="?1\.0"?(?![\w.])')


def _has_version_1_0(accept_lower: str) -> bool:
    """Detect 'version=1.0' parameter on application/x-ndjson media type.

    Boundary-aware: 'version=1.0' matches; 'version=1.00', 'version=1.01',
    'version=1.0a' do NOT match (trailing char is a word char or dot).

    Quoted values ('version="1.0"') and unquoted ('version=1.0') both match.
    Lowercased input expected.

    R6 anti-overengineering: regex with negative lookahead. Synthesis §3.4;
    §11.3 R18.
    """
    return bool(_VERSION_1_0_RE.search(accept_lower))


def _has_version_0(accept_lower: str) -> bool:
    """Detect 'version=0' parameter on application/x-ndjson media type.

    Boundary-aware: 'version=0' matches; 'version=00', 'version=0a',
    'version=0.5' do NOT match (trailing char is a word char or dot).

    Quoted values ('version="0"') and unquoted ('version=0') both match.
    Lowercased input expected.

    R6 anti-overengineering: regex with negative lookahead. Synthesis §11.3
    R18; §4 Phase 5 step 56 (p5-cutover-c-flip-default).
    """
    return bool(_VERSION_0_RE.search(accept_lower))

router = APIRouter(prefix="/chat", tags=["chat"])

class StreamRequest(BaseModel):
    session_id: str = Field(
        ...,
        description="The unique identifier for the session.",
        pattern=r"^[A-Za-z0-9_-]{1,128}$",
    )
    prompt: str = Field(
        ...,
        description="The user's prompt.",
        min_length=1,
        max_length=32768,
    )
    model_name: str = Field(
        ...,
        description="The name of the model to use for this generation.",
        pattern=r"^[A-Za-z0-9._-]{1,128}$",
    )
    mode: Literal["chat", "research"] = Field(
        default="chat",
        description=(
            "Orchestrator mode. 'chat' uses ChattyAgentOrchestrator; "
            "'research' uses NotebookOrchestrator (currently 501 Not Implemented). "
            "Honored on both Accept: text/event-stream (SSE) and "
            "application/x-ndjson (NDJSON back-compat) responses."
        ),
    )
    reasoning_effort: Optional[str] = Field(
        default=None,
        description=(
            "Per-request reasoning effort hint for models that advertise "
            "``supports_reasoning_effort=True`` in ``GET /models/details`` "
            "(e.g. GitHub Copilot SDK ``gpt-5``). When the chosen model "
            "does not support reasoning effort, or the supplied value is "
            "not in the model's accepted ``reasoning_efforts`` list, the "
            "server responds 422 before any streaming begins. Omit to "
            "use the provider's default."
        ),
        # Conservative pattern: short identifier-like values only. The
        # authoritative whitelist is enforced via ModelDetails below.
        pattern=r"^[A-Za-z0-9._-]{1,32}$",
    )
    provider_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional provider routing key. When omitted, the server uses "
            "providers.default_model_provider. Unknown values return 422; "
            "known-but-unhealthy values return 503."
        ),
        # Same alphabet as model_name; provider ids are config-controlled
        # and never user-typed at runtime, but keep the validator tight.
        pattern=r"^[A-Za-z0-9._-]{1,64}$",
    )


def _validate_reasoning_effort(
    engine,
    model_name: str,
    reasoning_effort: str,
    provider_id: Optional[str] = None,
) -> None:
    """Reject unsupported ``reasoning_effort`` values BEFORE streaming starts.

    Reads the engine's :class:`ModelDetails` list and confirms (a) the
    chosen model exists, (b) the model advertises reasoning support, and
    (c) the requested value is in its ``reasoning_efforts`` whitelist.
    Raises :class:`HTTPException` 422 otherwise.

    When ``provider_id`` is supplied (Phase 2.B / ADR-0021), only
    ``ModelDetails`` rows whose ``provider_id`` matches are considered so
    that the same model name hosted on two providers with different
    reasoning support doesn't cause a false 422.

    Called only when ``reasoning_effort`` is non-``None`` so the legacy
    request shape pays no extra cost.
    """
    try:
        details = engine.list_model_info()
    except Exception as exc:
        # Provider raised during model introspection — surface as 503 so
        # the client knows the server hasn't fully come up rather than
        # silently dropping reasoning_effort to None.
        raise HTTPException(
            status_code=503,
            detail=f"Could not fetch model metadata: {exc}",
        )
    for info in details:
        if info.id != model_name:
            continue
        # When a provider_id scope is active, skip rows from other providers.
        # Allow "_unwrapped_" sentinel through so pre-Phase-2.A providers
        # (which don't know their registry key) still participate in validation.
        if provider_id is not None and info.provider_id not in ("_unwrapped_", provider_id):
            continue
        if not info.supports_reasoning_effort:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Model '{model_name}' does not support reasoning_effort"
                    + (f" on provider '{provider_id}'" if provider_id else "")
                    + "; omit the field or pick a model with "
                    "supports_reasoning_effort=true in /models/details."
                ),
            )
        accepted = info.reasoning_efforts or []
        if reasoning_effort not in accepted:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"reasoning_effort='{reasoning_effort}' not accepted by "
                    f"model '{model_name}'. Accepted values: {accepted}."
                ),
            )
        return
    # Model not in details: don't 422 here — list_models is the source of
    # truth for model existence and the provider may still accept it.
    # The provider's own error path will run with reasoning_effort
    # forwarded.
    return


@router.post("/stream")
async def stream(request: Request, body: StreamRequest):
    """Stream chat events. Content negotiation via Accept header:

        text/event-stream                       -> SSE-framed v2 events (typed WireEvent)
        application/x-ndjson; version=0         -> NDJSON v0 dict events (LEGACY OPT-IN; deprecated)
        application/x-ndjson; version=1.0       -> NDJSON v2 events (explicit; same as default)
        application/x-ndjson                    -> NDJSON v2 events (NEW DEFAULT)
        (absent)                                -> NDJSON v2 events (NEW DEFAULT)

    Response carries X-Tether-Protocol-Version: 1.0 on every path.
    v0 legacy responses additionally carry Warning: 299 per RFC 9110 §5.6.7.

    Mode dispatch: body.mode selects the Orchestrator via the registry
    (settings.orchestrator.registry). Unimplemented modes (is_implemented=False)
    return 501 before any streaming begins. Unknown modes rejected by
    Pydantic Literal with 422. Briefing §2 Seam B item 4; synthesis §3.5.

    Default flipped to v2 NDJSON in p5-cutover-c-flip-default.
    Synthesis §11.3 R18; §4 Phase 5 step 56.
    """
    accept = request.headers.get("accept", "")
    accept_lower = accept.lower()

    # Three-way negotiation, default flipped to v2 NDJSON:
    #   Accept: text/event-stream                     -> SSE v2 (unchanged)
    #   Accept: application/x-ndjson; version=0       -> NDJSON v0 (legacy opt-in + Warning header)
    #   anything else (incl. application/x-ndjson;    -> NDJSON v2 (NEW DEFAULT)
    #       version=1.0 or no Accept header)
    use_sse = "text/event-stream" in accept_lower
    use_ndjson_v0_legacy = (
        not use_sse
        and "application/x-ndjson" in accept_lower
        and _has_version_0(accept_lower)
    )

    logger.info(
        f"/chat/stream called: session_id={body.session_id}, "
        f"model_name={body.model_name}, mode={body.mode}, "
        f"sse={use_sse}, ndjson_v0_legacy={use_ndjson_v0_legacy}, "
        f"reasoning_effort={body.reasoning_effort}, "
        f"provider_id={body.provider_id}"
    )

    engine = request.app.state.gen_svc
    headers = {"X-Tether-Protocol-Version": "1.0"}

    # --- Provider routing (ADR-0021 Phase 2.B) ---
    # Resolve the effective provider_id. Phase 2.A guarantees
    # engine.default_provider_id; on pre-2.A engines (tests) fall back to
    # None so the provider check below is skipped gracefully.
    pid: Optional[str] = body.provider_id or getattr(engine, "default_provider_id", None)

    # Validate provider availability BEFORE streaming begins.
    # engine.providers and engine._provider_start_failures are guaranteed by
    # Phase 2.A. On pre-2.A engines these attrs are absent and the guard
    # below skips the check, preserving all existing test behaviour.
    _providers = getattr(engine, "providers", None)
    if pid is not None and _providers is not None:
        if pid not in _providers:
            _failures = getattr(engine, "_provider_start_failures", {})
            if pid in _failures:
                # Code-review follow-up: the raw failure text in
                # _provider_start_failures may include filesystem paths,
                # tokens, or other internal state from the provider's
                # __init__ exception. Log the full message server-side
                # for forensics, but return a generic detail to clients.
                logger.error(
                    "/chat/stream provider unhealthy: "
                    f"session_id={body.session_id}, provider_id={pid}, "
                    f"failure={_failures[pid]}"
                )
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Provider '{pid}' is currently unavailable. "
                        "Check the server log for details, or query "
                        "/api/v1/readyz for the per-provider health map."
                    ),
                )
            raise HTTPException(
                status_code=422,
                detail=f"Unknown provider_id '{pid}'.",
            )

    # Forwarded into engine.chat / engine.stream when non-None (Phase 2.A
    # adds the kwarg; pre-2.A engines don't accept it so we omit it entirely
    # by using a conditional dict splice below to preserve backward compat).
    _provider_kwarg: dict = {"provider_id": pid} if pid is not None else {}

    # Validate reasoning_effort against the chosen model's metadata BEFORE
    # we begin streaming. Skipped when reasoning_effort is None so the
    # legacy request shape never pays the provider-introspection cost
    # (e.g. MLC filesystem scan).
    if body.reasoning_effort is not None:
        _validate_reasoning_effort(engine, body.model_name, body.reasoning_effort, provider_id=pid)

    # Eagerly resolve the Orchestrator class to return 501 before streaming
    # begins if the mode is a stub (is_implemented=False). Pydantic's Literal
    # already rejects unknown modes with 422, so this branch handles only the
    # known-but-unimplemented case. Briefing §2 Seam B item 4.
    from tether.protocol.orchestration.registry import (
        UnknownOrchestratorMode,
        resolve_orchestrator_class,
    )
    try:
        orchestrator_cls = resolve_orchestrator_class(
            body.mode, getattr(engine, "_orchestrator_registry", {
                "chat": "tether.protocol.orchestration.chatty.ChattyAgentOrchestrator",
                "research": "tether.protocol.orchestration.notebook.NotebookOrchestrator",
            })
        )
    except UnknownOrchestratorMode as exc:
        # Defensive: Pydantic Literal rejects unknowns at validation time,
        # so this branch is reached only when Pydantic is bypassed (library
        # mode). 400 is appropriate here.
        raise HTTPException(status_code=400, detail=str(exc))

    if not orchestrator_cls.is_implemented:
        raise HTTPException(
            status_code=501,
            detail="research mode tracked in docs/research/06_context_strategies.md",
        )

    if use_sse:
        # SSE path: Engine.chat() yields typed WireEvents; transport_sse frames them.
        # Lazy import keeps tether_service importable without triggering the full
        # transport module graph (library-first invariant). Synthesis §3.4.
        from tether.protocol.orchestration.cancel import AsyncEventCancelToken
        from tether.protocol.wire.transport_sse import transport_sse

        async def sse_generator():
            cancel_token = AsyncEventCancelToken()
            try:
                async def cancellable_chat():
                    async for event in engine.chat(
                        session_id=body.session_id,
                        prompt=body.prompt,
                        model_name=body.model_name,
                        mode=body.mode,
                        cancel_token=cancel_token,
                        reasoning_effort=body.reasoning_effort,
                        **_provider_kwarg,
                    ):
                        if await request.is_disconnected():
                            logger.info(
                                f"Client disconnected (SSE): session_id={body.session_id}"
                            )
                            cancel_token.set()
                            break
                        yield event

                async for chunk in transport_sse(cancellable_chat()):
                    yield chunk
            except Exception as e:
                logger.exception(f"Exception in /chat/stream (SSE): {e}")
                error_payload = {
                    "type": "error",
                    "message": f"Streaming error: {str(e)}",
                    "error_type": type(e).__name__,
                }
                yield (
                    f"event: error\ndata: {json.dumps(error_payload)}\n\n"
                ).encode("utf-8")
                # Phase 5 followups F7: synthesize a terminal MessageStop
                # frame so SSE consumers don't block on a missing terminal
                # event after a fatal streaming exception.
                stop_payload = {
                    "type": "message_stop",
                    "stop_reason": "error",
                }
                yield (
                    f"event: message_stop\ndata: {json.dumps(stop_payload)}\n\n"
                ).encode("utf-8")

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers=headers,
        )

    if use_ndjson_v0_legacy:
        # NDJSON v0 — legacy opt-in via Accept: application/x-ndjson; version=0.
        # Carries Warning: 299 per RFC 9110 §5.6.7 (miscellaneous persistent
        # warning; correct code for deprecation notices).
        # v0_compat_serialize stays wired; Phase 8 removes it.
        # Synthesis §11.3 R18; §4 Phase 5 step 56.
        legacy_headers = {
            **headers,
            "Warning": (
                '299 - "Tether NDJSON v0 vocabulary is deprecated; '
                'use Accept: application/x-ndjson; version=1.0 (or omit '
                'the version parameter for the new v2 default)"'
            ),
        }

        async def ndjson_v0_generator():
            cancel_event = asyncio.Event()
            try:
                async for chunk in engine.stream(
                    session_id=body.session_id,
                    prompt=body.prompt,
                    model_name=body.model_name,
                    mode=body.mode,
                    cancel_event=cancel_event,
                    reasoning_effort=body.reasoning_effort,
                    **_provider_kwarg,
                ):
                    if await request.is_disconnected():
                        logger.info(
                            f"Client disconnected (NDJSON v0 legacy): session_id={body.session_id}"
                        )
                        cancel_event.set()
                        break
                    yield chunk
            except Exception as e:
                logger.exception(f"Exception in /chat/stream (NDJSON v0 legacy): {e}")
                error_event = {
                    "type": "error",
                    "session_id": body.session_id,
                    "data": {
                        "message": f"Streaming error: {str(e)}",
                        "error_type": type(e).__name__,
                    },
                    # Phase 5 followups F7: was None, now an ISO timestamp
                    # like every other v0 frame.
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                yield (json.dumps(error_event) + "\n").encode("utf-8")
                # Phase 5 followups F7: synthesize a terminal v0 ``done``
                # event so legacy consumers see a complete stream after
                # a fatal streaming exception.
                done_event = {
                    "type": "done",
                    "session_id": body.session_id,
                    "data": {},
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                yield (json.dumps(done_event) + "\n").encode("utf-8")

        return StreamingResponse(
            ndjson_v0_generator(),
            media_type="application/x-ndjson",
            headers=legacy_headers,
        )

    # Default: NDJSON v2 (NEW DEFAULT after p5-cutover-c-flip-default).
    # Activated by: no Accept header, Accept: application/x-ndjson (any version
    # except 0), or Accept: application/x-ndjson; version=1.0.
    # Lazy imports preserve library-first invariant. Synthesis §3.4; §11.3 R18.
    from tether.protocol.orchestration.cancel import AsyncEventCancelToken
    from tether.protocol.wire.transport_ndjson import transport_ndjson

    async def ndjson_v2_generator():
        cancel_token = AsyncEventCancelToken()
        try:
            async def cancellable_chat():
                async for event in engine.chat(
                    session_id=body.session_id,
                    prompt=body.prompt,
                    model_name=body.model_name,
                    mode=body.mode,
                    cancel_token=cancel_token,
                    reasoning_effort=body.reasoning_effort,
                    **_provider_kwarg,
                ):
                    if await request.is_disconnected():
                        logger.info(
                            f"Client disconnected (NDJSON v2): session_id={body.session_id}"
                        )
                        cancel_token.set()
                        break
                    yield event

            async for chunk in transport_ndjson(cancellable_chat()):
                yield chunk
        except Exception as e:
            logger.exception(f"Exception in /chat/stream (NDJSON v2): {e}")
            # Hand-rolled v2-shaped error frame. Don't construct Pydantic
            # Error object outside orchestrator — fabricating session IDs,
            # turn_ids, seq, ts at the wire boundary is brittle.
            error_payload = {
                "protocol_version": "1.0",
                "session_id": body.session_id,
                "turn_id": "error",
                "seq": 0,
                "ts": datetime.now(timezone.utc).isoformat(),
                "type": "error",
                "message": f"Streaming error: {str(e)}",
                "error_type": type(e).__name__,
                "is_fatal": False,
            }
            yield (json.dumps(error_payload) + "\n").encode("utf-8")
            # Phase 5 followups F7: synthesize a terminal MessageStop
            # frame so v2 consumers don't block on a missing terminal
            # event after a fatal streaming exception. Same hand-rolled
            # shape as the error frame above (no Pydantic round-trip).
            stop_payload = {
                "protocol_version": "1.0",
                "session_id": body.session_id,
                "turn_id": "error",
                "seq": 1,
                "ts": datetime.now(timezone.utc).isoformat(),
                "type": "message_stop",
                "stop_reason": "error",
            }
            yield (json.dumps(stop_payload) + "\n").encode("utf-8")

    return StreamingResponse(
        ndjson_v2_generator(),
        media_type="application/x-ndjson",
        headers=headers,
    )
