import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from tether.core.errors import (
    AmbiguousModelError,
    ProviderUnhealthyError,
    UnknownModelError,
    UnknownProviderError,
)
from tether.core.logging import logger
from tether.core.provider_ids import PROVIDER_ID_PATTERN

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
        pattern=(
            r"^[A-Za-z0-9][A-Za-z0-9._-]*"
            r"(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?"
            r"(?::[A-Za-z0-9][A-Za-z0-9._-]*)?$"
        ),
        max_length=256,
    )
    mode: Optional[Literal["auto", "chat", "research"]] = Field(
        default=None,
        description=(
            "Orchestrator mode. Omit to use the server's configured default "
            "(`orchestrator.default`, which ships as 'auto'). 'auto' uses "
            "AutoOrchestrator: fact-based orchestration that triages each turn "
            "and answers directly when no external evidence is needed. 'chat' "
            "uses ChattyAgentOrchestrator (legacy tool loop). 'research' forces "
            "the full NotebookOrchestrator Plan→Search→Extract→Refine→"
            "Synthesize loop (ADR-0020) and is never downgraded by triage; it "
            "requires 'web_search' in tools.enabled. Any mode must be "
            "registered in orchestrator.registry. Honored on both "
            "Accept: text/event-stream (SSE) and application/x-ndjson "
            "(NDJSON back-compat) responses."
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
        pattern=r"^[A-Za-z0-9._-]{1,32}$",
    )
    provider_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional explicit provider routing key. When omitted, the "
            "server routes only if exactly one healthy provider advertises "
            "the model. Unknown or ambiguous models return 422; a "
            "known-but-unhealthy provider returns 503."
        ),
        pattern=PROVIDER_ID_PATTERN,
    )


def _validate_reasoning_effort(
    engine,
    model_name: str,
    reasoning_effort: str,
    provider_id: Optional[str] = None,
    *,
    details=None,
) -> None:
    """Reject unsupported ``reasoning_effort`` values BEFORE streaming starts.

    When ``provider_id`` is supplied (ADR-0021), only ``ModelDetails`` rows
    whose ``provider_id`` matches are considered.
    """
    if details is None:
        try:
            details = engine.list_model_info()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Could not fetch model metadata: {exc}",
            ) from exc

    matching = [
        info
        for info in details
        if info.id == model_name
        and (
            provider_id is None
            or info.provider_id in ("_unwrapped_", provider_id)
        )
    ]
    if len(matching) != 1:
        suffix = f" on provider '{provider_id}'" if provider_id else ""
        raise HTTPException(
            status_code=503,
            detail=(
                f"Could not fetch model metadata for '{model_name}'{suffix}. "
                "Retry when the provider is healthy."
            ),
        )

    info = matching[0]
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


def _reasoning_models_for_request(
    engine,
    *,
    mode: str,
    model_name: str,
) -> tuple[str, ...]:
    """Return every model that receives a request-scoped reasoning effort."""
    if mode != "research":
        return (model_name,)

    research_settings = getattr(engine, "_research_settings", None)
    phase_models = (
        getattr(research_settings, "planner_model", None) or model_name,
        getattr(research_settings, "extractor_model", None) or model_name,
        getattr(research_settings, "synthesizer_model", None) or model_name,
    )
    return tuple(dict.fromkeys(phase_models))


def _resolve_provider_id(
    engine,
    *,
    model_name: str,
    requested_provider_id: Optional[str],
    mode: str,
) -> Optional[str]:
    """Resolve a request before streaming so route errors keep HTTP status."""
    resolver = getattr(engine, "resolve_provider_id", None)
    if callable(resolver):
        try:
            provider_id = resolver(
                model_name,
                provider_id=requested_provider_id,
            )
            if mode == "research":
                validate_overrides = getattr(
                    engine,
                    "validate_research_model_overrides",
                    None,
                )
                if callable(validate_overrides):
                    validate_overrides(provider_id)
            return provider_id
        except ProviderUnhealthyError as exc:
            logger.error(
                "/chat/stream provider unhealthy: provider_id=%s",
                exc.provider_id,
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Provider '{exc.provider_id}' is currently unavailable. "
                    "Check the server log for details, or query "
                    "/api/v1/readyz for the per-provider health map."
                ),
            ) from exc
        except UnknownProviderError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown provider_id '{exc.provider_id}'.",
            ) from exc
        except UnknownModelError as exc:
            suffix = f" on provider '{exc.provider_id}'" if exc.provider_id is not None else ""
            raise HTTPException(
                status_code=422,
                detail=f"Model '{exc.model_name}' is not available{suffix}.",
            ) from exc
        except AmbiguousModelError as exc:
            raise HTTPException(
                status_code=422,
                detail=(f"Model '{exc.model_name}' is available from multiple providers; specify provider_id."),
            ) from exc

    # Compatibility for older Engine implementations that predate the
    # provider/model resolver. New Engines always take the branch above.
    provider_id = requested_provider_id or getattr(engine, "default_provider_id", None)
    providers = getattr(engine, "providers", None)
    if provider_id is not None and providers is not None and provider_id not in providers:
        failures = getattr(engine, "_provider_start_failures", {})
        if provider_id in failures:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Provider '{provider_id}' is currently unavailable. "
                    "Check the server log for details, or query "
                    "/api/v1/readyz for the per-provider health map."
                ),
            )
        raise HTTPException(
            status_code=422,
            detail=f"Unknown provider_id '{provider_id}'.",
        )
    return provider_id


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
    use_ndjson_v0_legacy = not use_sse and "application/x-ndjson" in accept_lower and _has_version_0(accept_lower)

    engine = request.app.state.gen_svc
    headers = {"X-Tether-Protocol-Version": "1.0"}

    # ``mode`` is optional on the wire: omitting it means "use the server's
    # configured orchestrator.default". Resolve it once here so every downstream
    # helper (provider resolution, reasoning-effort validation, orchestrator
    # lookup, and the engine call itself) agrees on the same effective mode.
    effective_mode: str = body.mode or str(
        getattr(engine, "_orchestrator_default_mode", "chat") or "chat"
    )

    logger.info(
        f"/chat/stream called: session_id={body.session_id}, "
        f"model_name={body.model_name}, mode={effective_mode}"
        f"{'' if body.mode else ' (default)'}, "
        f"sse={use_sse}, ndjson_v0_legacy={use_ndjson_v0_legacy}, "
        f"reasoning_effort={body.reasoning_effort}, "
        f"provider_id={body.provider_id}"
    )

    pid = _resolve_provider_id(
        engine,
        model_name=body.model_name,
        requested_provider_id=body.provider_id,
        mode=effective_mode,
    )
    _provider_kwarg: dict = {"provider_id": pid} if pid is not None else {}
    if (
        pid is not None
        and callable(getattr(engine, "resolve_provider_id", None))
    ):
        _provider_kwarg["_resolved_provider_id"] = pid

    # Validate reasoning_effort against chosen model's metadata BEFORE streaming.
    if body.reasoning_effort is not None:
        try:
            details = engine.list_model_info()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Could not fetch model metadata: {exc}",
            ) from exc
        for reasoning_model in _reasoning_models_for_request(
            engine,
            mode=effective_mode,
            model_name=body.model_name,
        ):
            _validate_reasoning_effort(
                engine,
                reasoning_model,
                body.reasoning_effort,
                provider_id=pid,
                details=details,
            )

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
            effective_mode,
            getattr(
                engine,
                "_orchestrator_registry",
                {
                    "chat": "tether.protocol.orchestration.chatty.ChattyAgentOrchestrator",
                    "research": "tether.protocol.orchestration.notebook.NotebookOrchestrator",
                },
            ),
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
                        mode=effective_mode,
                        cancel_token=cancel_token,
                        reasoning_effort=body.reasoning_effort,
                        **_provider_kwarg,
                    ):
                        if await request.is_disconnected():
                            logger.info(f"Client disconnected (SSE): session_id={body.session_id}")
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
                yield (f"event: error\ndata: {json.dumps(error_payload)}\n\n").encode("utf-8")
                # Phase 5 followups F7: synthesize a terminal MessageStop
                # frame so SSE consumers don't block on a missing terminal
                # event after a fatal streaming exception.
                stop_payload = {
                    "type": "message_stop",
                    "stop_reason": "error",
                }
                yield (f"event: message_stop\ndata: {json.dumps(stop_payload)}\n\n").encode("utf-8")

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
                "use Accept: application/x-ndjson; version=1.0 (or omit "
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
                    mode=effective_mode,
                    cancel_event=cancel_event,
                    reasoning_effort=body.reasoning_effort,
                    **_provider_kwarg,
                ):
                    if await request.is_disconnected():
                        logger.info(f"Client disconnected (NDJSON v0 legacy): session_id={body.session_id}")
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
                    mode=effective_mode,
                    cancel_token=cancel_token,
                    reasoning_effort=body.reasoning_effort,
                    **_provider_kwarg,
                ):
                    if await request.is_disconnected():
                        logger.info(f"Client disconnected (NDJSON v2): session_id={body.session_id}")
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
