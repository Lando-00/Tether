# mlc_stream_patch.py
import time, json
from collections import defaultdict
from typing import Optional, List

# Import the target and protocol
from mlc_llm.serve import engine_base as eb
from mlc_llm.protocol import openai_api_protocol as proto

# Per-request, per-choice text buffers: buffers[request_id][choice_idx] -> str
_buffers = defaultdict(lambda: defaultdict(str))

# Keep the original for fallback
_original = eb.process_chat_completion_stream_output

import re

# Only parse calls starting with "__tool_"
TOOL_PREFIX_STR = "__tool_"
CALL_LINE = re.compile(rf'^{re.escape(TOOL_PREFIX_STR)}[A-Za-z_]\w*\([^)]*\)\s*$')

def _first_line(s: str) -> str:
    for ln in (s or "").splitlines():
        ln = ln.strip()
        if ln:
            return ln
    return ""

def _extract_single_call(text: str) -> Optional[str]:
    """
    Return a single-line tool call like '__tool_name(arg=val)' if the FIRST
    non-empty line is exactly a prefixed call. Otherwise return None.
    """
    line = _first_line(text)
    return line if line and CALL_LINE.match(line) else None

def _build_tool_calls_from_text(call_text: str) -> Optional[List[dict]]:
    # Uses MLC's own converter; expects keyword args (name=..., not dict positional)
    try:
        fn_json_list = eb.convert_function_str_to_json(call_text)
    except Exception:
        return None
    calls = []
    for j, fn in enumerate(fn_json_list or []):
        if not fn:
            continue
        name = fn.get("name")
        args = fn.get("arguments", {})
        if not name or not isinstance(args, dict):
            continue
        calls.append({
            "id": f"call_{int(time.time())}_{j}",
            "type": "function",
            "function": {"name": name, "arguments": args},  # <-- dict, not JSON string
        })
    return calls or None


def patched_process_chat_completion_stream_output(
    delta_outputs,
    request: proto.ChatCompletionRequest,
    request_id: str,
    engine_state: eb.EngineState,
    use_function_calling: bool,
    finish_reasons: list,
):
    # Handle the special "final usage" chunk; clear buffers and fully delegate
    is_final_chunk = delta_outputs[0].request_final_usage_json_str is not None
    if is_final_chunk:
        _buffers.pop(request_id, None)
        return _original(delta_outputs, request, request_id, engine_state,
                         use_function_calling, finish_reasons)

    # Normal streaming chunk (mirrors original, but with buffering + final synth)
    assert len(delta_outputs) == request.n
    choices = []

    for i, d in enumerate(delta_outputs):
        # 1) Accumulate any text emitted for this choice
        if getattr(d, "delta_text", ""):
            _buffers[request_id][i] += d.delta_text

        # 2) Compute finish_reason as upstream does
        finish_reason_updated = False
        if d.finish_reason is not None and finish_reasons[i] is None:
            finish_reasons[i] = (d.finish_reason if not use_function_calling else "tool_calls")
            finish_reason_updated = True

        # 3) If no new text and not finishing, skip (as upstream)
        if not finish_reason_updated and getattr(d, "delta_text", "") == "":
            engine_state.record_event(request_id, event="skip empty delta text")
            continue

        # 4) Prepare delta message kwargs (default: pass through text)
        delta_kwargs = {"content": getattr(d, "delta_text", ""), "role": "assistant"}

        # 5) On the *final* chunk of this choice, if function-calling is on,
        #    parse the full buffered text and inject tool_calls if it looks like a call.
        if use_function_calling and finish_reason_updated:
            # We won't trust the auto-flipped "tool_calls".
            # Try to extract & parse a single call from the first line only.
            candidate = _extract_single_call(_buffers[request_id][i])
            tool_calls = _build_tool_calls_from_text(candidate) if candidate else None
            if tool_calls:
                # Do not leak raw call text on the final chunk; emit structured tool_calls instead
                delta_kwargs["content"] = ""
                delta_kwargs["tool_calls"] = tool_calls
                # Clear buffer for this choice
                _buffers[request_id].pop(i, None)
                finish_reasons[i] = "tool_calls"
            else:
                # Not a call: DO NOT synthesize, and restore the original finish reason
                finish_reasons[i] = d.finish_reason or "stop"

        # 6) Build the choice (carry logprobs if present, as upstream)
        choice = proto.ChatCompletionStreamResponseChoice(
            index=i,
            finish_reason=finish_reasons[i],
            delta=proto.ChatCompletionMessage(**delta_kwargs),
            logprobs=(
                proto.LogProbs(content=[
                    proto.LogProbsContent.model_validate_json(j)
                    for j in (d.delta_logprob_json_strs or [])
                ])
                if getattr(d, "delta_logprob_json_strs", None) is not None else None
            ),
        )
        choices.append(choice)

    if not choices:
        return None

    response = proto.ChatCompletionStreamResponse(
        id=request_id, choices=choices, model=request.model, system_fingerprint=""
    )
    engine_state.record_event(request_id, event="yield delta output (patched)")
    return response

def apply():
    eb.process_chat_completion_stream_output = patched_process_chat_completion_stream_output