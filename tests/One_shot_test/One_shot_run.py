#!/usr/bin/env python3
#
# Usage examples:
# python .\tests\One_shot_test\One_shot_run.py `
#        --model ".\dist\Qwen3-4B-q4f16_0-MLC\" `
#        --model-lib ".\dist\libs\Qwen3-4B-q4f16_0-adreno.dll" `
#        --device "opencl:0" `
#        --tools .\tests\One_shot_test\tools_example.json `
#        --system "You are a helpful assistant that uses tools when appropriate." `
#        --user "What's the current EUR->USD rate? Use fx_rate tool if needed." `
#        --stream `
#
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# file: tolerant_tool_parser.py
import ast
import json
import regex as re
from mlc_llm.protocol import openai_api_protocol
from typing import Sequence, Dict, Union, Optional, Any, List, Tuple
from simple_sliding_tool_parser import SimpleSlidingToolStreamParser, StreamEvent
# ...(no change, imports already include needed modules)

# Heuristics
# best-effort nested JSON extraction using the 'regex' module (not 're').
# This pattern may fail for deeply nested or malformed JSON objects.
# Ensure 'regex' is installed and imported as 'import regex as re'.
_JSON_OBJECT_RE = re.compile(r"\{(?:[^{}]|(?R))*\}", re.DOTALL)
_FN_BLOCK_RE = re.compile(r"<<function_calls?>>\s*(" + r"\{(?:[^{}]|(?R))*\}" + r")", re.DOTALL | re.IGNORECASE)
think_re = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def _clean_text(s: str) -> str:
    s = think_re.sub("", s)
    s = re.sub(r"<<response>>", "", s, flags=re.IGNORECASE)
    return s.strip()

def _try_parse_openai_toolcalls_json(obj: Any) -> Optional[List[Dict[str, Any]]]:
    # {"tool_calls":[{"type":"function","function":{"name":"..","arguments":{..}}}, ...]}
    if not isinstance(obj, dict):
        return None
    tcs = obj.get("tool_calls")
    if not isinstance(tcs, list):
        return None
    out = []
    for tc in tcs:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                continue
        if isinstance(name, str) and isinstance(args, dict):
            out.append({"name": name, "arguments": args})
    return out or None

def _try_parse_simple_json_call(obj: Any) -> Optional[List[Dict[str, Any]]]:
    # {"name":"fx_rate","parameters":{"pair":"EURUSD"}}
    # or {"function":{"name":"fx_rate","arguments":{...}}}
    if not isinstance(obj, dict):
        return None
    if "name" in obj and isinstance(obj.get("parameters"), dict):
        return [{"name": obj["name"], "arguments": obj["parameters"]}]
    fn = obj.get("function")
    if isinstance(fn, dict) and "name" in fn:
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return None
        if isinstance(args, dict):
            return [{"name": fn["name"], "arguments": args}]
    return None

def _parse_python_call(call_str: str) -> Optional[Dict[str, Any]]:
    try:
        node = ast.parse(call_str, mode="eval")
        call_node = node.body
        if isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Name):
            name = call_node.func.id
            arguments = {}
            for kw in call_node.keywords:
                arguments[kw.arg] = ast.literal_eval(kw.value)
            return {"name": name, "arguments": arguments}
    except Exception:
        return None
    return None

def convert_function_str_to_json_v2(stringified_calls: str) -> Sequence[Optional[Dict[str, Any]]]:
    """
    Accepts:
      - OpenAI JSON tool_calls
      - JSON after <<function_call[s]>> marker
      - First JSON object in the text
      - Python call: fx_rate(pair="EURUSD")
      - List of python calls: ["fx_rate(pair=\"EURUSD\")", ...]
    Returns: list of {"name": str, "arguments": dict} or [None] if nothing parsed.
    """
    if not stringified_calls:
        return [None]

    s = _clean_text(stringified_calls)

    # 1) Entire text is JSON?
    try:
        obj = json.loads(s)
        parsed = _try_parse_openai_toolcalls_json(obj) or _try_parse_simple_json_call(obj)
        if parsed:
            return parsed
    except Exception:
        pass

    # 2) JSON after <<function_call>> or <<function_calls>>
    m = _FN_BLOCK_RE.search(s)
    if m:
        json_str = m.group(1).strip()
        try:
            obj = json.loads(json_str)
            parsed = _try_parse_openai_toolcalls_json(obj) or _try_parse_simple_json_call(obj)
            if parsed:
                return parsed
        except Exception:
            pass

    # 3) First JSON object in text
    m2 = _JSON_OBJECT_RE.search(s)
    if m2:
        try:
            obj = json.loads(m2.group(0))
            parsed = _try_parse_openai_toolcalls_json(obj) or _try_parse_simple_json_call(obj)
            if parsed:
                return parsed
        except Exception:
            pass

    # 4) Python list of calls: ["fx_rate(pair=\"EURUSD\")", ...]
    s2 = s.strip()
    if s2.startswith("[") and s2.endswith("]"):
        try:
            maybe_list = ast.literal_eval(s2)
            if isinstance(maybe_list, list):
                out: List[Optional[Dict[str, Any]]] = []
                for item in maybe_list:
                    out.append(_parse_python_call(item) if isinstance(item, str) else None)
                return out
        except Exception:
            pass

    # 5) Single python call
    py = _parse_python_call(s2)
    if py:
        return [py]

    return [None]

TOOL_NAMES = {"fx_rate", "weather_api"}  # populate dynamically if you like
_ATTEMPT_PATTERNS = [
    re.compile(r"<<\s*function_call[s]?\s*>>", re.IGNORECASE),
    re.compile(r"\"tool_calls\"\s*:", re.IGNORECASE),
    # Only flag python-ish calls if they start with a known tool name
    re.compile(r"\b(" + "|".join(map(re.escape, TOOL_NAMES)) + r")\s*\(", re.MULTILINE) if TOOL_NAMES else None,
]
_ATTEMPT_PATTERNS = [p for p in _ATTEMPT_PATTERNS if p is not None]

def _looks_like_attempt(text: str) -> bool:
    s = text or ""
    for pat in _ATTEMPT_PATTERNS:
        if pat.search(s):
            return True
    # Also try cheap JSON object detection and check for 'function'/'arguments' keys
    s_trim = s.strip()
    if s_trim.startswith("{") and s_trim.endswith("}"):
        try:
            obj = json.loads(s_trim)
            if isinstance(obj, dict) and (
                "tool_calls" in obj
                or "function" in obj
                or ("name" in obj and ("parameters" in obj or "arguments" in obj))
            ):
                return True
        except Exception:
            pass
    return False

def patched_process_function_call_output(
    output_texts: List[str], finish_reasons: List[str]
) -> Tuple[bool, List[List[openai_api_protocol.ChatToolCall]]]:
    """
    Improved logic:
    - Only error if output *attempts* a tool call but is malformed.
    - Allow normal text responses even when tool mode was enabled upstream.
    - Set finish_reason='tool_calls' *only* when actual calls are parsed.
    """
    n = len(output_texts)
    tool_calls_list: List[List[openai_api_protocol.ChatToolCall]] = [[] for _ in range(n)]
    any_real_calls = False

    for i, text in enumerate(output_texts):
        # 1) Parse with tolerant parser
        try:
            parsed_list = convert_function_str_to_json_v2(text)  # -> list of dicts or [None]
        except Exception:
            parsed_list = [None]

        valid = [p for p in parsed_list if isinstance(p, dict) and "name" in p and "arguments" in p]

        if valid:
            # 2) We have real tool calls -> normalize to tool_calls
            any_real_calls = True
            tool_calls_list[i] = [
                openai_api_protocol.ChatToolCall(
                    type="function",
                    function=openai_api_protocol.ChatFunctionCall(
                        name=p["name"], arguments=p["arguments"]
                    ),
                )
                for p in valid
            ]
            finish_reasons[i] = "tool_calls"
            continue

        # 3) No parsed calls. Decide if this was an attempted call or plain text.
        attempted = _looks_like_attempt(text or "")

        if attempted:
            # Looks like a tool-call attempt but we couldn't parse it -> error
            finish_reasons[i] = "error"
        else:
            # Plain text completion. If upstream pre-set "tool_calls", downgrade to normal stop.
            if finish_reasons[i] == "tool_calls":
                finish_reasons[i] = "stop"
            # otherwise leave whatever non-tool finish_reason is already there (e.g., "stop"/"length")

    return any_real_calls, tool_calls_list

# --- --- --- --- --- --- --- --- --- --- ---
import time

class DebouncedWriter:
    def __init__(self, write_fn, interval=0.03):
        self.write_fn = write_fn
        self.buffer = []
        self.last = 0
        self.interval = interval

    def write(self, text: str, immediate=False):
        self.buffer.append(text)
        now = time.perf_counter()
        if immediate or (now - self.last >= self.interval):
            self.flush()
            self.last = now

    def flush(self):
        if self.buffer:
            self.write_fn("".join(self.buffer))
            self.buffer.clear()

# usage
writer = DebouncedWriter(lambda s: print(s, end="", flush=True))
def ui_write(text: str): writer.write(text)

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
import argparse
import json
import sys
from typing import Any, Dict, List, Optional

try:
    from mlc_llm import MLCEngine
except Exception as e:
    print("ERROR: Could not import MLCEngine from mlc_llm. Is the package installed?")
    print(e)
    sys.exit(1)


def load_tools(path: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "tools" in data:
        return data["tools"]
    if isinstance(data, list):
        return data
    raise ValueError("Tools file must be a JSON array of tools or an object with a 'tools' key.")

def build_messages(system_msg: Optional[str], user_msg: str) -> List[Dict[str, str]]:
    msgs = []
    if system_msg:
        msgs.append({"role": "system", "content": system_msg})
    msgs.append({"role": "user", "content": user_msg})
    return msgs

def print_divider(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")

def pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)

def asdict_maybe(model_or_dict: Any) -> Dict[str, Any]:
    """
    Normalize ChatCompletionResponse (Pydantic v2), Pydantic v1 models, or dict into dict.
    """
    # Pydantic v2
    try:
        model_dump = getattr(model_or_dict, "model_dump", None)
        if callable(model_dump):
            result = model_dump()
            if isinstance(result, dict):
                return result
    except Exception:
        pass
    # Pydantic v1
    try:
        to_dict = getattr(model_or_dict, "dict", None)
        if callable(to_dict):
            result = to_dict()
            if isinstance(result, dict):
                return result
    except Exception:
        pass
    # Already dict?
    if isinstance(model_or_dict, dict):
        return model_or_dict
    # Fallback: try JSON then parse
    try:
        return json.loads(str(model_or_dict))
    except Exception:
        # Last resort: wrap a string
        return {"_raw": str(model_or_dict)}

def try_chat_completions(engine: MLCEngine,
                          messages: List[Dict[str, str]],
                          tools: Optional[List[Dict[str, Any]]],
                          temperature: float,
                          stream: bool) -> Dict[str, Any]:
    chat = getattr(engine, "chat", None)
    if chat is None:
        raise RuntimeError("This MLCEngine does not expose 'engine.chat'. Update mlc_llm.")

    completions = getattr(chat, "completions", None)
    if completions is None:
        raise RuntimeError("This MLCEngine does not expose 'engine.chat.completions'.")

    create = getattr(completions, "create", None)
    if create is None:
        raise RuntimeError("This MLCEngine does not expose 'engine.chat.completions.create'.")

    if stream:
        parser = SimpleSlidingToolStreamParser()
        # setup spinner
        import itertools
        spinner = itertools.cycle(['|', '/', '-', '\\'])
        print_divider("Streaming tokens…")
        collected_events = []
        tool_payload = None
        collected_text = []
        is_thinking = False
        for event in create(messages=messages, tools=tools, tool_choice="auto", temperature=temperature, stream=True):
            d = asdict_maybe(event)
            choices = d.get("choices") or []
            if not choices: continue
            delta = choices[0].get("delta") or {}
            finish_reason = delta.get("finish_reason")
            if finish_reason:
                print(f"\nfinish_reason: {finish_reason}")
            chunk = delta.get("content") or ""

            for ev in parser.feed(chunk):
                collected_events.append(ev)
                if ev["type"] == StreamEvent.TEXT:
                    if is_thinking:
                        writer.write("\n", immediate=True)  # Newline after thinking is done
                        is_thinking = False
                    ui_write(ev["text"])
                    collected_text.append(ev["text"])
                elif ev["type"] == StreamEvent.THINK_STREAM:
                    if not is_thinking:
                        is_thinking = True
                        writer.write("[THINKING] ", immediate=True)
                    ui_write(ev["text"])
                elif ev["type"] == StreamEvent.TOOL_STARTED:
                    # start spinner animation
                    print("\t" + next(spinner), end=' ', flush=True)
                    print("\b\b\b", end='', flush=True)
                elif ev["type"] == StreamEvent.TOOL_PROGRESS:
                    # update spinner on progress tick
                    print("\t" + next(spinner), end=' ', flush=True)
                    print("\b\b\b", end='', flush=True)
                elif ev["type"] == StreamEvent.TOOL_COMPLETE:
                    # clear spinner line
                    print("\r", end='', flush=True)
                    # a tool call was detected
                    print("Tool CALL DETECTED!")
                    parsed_calls = convert_function_str_to_json_v2(ev.get("raw", ""))
                    valid_calls = [p for p in parsed_calls if p]
                    if valid_calls:
                        tool_payload = valid_calls
                        print(valid_calls)

        # After the loop, finalize the parser to flush any remaining text
        for ev in parser.finalize():
            collected_events.append(ev)
            if ev["type"] == StreamEvent.TEXT:
                if is_thinking:
                    writer.write("\n", immediate=True)
                    is_thinking = False
                ui_write(ev["text"])
                collected_text.append(ev["text"])

        writer.flush()
        return {
            "streamed": True,
            "final_content": "".join(collected_text) if collected_text else None,
            "tool_calls": tool_payload,
            "events": collected_events,
        }

    # Non-streaming
    resp = create(messages=messages, tool_choice="auto", tools=tools, temperature=temperature, stream=False)
    print_divider("Full response")
    print(type(resp), repr(resp))
    return asdict_maybe(resp)

def summarize_result(result: Dict[str, Any]):
    print_divider("RESULT SUMMARY")

    # Streaming path
    if result.get("streamed"):
        print("Streamed final content:", result.get("final_content"))
        if result.get("tool_calls") is not None:
            print("\n✅ Detected tool_calls (stream):")
            print(pretty(result["tool_calls"]))
        elif result.get("error") is not None:
            print("\n❌ Detected Error :", result["error"])
        else:
            print("\n⚠️ No tool_calls detected in stream.")
        return

    # Non-streaming, OpenAI-like shape
    print("Raw response:")
    print(pretty(result))

    choices = result.get("choices") or []
    if not choices:
        print("\nNo choices in response.")
        return

    c0 = choices[0]
    finish_reason = c0.get("finish_reason")
    msg = c0.get("message") or {}

    print("\nfinish_reason:", finish_reason)
    if msg.get("tool_calls"):
        print("\n✅ tool_calls present:")
        print(pretty(msg["tool_calls"]))
    else:
        print("\n⚠️ No tool_calls found in message.")

    if msg.get("content"):
        print("\nAssistant content:")
        print(msg["content"])


def main():
    ap = argparse.ArgumentParser(description="Probe OpenAI-style tool calling with MLC LLM Python engine.")
    ap.add_argument("--model", required=True, help="Path to compiled model folder (contains mlc-chat-config.json).")
    ap.add_argument("--model-lib", required=True, help="Path to compiled model library (.dll/.so/.dylib).")
    ap.add_argument("--device", default="cpu:0", help='Device string (e.g., "opencl:0", "cuda:0", "cpu:0").')
    ap.add_argument("--system", default=None, help="Optional system message.")
    ap.add_argument("--user", required=True, help="User prompt to send.")
    ap.add_argument("--tools", default=None, help="Path to tools JSON (OpenAI tools schema).")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--stream", action="store_true", help="Stream tokens.")
    args = ap.parse_args()

    tools = load_tools(args.tools)
    messages = build_messages(args.system, args.user)

    print_divider("ENGINE INIT")
    print(f"Model     : {args.model}")
    print(f"Model lib : {args.model_lib}")
    print(f"Device    : {args.device}")
    if tools:
        print(f"Loaded tools ({len(tools)}):")
        print(pretty(tools))
    else:
        print("No tools provided.")

    engine = MLCEngine(model=args.model, model_lib=args.model_lib, device=args.device)

    print_divider("REQUEST")
    print("Messages:")
    print(pretty(messages))

    try:
        result = try_chat_completions(
            engine=engine,
            messages=messages,
            tools=tools,
            temperature=args.temperature,
            stream=args.stream,
        )
    except Exception as e:
        print_divider("ERROR")
        print("Chat completions call failed:")
        print(e)
        sys.exit(2)

    summarize_result(result)

import mlc_llm.serve.engine_base as engine_base

if __name__ == "__main__":
    orig_parser = engine_base.convert_function_str_to_json
    engine_base.convert_function_str_to_json = convert_function_str_to_json_v2
    engine_base.process_function_call_output = patched_process_function_call_output
    main()
