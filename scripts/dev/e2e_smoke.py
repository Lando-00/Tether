"""E2E baseline smoke for tether-cli's HTTP layer.

Mimics what `src/tether/cli/main.py` does at the wire level: create session,
POST /chat/stream with v2 NDJSON Accept header, consume + print events.
Used to validate the backend works end-to-end without driving the
interactive Typer UI.
"""
import json
import sys

import requests

BASE = "http://127.0.0.1:8080/api/v1"

r = requests.post(BASE + "/sessions")
r.raise_for_status()
data = r.json()
sid = data.get("session_id") or data.get("id")
print(f"[OK] session created: {sid}")

prompt = "What time is it in Dublin?"
print(f"[USER] {prompt}")

events = []
text_buf = []
tool_calls = []
tool_results = []

with requests.post(
    BASE + "/chat/stream",
    json={"session_id": sid, "prompt": prompt, "model_name": "Qwen3-4B-q4f16_1-MLC"},
    headers={"Accept": "application/x-ndjson; version=1.0"},
    stream=True,
    timeout=300,
) as resp:
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        events.append(ev)
        t = ev.get("type")
        if t == "message_start":
            tool_names = [tool.get("name") for tool in ev.get("tools", [])]
            print(f"[message_start] turn_id={ev.get('turn_id')} tools={tool_names}")
        elif t == "text_delta":
            text_buf.append(ev.get("text", ""))
        elif t == "tool_call":
            tool_calls.append((ev.get("name"), ev.get("arguments")))
            print(f"[tool_call] {ev.get('name')}({ev.get('arguments')})")
        elif t == "tool_result":
            tool_results.append(
                (ev.get("name"), ev.get("status"), ev.get("result"))
            )
            preview = str(ev.get("result", ""))[:80]
            print(f"[tool_result] {ev.get('name')} status={ev.get('status')} preview={preview}")
        elif t == "message_stop":
            print(f"[message_stop] reason={ev.get('reason')}")

print("---SUMMARY---")
print(f"total events: {len(events)}")
text_full = "".join(text_buf)
print(f"text length: {len(text_full)} chars")
print(f"tool calls: {len(tool_calls)}; tool results: {len(tool_results)}")
print("---FULL TEXT---")
print(text_full[:500])
