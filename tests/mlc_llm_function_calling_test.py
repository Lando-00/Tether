from mlc_llm import MLCEngine

MODEL_DIR = r"D:\Dev\Adreano\dist\Qwen2.5-7B-q4f16_0-MLC"
MODEL_LIB = r"D:\Dev\Adreano\dist\libs\Qwen2.5-7B-q4f16-adreno.dll"

engine = MLCEngine(model=MODEL_DIR, model_lib=MODEL_LIB, device="opencl")  # or "windows:adreno_x86"
conv = engine.conv_template
print(conv)

tools = [{
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Return the current time at the given timezone.",
        "parameters": {
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
            "required": ["timezone"],
            "additionalProperties": False
        }
    }
}]

resp = engine.chat.completions.create(
    model=MODEL_DIR,
    messages=[{"role":"user","content":"What time is it in Europe/Dublin?"}],
    tools=tools,
    # tool_choice={"type":"function","function":{"name":"get_current_time"}},
    temperature=0.0,
    stream=False,
)

print("Response:", resp)
c = resp.choices[0]
print("finish_reason:", c.finish_reason)
try:
    print("RAW assistant message:", resp.choices[0].message)
except Exception as e:
    print("Could not print raw message:", e)

engine.terminate()
