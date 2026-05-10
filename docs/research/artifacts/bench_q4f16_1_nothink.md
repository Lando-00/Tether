# Benchmark: Qwen3-4B-q4f16_1 + /no_think

- Model: `D:\Dev\TetherWorkspace\dist\Qwen3-4B-q4f16_1-MLC`
- Lib:   `D:\Dev\TetherWorkspace\dist\libs\Qwen3-4B-q4f16_1-adreno.dll`
- Mode:  `interactive`
- Warmup: 122.64 s

## Steady-state (warm — second iteration of each prompt)

This is the metric that matters for sustained UX.

| Prompt | Prompt tok | Resp tok | TTFT (s) | Prefill tok/s | Decode tok/s | Total (s) | FC marker |
|--------|-----------:|---------:|---------:|--------------:|-------------:|----------:|:---------:|
| `tiny` | 21 | 5 | 0.54 | 38.6 | 16.1 | 0.85 | — |
| `medium` | 17 | 149 | 0.52 | 33.0 | 17.9 | 8.83 | — |
| `long-context` | 173 | 29 | 0.55 | 312.6 | 17.3 | 2.23 | — |
| `tool-call` | 30 | 29 | 0.48 | 62.2 | 17.9 | 2.10 | ✅ |

## Response previews (warm iteration)

### `tiny`
```
<think>  </think>  OK
```

### `medium`
```
<think>  </think>  TCP (Transmission Control Protocol) and UDP (User Datagram Protocol) are both internet protocols used for transmitting data over a network, but they differ in their approach and use cases. TCP is a connection-oriented pro
```

### `long-context`
```
<think>  </think>  Apple's CEO is Tim Cook, and Apple earned $85 billion in services revenue in fiscal 2024.
```

### `tool-call`
```
<think>  </think>  <<function_call>>   {"name": "get_current_time", "params": {"timezone": "Europe/Dublin"}}
```

