# Benchmark: Qwen3-4B-q4f16_1 + all + /no_think

- Model: `D:\Dev\TetherWorkspace\dist\Qwen3-4B-q4f16_1-MLC`
- Lib:   `D:\Dev\TetherWorkspace\dist\libs\Qwen3-4B-q4f16_1-adreno-all.dll`
- Mode:  `interactive`
- Warmup: 120.02 s

## Steady-state (warm — second iteration of each prompt)

This is the metric that matters for sustained UX.

| Prompt | Prompt tok | Resp tok | TTFT (s) | Prefill tok/s | Decode tok/s | Total (s) | FC marker |
|--------|-----------:|---------:|---------:|--------------:|-------------:|----------:|:---------:|
| `tiny` | 21 | 5 | 0.46 | 45.5 | 19.7 | 0.71 | — |
| `medium` | 17 | 149 | 0.68 | 25.1 | 18.0 | 8.97 | — |
| `long-context` | 173 | 29 | 0.55 | 317.4 | 17.3 | 2.22 | — |
| `tool-call` | 30 | 29 | 0.50 | 59.8 | 18.2 | 2.09 | ✅ |

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

