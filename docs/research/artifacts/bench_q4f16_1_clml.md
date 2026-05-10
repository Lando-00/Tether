# Benchmark: Qwen3-4B-q4f16_1 + CLML + /no_think

- Model: `D:\Dev\TetherWorkspace\dist\Qwen3-4B-q4f16_1-MLC`
- Lib:   `D:\Dev\TetherWorkspace\dist\libs\Qwen3-4B-q4f16_1-adreno-clml.dll`
- Mode:  `interactive`
- Warmup: 121.08 s

## Steady-state (warm — second iteration of each prompt)

This is the metric that matters for sustained UX.

| Prompt | Prompt tok | Resp tok | TTFT (s) | Prefill tok/s | Decode tok/s | Total (s) | FC marker |
|--------|-----------:|---------:|---------:|--------------:|-------------:|----------:|:---------:|
| `tiny` | 21 | 5 | 0.62 | 33.8 | 18.3 | 0.89 | — |
| `medium` | 17 | 149 | 0.56 | 30.4 | 18.5 | 8.60 | — |
| `long-context` | 173 | 29 | 0.58 | 296.3 | 16.3 | 2.36 | — |
| `tool-call` | 30 | 29 | 0.59 | 50.9 | 18.5 | 2.15 | ✅ |

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

