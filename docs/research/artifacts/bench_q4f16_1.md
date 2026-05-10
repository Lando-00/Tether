# Benchmark: Qwen3-4B-q4f16_1 (new)

- Model: `D:\Dev\TetherWorkspace\dist\Qwen3-4B-q4f16_1-MLC`
- Lib:   `D:\Dev\TetherWorkspace\dist\libs\Qwen3-4B-q4f16_1-adreno.dll`
- Mode:  `interactive`
- Warmup: 122.63 s

## Steady-state (warm — second iteration of each prompt)

This is the metric that matters for sustained UX.

| Prompt | Prompt tok | Resp tok | TTFT (s) | Prefill tok/s | Decode tok/s | Total (s) | FC marker |
|--------|-----------:|---------:|---------:|--------------:|-------------:|----------:|:---------:|
| `tiny` | 17 | 16 | 0.51 | 33.3 | 20.0 | 1.31 | — |
| `medium` | 13 | 160 | 0.58 | 22.4 | 17.4 | 9.79 | — |
| `long-context` | 169 | 160 | 0.56 | 301.9 | 16.4 | 10.34 | — |
| `tool-call` | 26 | 84 | 0.58 | 44.8 | 17.5 | 5.38 | ✅ |

## Cold-start (first encounter — TVM/CLML JIT on each new shape)

Reflects the *first* request a fresh engine sees for each prompt-length / output-length combination. Subsequent requests of the same shape are warm.

| Prompt | Prompt tok | Resp tok | TTFT (s) | Prefill tok/s | Decode tok/s | Total (s) | FC marker |
|--------|-----------:|---------:|---------:|--------------:|-------------:|----------:|:---------:|
| `tiny` | 17 | 16 | 32.17 | 0.5 | 21.0 | 32.93 | — |
| `medium` | 13 | 160 | 133.50 | 0.1 | 17.6 | 142.58 | — |
| `long-context` | 169 | 160 | 206.79 | 0.8 | 16.1 | 216.74 | — |
| `tool-call` | 26 | 84 | 42.75 | 0.6 | 17.6 | 47.53 | ✅ |

## Response previews (warm iteration)

### `tiny`
```
<think> Okay, the user wants me to reply with exactly "OK". Let
```

### `medium`
```
<think> Okay, the user is asking for a concise paragraph explaining the difference between TCP and UDP. Let me start by recalling what I know about these protocols. TCP is connection-oriented, right? It ensures reliable delivery of data by 
```

### `long-context`
```
<think> Okay, let's see. The user is asking about Apple's CEO and their services revenue in fiscal 2024 based on the provided notes.  First, I need to find the CEO. The notes say that Tim Cook has been Apple's CEO since 2011. So that's stra
```

### `tool-call`
```
<think> Okay, the user is asking for the current time in Europe/Dublin. I need to use the get_current_time tool. The tool requires the 'timezone' parameter. The correct timezone for Dublin is Europe/Dublin. So I'll call the function with th
```

