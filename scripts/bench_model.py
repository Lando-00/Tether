"""
Benchmark an MLC-LLM model on Adreno OpenCL.

Captures the metrics that matter for Tether's UX:

* **TTFT** (time-to-first-token) -- perceived latency
* **Prefill tok/s** -- how fast the prompt is ingested
* **Decode tok/s** -- sustained generation speed
* **Total response tokens** -- sanity check on stop conditions
* **Tool-call emission** -- does the model produce the
  ``<<function_call>>`` marker the conv_template steers it toward?

Loads the model in ``mode="interactive"`` to match Tether's runtime
defaults, runs a fixed prompt set with deterministic settings, and
emits both a Markdown table to stdout and a JSON blob (with --out-json)
suitable for comparing across runs / models.

Tokens are counted with the model's own tokenizer (loaded from the
model directory), not from streaming chunk counts -- so the numbers
reflect real generated tokens regardless of how the engine batches
deltas in the SSE stream.

Usage::

    python scripts\\bench_model.py ^
        --model-dir .\\models\\Qwen3-4B-q4f16_1-MLC ^
        --model-lib .\\models\\libs\\Qwen3-4B-q4f16_1-adreno.dll ^
        --label Qwen3-4B-q4f16_1 ^
        --out-json bench_q4f16_1.json
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Default prompt set. Each is a (label, system_prompt_or_None, user_prompt, max_tokens).
PROMPTS: list[tuple[str, Optional[str], str, int]] = [
    (
        "tiny",
        "You are a terse assistant. Output ONLY the requested text.",
        "Reply with exactly: OK",
        16,
    ),
    (
        "medium",
        None,  # use conv_template default system message
        "Write a single concise paragraph explaining the difference between TCP and UDP.",
        160,
    ),
    (
        "long-context",
        None,
        # ~700-char synthetic "retrieved context" + question
        "Below is some research notes. Read them and answer the question at the end.\n\n"
        "[notes]\n"
        "- Apple Inc. is headquartered in Cupertino, California.\n"
        "- Apple's CEO since 2011 is Tim Cook.\n"
        "- Apple's fiscal 2024 revenue was approximately $391 billion.\n"
        "- The iPhone accounts for the majority of Apple's revenue.\n"
        "- Apple's services segment includes the App Store, iCloud, and Apple Music.\n"
        "- Apple announced its M-series silicon transition in 2020.\n"
        "- Apple's services revenue exceeded $85 billion in fiscal 2024.\n"
        "- Apple Intelligence was announced at WWDC 2024.\n"
        "[/notes]\n\n"
        "Question: According to the notes, who is Apple's CEO and how much "
        "did Apple earn in services revenue in fiscal 2024?",
        160,
    ),
    (
        "tool-call",
        # Override system to look like Tether's tool-calling steering.
        None,  # use conv_template's tool-calling system message
        "What is the current time in Europe/Dublin? You have a get_current_time "
        "tool that takes a 'timezone' parameter.",
        96,
    ),
]


@dataclass
class PromptResult:
    label: str
    iteration: int  # 1 = cold, 2 = warm
    prompt_tokens: int
    response_tokens: int
    response_chars: int
    ttft_s: float
    total_s: float
    decode_s: float
    decode_toks_per_s: float
    prefill_toks_per_s: float
    response_preview: str
    has_function_call_marker: bool = False
    error: Optional[str] = None


@dataclass
class BenchRun:
    label: str
    model_dir: str
    model_lib: str
    mode: str
    warmup_s: Optional[float] = None
    results: list[PromptResult] = field(default_factory=list)


def _print(msg: str) -> None:
    print(f"==> {msg}", flush=True)


def _load_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)


async def _run_one(engine, tok, label: str, iteration: int, system: Optional[str],
                   user: str, max_tokens: int, *, no_think: bool = False) -> PromptResult:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    user_to_send = user
    if no_think:
        # Qwen3's "no thinking" directive — instructs the model to skip the
        # <think>...</think> block entirely. Recognised at the prompt level by
        # the model regardless of chat-template support.
        user_to_send = f"{user} /no_think"
    messages.append({"role": "user", "content": user_to_send})

    # tokenize prompt once for prefill-rate estimate (rough: doesn't include
    # template-applied special tokens, but close enough for relative comparison)
    prompt_token_count = len(tok.encode(user_to_send))
    if system:
        prompt_token_count += len(tok.encode(system))

    request_id = f"bench-{label}-{uuid.uuid4().hex[:8]}"
    response_chars = 0
    full_response = ""
    first_tok_at: Optional[float] = None
    error: Optional[str] = None

    t0 = time.perf_counter()
    try:
        gen = await engine.chat.completions.create(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            stream=True,
            request_id=request_id,
        )
        async for resp in gen:
            for choice in resp.choices:
                delta = choice.delta.content or ""
                if delta:
                    if first_tok_at is None:
                        first_tok_at = time.perf_counter()
                    response_chars += len(delta)
                    full_response += delta
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    t1 = time.perf_counter()
    if first_tok_at is None:
        first_tok_at = t1  # nothing came back
    response_token_count = len(tok.encode(full_response)) if full_response else 0

    ttft = first_tok_at - t0
    total = t1 - t0
    decode_s = max(t1 - first_tok_at, 1e-9)

    return PromptResult(
        label=label,
        iteration=iteration,
        prompt_tokens=prompt_token_count,
        response_tokens=response_token_count,
        response_chars=response_chars,
        ttft_s=ttft,
        total_s=total,
        decode_s=decode_s,
        decode_toks_per_s=response_token_count / decode_s if response_token_count else 0.0,
        prefill_toks_per_s=prompt_token_count / ttft if ttft > 0 else 0.0,
        response_preview=full_response[:240].replace("\n", " "),
        has_function_call_marker="<<function_call>>" in full_response,
        error=error,
    )


async def _bench_async(args: argparse.Namespace) -> BenchRun:
    from mlc_llm import AsyncMLCEngine

    _print(f"loading tokenizer from {args.model_dir}")
    tok = _load_tokenizer(args.model_dir)

    _print(f"loading {args.model_dir.name} on Adreno (mode={args.mode})")
    engine = AsyncMLCEngine(
        model=str(args.model_dir),
        model_lib=str(args.model_lib),
        device="auto",
        mode=args.mode,
    )

    run = BenchRun(
        label=args.label or args.model_dir.name,
        model_dir=str(args.model_dir),
        model_lib=str(args.model_lib),
        mode=args.mode,
    )

    try:
        # warmup so cache+kernel JIT doesn't taint the first measurement
        _print("warming up (single short prompt) ...")
        wt0 = time.perf_counter()
        await _run_one(engine, tok, "warmup", 0, None, "Hi.", 4)
        run.warmup_s = time.perf_counter() - wt0
        _print(f"  warmup took {run.warmup_s:.2f} s")

        # Each benchmark prompt is run twice. Iteration 1 pays cold-shape
        # JIT/autotuning costs on Adreno+TVM (hundreds of seconds per new
        # sequence-length on first encounter); iteration 2 reflects
        # steady-state behaviour once kernels for that shape are cached.
        iters = (2,) if args.skip_cold else (1, 2)
        for label, system, user, max_toks in PROMPTS:
            # When skipping cold, do an unmeasured prep pass to get this
            # prompt's shape JIT'd before the warm measurement.
            if args.skip_cold:
                _print(f"prepping '{label}' (unmeasured) ...")
                await _run_one(engine, tok, label, 0, system, user, max_toks,
                               no_think=args.no_think)
            for iteration in iters:
                tag = "cold" if iteration == 1 else "warm"
                _print(f"running '{label}' iter {iteration} ({tag}) ...")
                res = await _run_one(engine, tok, label, iteration, system,
                                     user, max_toks, no_think=args.no_think)
                run.results.append(res)
                _print(
                    f"  ttft={res.ttft_s:.2f}s "
                    f"prefill={res.prefill_toks_per_s:.1f} tok/s "
                    f"decode={res.decode_toks_per_s:.1f} tok/s "
                    f"({res.response_tokens} tok in {res.decode_s:.2f}s)"
                )
                if res.error:
                    _print(f"  ERROR: {res.error}")
    finally:
        try:
            engine.terminate()
        except Exception as exc:
            _print(f"engine.terminate() raised: {exc!r}")

    return run


def render_md(run: BenchRun) -> str:
    lines: list[str] = []
    lines.append(f"# Benchmark: {run.label}")
    lines.append("")
    lines.append(f"- Model: `{run.model_dir}`")
    lines.append(f"- Lib:   `{run.model_lib}`")
    lines.append(f"- Mode:  `{run.mode}`")
    if run.warmup_s is not None:
        lines.append(f"- Warmup: {run.warmup_s:.2f} s")
    lines.append("")

    cold = [r for r in run.results if r.iteration == 1]
    warm = [r for r in run.results if r.iteration == 2]

    def _row(r: PromptResult) -> str:
        fc = "✅" if r.has_function_call_marker else "—"
        return (
            f"| `{r.label}` | {r.prompt_tokens} | {r.response_tokens} | "
            f"{r.ttft_s:.2f} | {r.prefill_toks_per_s:.1f} | "
            f"{r.decode_toks_per_s:.1f} | {r.total_s:.2f} | {fc} |"
        )

    if warm:
        lines.append("## Steady-state (warm — second iteration of each prompt)")
        lines.append("")
        lines.append("This is the metric that matters for sustained UX.")
        lines.append("")
        lines.append("| Prompt | Prompt tok | Resp tok | TTFT (s) | Prefill tok/s | Decode tok/s | Total (s) | FC marker |")
        lines.append("|--------|-----------:|---------:|---------:|--------------:|-------------:|----------:|:---------:|")
        for r in warm:
            lines.append(_row(r))
        lines.append("")

    if cold:
        lines.append("## Cold-start (first encounter — TVM/CLML JIT on each new shape)")
        lines.append("")
        lines.append("Reflects the *first* request a fresh engine sees for each prompt-length / output-length combination. Subsequent requests of the same shape are warm.")
        lines.append("")
        lines.append("| Prompt | Prompt tok | Resp tok | TTFT (s) | Prefill tok/s | Decode tok/s | Total (s) | FC marker |")
        lines.append("|--------|-----------:|---------:|---------:|--------------:|-------------:|----------:|:---------:|")
        for r in cold:
            lines.append(_row(r))
        lines.append("")

    lines.append("## Response previews (warm iteration)")
    lines.append("")
    for r in warm or run.results:
        lines.append(f"### `{r.label}`")
        lines.append("```")
        lines.append(r.response_preview)
        lines.append("```")
        if r.error:
            lines.append(f"⚠️ error: {r.error}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--model-lib", type=Path, required=True)
    p.add_argument("--label", default=None,
                   help="Display label (default: model dir name)")
    p.add_argument("--mode", default="interactive",
                   choices=["local", "interactive", "server"])
    p.add_argument("--no-think", action="store_true",
                   help="Append Qwen3's /no_think directive to every prompt to disable <think> blocks")
    p.add_argument("--skip-cold", action="store_true",
                   help="Skip cold-iteration measurement and only report warm steady-state (faster: ~3 min vs ~10 min)")
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-md", type=Path, default=None)
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    if not args.model_dir.exists():
        print(f"error: model dir not found: {args.model_dir}", file=sys.stderr)
        return 2
    if not args.model_lib.exists():
        print(f"error: model lib not found: {args.model_lib}", file=sys.stderr)
        return 2

    run = asyncio.run(_bench_async(args))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(dataclasses.asdict(run), indent=2),
            encoding="utf-8",
        )
        _print(f"wrote {args.out_json}")

    md = render_md(run)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
        _print(f"wrote {args.out_md}")
    else:
        sys.stdout.write("\n" + md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
