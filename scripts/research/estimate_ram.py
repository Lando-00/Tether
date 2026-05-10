"""
Estimate the runtime memory footprint of an MLC-LLM model on Snapdragon
X Elite (or any other unified-memory device) so we can pre-screen
candidates against a hard RAM budget before downloading anything.

Approximates three contributors:

1. **Weights** -- params (B) * bits/8 (per quantization label).
2. **KV cache** -- 2 * num_layers * num_kv_heads * head_dim * ctx * 2 bytes.
3. **Activations / scratch** -- flat ~256 MB working-set reserve.

Examples::

    python estimate_ram.py --model-config models\\Qwen3-4B-q4f16_0-MLC\\mlc-chat-config.json
    python estimate_ram.py --params-b 4 --quant q4f16_1 --num-layers 36 --num-kv-heads 8 --head-dim 128 --sweep-ctx 4096,16384,32768,40960
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

QUANT_BITS_PER_WEIGHT: dict[str, float] = {
    "q0f16": 16.0,
    "q0f32": 32.0,
    "q3f16_0": 3.5,
    "q3f16_1": 3.5,
    "q4f16_0": 4.5,
    "q4f16_1": 4.5,
    "q4f16_2": 4.5,
    "q4f32_1": 4.5,
    "q8f16_0": 8.5,
    "q8f16_1": 8.5,
    "q8f32_1": 8.5,
}

KV_BYTES_PER_ELEM = 2  # fp16 KV cache
ACTIVATION_RESERVE_BYTES = 256 * 1024 * 1024


@dataclass
class Estimate:
    label: str
    weights_gb: float
    kv_cache_gb: float
    activation_gb: float

    @property
    def total_gb(self) -> float:
        return self.weights_gb + self.kv_cache_gb + self.activation_gb


def estimate(*, params_b: float, quant: str, num_layers: int, num_kv_heads: int,
             head_dim: int, ctx: int, label: str = "") -> Estimate:
    bits = QUANT_BITS_PER_WEIGHT.get(quant)
    if bits is None:
        raise ValueError(f"unknown quant {quant!r}")
    weight_bytes = params_b * 1e9 * bits / 8.0
    kv_bytes = 2 * num_layers * num_kv_heads * head_dim * ctx * KV_BYTES_PER_ELEM
    return Estimate(
        label=label,
        weights_gb=weight_bytes / (1024 ** 3),
        kv_cache_gb=kv_bytes / (1024 ** 3),
        activation_gb=ACTIVATION_RESERVE_BYTES / (1024 ** 3),
    )


def estimate_from_config(cfg_path: Path, *, params_b: Optional[float] = None,
                         ctx_override: Optional[int] = None,
                         quant_override: Optional[str] = None) -> Estimate:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    quant = quant_override or cfg.get("quantization")
    ctx = ctx_override or cfg.get("context_window_size") or 4096
    mc = cfg.get("model_config") or {}
    if "text_config" in mc:
        mc = mc["text_config"]

    num_layers = mc.get("num_hidden_layers")
    num_kv_heads = mc.get("num_key_value_heads") or mc.get("num_attention_heads")
    head_dim = mc.get("head_dim")
    if head_dim is None and mc.get("hidden_size") and mc.get("num_attention_heads"):
        head_dim = mc["hidden_size"] // mc["num_attention_heads"]

    if params_b is None:
        shard_total = sum(
            shard.stat().st_size
            for shard in cfg_path.parent.glob("params_shard_*.bin")
        )
        if shard_total > 0:
            bits = QUANT_BITS_PER_WEIGHT.get(quant)
            if bits:
                params_b = (shard_total * 8 / bits) / 1e9
    if params_b is None:
        raise ValueError(
            f"could not determine params_b for {cfg_path}; pass --params-b"
        )

    missing = [k for k, v in dict(
        num_layers=num_layers, num_kv_heads=num_kv_heads,
        head_dim=head_dim, quant=quant,
    ).items() if v is None]
    if missing:
        raise ValueError(f"config {cfg_path} missing fields: {missing}")

    return estimate(
        params_b=params_b, quant=quant,
        num_layers=num_layers, num_kv_heads=num_kv_heads,
        head_dim=head_dim, ctx=ctx,
        label=str(cfg_path.parent.name),
    )


def render(est: Estimate, budget_gb: float) -> str:
    fits = "✅ fits" if est.total_gb <= budget_gb else "❌ over budget"
    return (
        f"{est.label or 'model':<55s} "
        f"weights {est.weights_gb:>5.2f} GB  "
        f"KV {est.kv_cache_gb:>5.2f} GB  "
        f"act {est.activation_gb:>4.2f} GB  "
        f"= {est.total_gb:>5.2f} GB / {budget_gb:.0f} GB  {fits}"
    )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--model-config", type=Path, default=None)
    p.add_argument("--params-b", type=float, default=None)
    p.add_argument("--quant", type=str, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--num-kv-heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=None)
    p.add_argument("--ctx", type=int, default=None)
    p.add_argument("--sweep-ctx", type=str, default=None)
    p.add_argument("--budget-gb", type=float, default=9.0)
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    if args.sweep_ctx:
        contexts = [int(x) for x in args.sweep_ctx.split(",")]
    elif args.ctx:
        contexts = [args.ctx]
    elif args.model_config:
        cfg = json.loads(args.model_config.read_text(encoding="utf-8"))
        contexts = [cfg.get("context_window_size") or 4096]
    else:
        p.error("specify --ctx, --sweep-ctx, or --model-config")
        return 2

    for ctx in contexts:
        try:
            if args.model_config:
                est = estimate_from_config(
                    args.model_config,
                    params_b=args.params_b,
                    ctx_override=ctx,
                    quant_override=args.quant,
                )
            else:
                missing = [k for k, v in dict(
                    params_b=args.params_b, quant=args.quant,
                    num_layers=args.num_layers, num_kv_heads=args.num_kv_heads,
                    head_dim=args.head_dim,
                ).items() if v is None]
                if missing:
                    p.error(f"missing args without --model-config: {missing}")
                est = estimate(
                    params_b=args.params_b,
                    quant=args.quant,
                    num_layers=args.num_layers,
                    num_kv_heads=args.num_kv_heads,
                    head_dim=args.head_dim,
                    ctx=ctx,
                    label=f"{args.params_b:.1f}B-{args.quant}",
                )
            est.label = f"{est.label}@ctx={ctx}"
            print(render(est, args.budget_gb))
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
