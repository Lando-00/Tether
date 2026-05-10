"""
Inventory script for Tether's locally-installed MLC models.

Walks ``models/`` (relative to the repo root by default), parses each
``mlc-chat-config.json``, cross-references compiled libraries in
``models/libs/``, and emits a Markdown report summarising:

* model directory name
* model_type / quantization / context window / prefill chunk size
* whether the conv_template advertises native function calling
* total on-disk size of the weight shards (``params_shard_*.bin``)
* whether a matching compiled library (``*-adreno.dll`` etc.) exists in
  ``models/libs/``

Usage::

    python scripts/research/inventory_models.py                       # print to stdout
    python scripts/research/inventory_models.py --out 01_inventory.md

The script has **no Tether runtime dependencies** so it can be run
mid-refactor.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIST = REPO_ROOT / "models"


def _relpath(p: Path) -> str:
    """Best-effort relative path against the repo root for display only."""
    p = p.resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


# Bits per weight for each MLC quantization label.
QUANT_BITS_PER_WEIGHT: dict[str, float] = {
    "q0f16": 16.0,
    "q0f32": 32.0,
    "q3f16_1": 3.5,
    "q4f16_0": 4.5,
    "q4f16_1": 4.5,
    "q4f32_1": 4.5,
    "q8f16_1": 8.5,
}

LIB_EXT = {"Windows": ".dll", "Darwin": ".dylib"}.get(platform.system(), ".so")


@dataclass
class ModelEntry:
    name: str
    model_type: Optional[str] = None
    quantization: Optional[str] = None
    context_window: Optional[int] = None
    prefill_chunk: Optional[int] = None
    conv_template: Optional[str] = None
    use_function_calling: Optional[bool] = None
    weights_bytes: int = 0
    matching_libs: list[Path] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def weights_gb(self) -> float:
        return self.weights_bytes / (1024 ** 3)

    @property
    def has_lib(self) -> bool:
        return bool(self.matching_libs)

    @property
    def estimated_params_b(self) -> Optional[float]:
        if not self.quantization or self.weights_bytes == 0:
            return None
        bits = QUANT_BITS_PER_WEIGHT.get(self.quantization)
        if bits is None:
            return None
        params = (self.weights_bytes * 8) / bits
        return params / 1e9


def _base_key_from_model_name(model_name: str) -> str:
    s = re.sub(r"([_-]mlc)$", "", model_name, flags=re.IGNORECASE)
    parts = re.split(r"-q\d+f?\d*[_-]?\d*-?mlc", model_name, flags=re.IGNORECASE)
    return parts[0] if parts else s


def _match_libs(model_name: str, dlls: Iterable[Path]) -> list[Path]:
    base_key = _base_key_from_model_name(model_name).lower()
    matches: list[Path] = []
    for dll in dlls:
        if base_key and base_key in dll.name.lower():
            matches.append(dll)
    if not matches:
        loose = model_name.split("-q")[0].lower()
        for dll in dlls:
            if loose in dll.name.lower():
                matches.append(dll)
    return matches


def _sum_weight_shards(model_dir: Path) -> int:
    total = 0
    for shard in model_dir.glob("params_shard_*.bin"):
        try:
            total += shard.stat().st_size
        except OSError:
            pass
    return total


def _read_config(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scan_dist(dist_root: Path) -> list[ModelEntry]:
    libs_dir = dist_root / "libs"
    dlls = list(libs_dir.glob(f"*{LIB_EXT}")) if libs_dir.exists() else []

    entries: list[ModelEntry] = []
    for cfg_path in sorted(dist_root.rglob("mlc-chat-config.json")):
        model_dir = cfg_path.parent
        try:
            model_dir.relative_to(libs_dir)
            continue
        except ValueError:
            pass

        entry = ModelEntry(name=model_dir.name)
        try:
            cfg = _read_config(cfg_path)
            entry.model_type = cfg.get("model_type")
            entry.quantization = cfg.get("quantization")
            entry.context_window = cfg.get("context_window_size")
            entry.prefill_chunk = cfg.get("prefill_chunk_size")
            conv = cfg.get("conv_template") or {}
            entry.conv_template = conv.get("name")
            entry.use_function_calling = bool(conv.get("use_function_calling", False))
        except Exception as exc:
            entry.error = f"config parse error: {exc}"

        entry.weights_bytes = _sum_weight_shards(model_dir)
        entry.matching_libs = _match_libs(model_dir.name, dlls)
        entries.append(entry)

    return entries


def render_markdown(entries: list[ModelEntry], dist_root: Path) -> str:
    lines: list[str] = []
    lines.append("# Tether — Local Model Inventory")
    lines.append("")
    lines.append(
        f"_Auto-generated by `scripts/research/inventory_models.py`. "
        f"Source: `{_relpath(dist_root)}/`._"
    )
    lines.append("")

    if not entries:
        lines.append("> No models found.")
        return "\n".join(lines) + "\n"

    libs_dir = dist_root / "libs"
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Model | Type | Quant | Ctx | Prefill | Conv template | FC | "
        "Weights (GB) | Est. params | Matching lib |"
    )
    lines.append(
        "|-------|------|-------|----:|--------:|----------------|:--:|"
        "-------------:|-------------|--------------|"
    )
    for e in entries:
        params = (
            f"~{e.estimated_params_b:.1f} B" if e.estimated_params_b else "—"
        )
        lib_cell = (
            ", ".join(f"`{p.name}`" for p in e.matching_libs)
            if e.matching_libs else "❌ **none**"
        )
        fc_cell = "✅" if e.use_function_calling else "—"
        conv_cell = f"`{e.conv_template}`" if e.conv_template else "—"
        lines.append(
            f"| `{e.name}` | {e.model_type or '—'} | "
            f"{e.quantization or '—'} | "
            f"{e.context_window or '—'} | "
            f"{e.prefill_chunk or '—'} | "
            f"{conv_cell} | {fc_cell} | "
            f"{e.weights_gb:.2f} | {params} | {lib_cell} |"
        )
    lines.append("")

    lines.append("## Library directory")
    lines.append("")
    if libs_dir.exists():
        dlls = sorted(libs_dir.glob(f"*{LIB_EXT}"))
        if dlls:
            lines.append(
                f"_{len(dlls)} compiled libraries in_ `{_relpath(libs_dir)}/`:"
            )
            lines.append("")
            for dll in dlls:
                size_mb = dll.stat().st_size / (1024 ** 2)
                lines.append(f"- `{dll.name}` ({size_mb:.1f} MB)")
        else:
            lines.append(f"_No `{LIB_EXT}` libraries found in `{libs_dir}`._")
    else:
        lines.append(f"_Libraries directory not found: `{libs_dir}`._")
    lines.append("")

    lines.append("## Highlights")
    lines.append("")
    missing_libs = [e for e in entries if not e.has_lib]
    if missing_libs:
        lines.append(
            "**Models without a matching compiled library** — these "
            "weights are present on disk but the runtime cannot resolve a "
            "library for them and will raise `ValueError: No matching "
            "model library found ...`:"
        )
        lines.append("")
        for e in missing_libs:
            lines.append(f"- `{e.name}`")
        lines.append("")

    small_prefill = [
        e for e in entries
        if e.prefill_chunk and e.prefill_chunk <= 256
    ]
    if small_prefill:
        lines.append(
            "**Small prefill chunks (≤256)** — historically associated "
            "with the OpenCL shutdown-hang fix in "
            "`tether_service/providers/mlc/provider.py`:"
        )
        lines.append("")
        for e in small_prefill:
            lines.append(
                f"- `{e.name}` (prefill_chunk_size = {e.prefill_chunk})"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dist", type=Path, default=DEFAULT_DIST)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.dist.exists():
        print(f"error: dist root not found: {args.dist}", file=sys.stderr)
        return 2

    args.dist = args.dist.resolve()
    entries = scan_dist(args.dist)
    md = render_markdown(entries, args.dist)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"wrote {args.out} ({len(entries)} models)")
    else:
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
