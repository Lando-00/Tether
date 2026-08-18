"""
Set up a new MLC model for Tether on Snapdragon X Elite.

End-to-end pipeline that:

1. Downloads a pre-quantized MLC model from a HuggingFace repo (e.g.
   ``mlc-ai/Qwen3-4B-q4f16_1-MLC``) into
   ``<models-root>/<output_name>/``.
2. Optionally patches the ``mlc-chat-config.json`` to swap in a
   ``conv_template`` block copied from an existing model directory --
   this is how we maintain Tether's custom ``qwen3-openai-tools-min``
   tool-calling steering across new model downloads.
3. Compiles an Adreno-OpenCL DLL to
   ``<models-root>/libs/<output_name>-adreno.dll``
   via ``mlc_llm compile --device windows:adreno_x86``.
4. Optionally runs a smoke test: load the model in ``AsyncMLCEngine``
   and stream one short completion to confirm the GPU path works
   end-to-end.

Designed to be re-runnable for future model adds. Stdlib only besides
``huggingface_hub`` (already in ``mlc-venv2``) and ``mlc_llm`` itself.

Background:
* ``mlc-ai/*-MLC`` HF repos publish weights + config but **no Adreno
  DLL** -- Vulkan/CUDA libs at most. The compile step is therefore
  required for our hardware.
* Tether targets the Adreno X1 GPU via OpenCL through Qualcomm's
  CodeLinaro MLC-LLM ``2025.06.r1`` distribution.
* Practical context window depends on KV-cache RAM, not the
  advertised ``context_window_size``.

Usage::

    python scripts\\setup_model.py ^
        --hf-repo mlc-ai/Qwen3-4B-q4f16_1-MLC ^
        --models-root .\\models ^
        --conv-template-from .\\models\\Qwen3-4B-q4f16_0-MLC ^
        --verify

    # subsequent runs can reuse the staged weights
    python scripts\\setup_model.py ^
        --hf-repo mlc-ai/Qwen3-4B-q4f16_1-MLC ^
        --models-root .\\models ^
        --skip-download

The script writes nothing into Tether's source tree -- only into the
``--models-root`` directory you point it at (defaults to ``./models/``).
The ``--dist`` flag is kept as a deprecated alias for one cycle.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_ROOT = REPO_ROOT / "models"
ADRENO_DEVICE = "windows:adreno_x86"  # per Qualcomm Feb 2025 blog
LIB_EXT = {"Windows": ".dll", "Darwin": ".dylib"}.get(platform.system(), ".so")


def _print(msg: str, *, level: str = "info") -> None:
    prefix = {"info": "==>", "warn": "WARN:", "ok": "OK:"}.get(level, "==>")
    print(f"{prefix} {msg}", flush=True)


def _run(cmd: list[str], *, env: Optional[dict] = None) -> int:
    """Run a subprocess streaming stdout/stderr, return its exit code."""
    _print(" ".join(cmd))
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def step_download(hf_repo: str, target_dir: Path) -> None:
    """Snapshot a HF repo into ``target_dir`` using the local hub cache."""
    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    _print(f"downloading {hf_repo} -> {target_dir}")
    snapshot_download(
        repo_id=hf_repo,
        local_dir=str(target_dir),
    )
    total_bytes = sum(p.stat().st_size for p in target_dir.rglob("*") if p.is_file())
    _print(f"download complete: {total_bytes / (1024**3):.2f} GB on disk", level="ok")


def step_patch_conv_template(target_cfg: Path, source_cfg: Path) -> None:
    """Copy the ``conv_template`` block from ``source_cfg`` into ``target_cfg``."""
    if not source_cfg.exists():
        raise FileNotFoundError(f"source config not found: {source_cfg}")
    if not target_cfg.exists():
        raise FileNotFoundError(f"target config not found: {target_cfg}")

    src = json.loads(source_cfg.read_text(encoding="utf-8"))
    tgt = json.loads(target_cfg.read_text(encoding="utf-8"))

    if "conv_template" not in src:
        raise ValueError(f"no conv_template in {source_cfg}")

    old_template = (tgt.get("conv_template") or {}).get("name", "<missing>")
    new_template = src["conv_template"].get("name", "<unknown>")
    tgt["conv_template"] = src["conv_template"]
    target_cfg.write_text(json.dumps(tgt, indent=2), encoding="utf-8")
    _print(
        f"patched conv_template in {target_cfg.name}: "
        f"{old_template!r} -> {new_template!r}",
        level="ok",
    )


def step_compile(model_dir: Path, lib_out: Path, device: str) -> None:
    """Run ``mlc_llm compile`` to produce an Adreno-targeted DLL."""
    cfg = model_dir / "mlc-chat-config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"mlc-chat-config.json missing in {model_dir}")
    lib_out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlc_llm", "compile",
        str(cfg),
        "--device", device,
        "-o", str(lib_out),
    ]
    code = _run(cmd)
    if code != 0:
        raise RuntimeError(f"mlc_llm compile failed with exit code {code}")
    if not lib_out.exists():
        raise RuntimeError(f"compile reported success but {lib_out} is missing")
    _print(
        f"compiled {lib_out.name} ({lib_out.stat().st_size / (1024**2):.1f} MB)",
        level="ok",
    )


async def _smoke_test_async(model_dir: Path, lib_path: Path) -> None:
    from mlc_llm import AsyncMLCEngine

    _print(f"loading {model_dir.name} on Adreno OpenCL ...")
    engine = AsyncMLCEngine(
        model=str(model_dir),
        model_lib=str(lib_path),
        device="auto",
        mode="interactive",
    )
    try:
        _print("streaming a short test prompt ...")
        out_chars = 0
        request_id = "tether-setup-smoke"
        gen = await engine.chat.completions.create(
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=16,
            stream=True,
            request_id=request_id,
        )
        async for resp in gen:
            for choice in resp.choices:
                delta = choice.delta.content or ""
                if delta:
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                    out_chars += len(delta)
        sys.stdout.write("\n")
        sys.stdout.flush()
        if out_chars == 0:
            raise RuntimeError("smoke test produced no tokens")
        _print(f"smoke test produced {out_chars} chars of output", level="ok")
    finally:
        try:
            engine.terminate()
        except Exception as exc:
            _print(f"engine.terminate() raised: {exc!r}", level="warn")


def step_verify(model_dir: Path, lib_path: Path) -> None:
    asyncio.run(_smoke_test_async(model_dir, lib_path))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--hf-repo", required=True,
                   help="HuggingFace repo id, e.g. mlc-ai/Qwen3-4B-q4f16_1-MLC")
    p.add_argument("--output-name", default=None,
                   help="Directory name under <models-root> (default: last segment of hf-repo)")
    p.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT,
                   help=f"Path to the models/ root (default: {DEFAULT_MODELS_ROOT}). "
                        "This was named --dist before the Phase 8 rename; --dist is "
                        "kept as a deprecated alias.")
    p.add_argument("--dist", type=Path, default=None,
                   help=argparse.SUPPRESS)  # deprecated alias for --models-root
    p.add_argument("--device", default=ADRENO_DEVICE,
                   help=f"mlc_llm compile --device value (default: {ADRENO_DEVICE})")
    p.add_argument("--conv-template-from", type=Path, default=None,
                   help="Optional model dir or mlc-chat-config.json whose conv_template block should overwrite the freshly-downloaded one")
    p.add_argument("--skip-download", action="store_true",
                   help="Reuse already-downloaded weights in <models-root>/<output_name>/")
    p.add_argument("--skip-compile", action="store_true")
    p.add_argument("--skip-patch", action="store_true",
                   help="Don't patch conv_template even if --conv-template-from is set")
    p.add_argument("--verify", action="store_true",
                   help="After compile, load the model and run a short smoke test")
    p.add_argument("--force", action="store_true",
                   help="Allow overwriting an existing output_name directory")
    args = p.parse_args(argv)
    # Phase-8 rename compat: --dist remains a deprecated alias for --models-root.
    if args.dist is not None:
        print(
            "WARN: --dist is deprecated since the dist/->models/ rename; "
            "use --models-root instead.",
            file=sys.stderr,
        )
        args.models_root = args.dist
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    output_name = args.output_name or args.hf_repo.split("/", 1)[-1]
    target_dir = (args.models_root / output_name).resolve()
    libs_dir = (args.models_root / "libs").resolve()

    base_lib_name = output_name.removesuffix("-MLC")
    lib_out = libs_dir / f"{base_lib_name}-adreno{LIB_EXT}"

    _print(f"output model dir: {target_dir}")
    _print(f"output lib path:  {lib_out}")

    if not args.skip_download:
        if target_dir.exists() and any(target_dir.iterdir()):
            if not args.force:
                _print(
                    f"{target_dir} already exists and is non-empty; "
                    "use --skip-download to reuse it or --force to overwrite",
                    level="warn",
                )
                return 2
            shutil.rmtree(target_dir)
        step_download(args.hf_repo, target_dir)
    else:
        if not target_dir.exists():
            _print(f"--skip-download set but {target_dir} does not exist", level="warn")
            return 2
        _print(f"skipping download; reusing {target_dir}")

    if args.conv_template_from and not args.skip_patch:
        src = args.conv_template_from
        if src.is_dir():
            src = src / "mlc-chat-config.json"
        step_patch_conv_template(target_dir / "mlc-chat-config.json", src)

    if not args.skip_compile:
        step_compile(target_dir, lib_out, args.device)
    else:
        _print("skipping compile")

    if args.verify:
        if not lib_out.exists():
            _print(f"--verify requested but {lib_out} is missing", level="warn")
            return 3
        step_verify(target_dir, lib_out)

    _print("setup complete", level="ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
