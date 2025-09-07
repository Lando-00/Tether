#!/usr/bin/env python
"""
MLC-LLM Windows launcher for Snapdragon (Adreno OpenCL).

Features:
- Scans `dist/**/mlc-chat-config.json` to find compiled MLC model folders.
- Scans `dist/libs/*.dll` for an Adreno-targeted library and auto-matches to the model.
- Interactive menu to choose model, library, device (opencl/cpu), runner (python/native).
- Runs either:
    python -m mlc_llm chat --device opencl --model-lib <dll> <model_dir>
  or:
    mlc_cli_chat.exe --model <model_dir> --model-lib <dll> --device opencl [--with-prompt "..."]

Requirements:
- Run inside your Conda env that has mlc-llm + tvm Adreno wheels installed (mlc-venv).
- Optional: Set MLC_UTILS_BIN to the directory containing mlc_cli_chat.exe to use native runner.

Usage (interactive):
    conda activate mlc-venv
    python run_mlc_chat.py
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import re

def find_models(dist_root: Path):
    """Find model dirs that contain mlc-chat-config.json; return list of dicts."""
    results = []
    for cfg in dist_root.glob("**/mlc-chat-config.json"):
        model_dir = cfg.parent
        # Skip libs folder just in case
        if "libs" in model_dir.parts:
            continue
        shards = list(model_dir.glob("params_shard_*.bin"))
        has_cache = (model_dir / "ndarray-cache.json").exists()
        results.append({
            "name": model_dir.name,
            "dir": model_dir,
            "shard_count": len(shards),
            "has_cache": has_cache
        })
    # Sort by name for stable display
    results.sort(key=lambda x: x["name"].lower())
    return results

def find_dlls(libs_dir: Path):
    """Return list of .dll paths in dist/libs."""
    if not libs_dir.exists():
        return []
    dlls = list(libs_dir.glob("*.dll"))
    dlls.sort(key=lambda p: p.name.lower())
    return dlls

def base_key_from_model_name(model_name: str) -> str:
    """
    Extract a base key from model folder name to match DLLs.
    E.g. 'Qwen2.5-7B-q4f16_0-MLC' -> 'Qwen2.5-7B'
         'DeepSeek-R1-Distill-Qwen-1.5B-q4f16_0-MLC' -> 'DeepSeek-R1-Distill-Qwen-1.5B'
    """
    m = re.split(r"-q\d+f?\d*[_-]?\d*-?mlc", model_name, flags=re.IGNORECASE)
    return m[0] if m else model_name

def match_model_dlls(model_name: str, dlls):
    base_key = base_key_from_model_name(model_name).lower()
    matches = []
    for dll in dlls:
        name = dll.name.lower()
        if base_key and base_key in name:
            matches.append(dll)
    # Fallback: if nothing matched, try looser heuristic on tokens before '-q'
    if not matches:
        loose_key = model_name.split("-q")[0].lower()
        for dll in dlls:
            if loose_key in dll.name.lower():
                matches.append(dll)
    return matches

def choose(prompt: str, options: list, default_index: int = 0) -> int:
    """Simple numeric chooser."""
    while True:
        raw = input(f"{prompt} [1-{len(options)}] (default {default_index+1}): ").strip()
        if not raw:
            return default_index
        if raw.isdigit():
            i = int(raw) - 1
            if 0 <= i < len(options):
                return i
        print("Invalid choice. Please enter a valid number.")

def run_subprocess(args_list, cwd: Path = None):
    print("\n> Running command:\n  " + " ".join(f'"{a}"' if " " in str(a) else str(a) for a in args_list))
    proc = subprocess.run(args_list, cwd=str(cwd) if cwd else None)
    return proc.returncode

def main():
    parser = argparse.ArgumentParser(description="Interactive MLC-LLM launcher for Windows (Adreno OpenCL).")
    parser.add_argument("--base-dir", default=".", help="Project base directory (default: current dir).")
    parser.add_argument("--runner", choices=["python", "native"], default="python",
                        help="Use python module runner or native mlc_cli_chat.exe (default: python).")
    parser.add_argument("--native-bin", default=os.environ.get("MLC_UTILS_BIN", ""),
                        help="Path to folder containing mlc_cli_chat.exe (used if --runner native).")
    parser.add_argument("--device", choices=["opencl", "cpu", "auto"], default="opencl",
                        help="Device for chat (default: opencl).")
    parser.add_argument("--oneshot", default="", help="If set, sends one prompt and exits (no history).")
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    dist = base / "dist"
    libs = dist / "libs"

    if not dist.exists():
        print(f"ERROR: '{dist}' not found. Run this from your project folder (with 'dist').")
        sys.exit(1)

    # 1) Discover models and libs
    models = find_models(dist)
    dlls = find_dlls(libs)

    if not models:
        print("No models found. I look for folders containing 'mlc-chat-config.json' under 'dist/'.")
        print("Example: dist\\Qwen2.5-7B-q4f16_0-MLC\\mlc-chat-config.json")
        sys.exit(1)

    print("\nDiscovered models:")
    for idx, m in enumerate(models, start=1):
        maybe = match_model_dlls(m["name"], dlls)
        tag = f"{len(maybe)} dll match(es)" if maybe else "no dll match"
        shards = f"{m['shard_count']} shard(s)" if m['shard_count'] else "0 shards?"
        print(f"  {idx:>2}. {m['name']}   [{shards}; {tag}]")

    # 2) Choose model
    mi = choose("\nSelect a model", models, default_index=0)
    model = models[mi]
    model_dir: Path = model["dir"]

    # 3) Choose dll (if any)
    matches = match_model_dlls(model["name"], dlls)
    dll_path = None
    if matches:
        if len(matches) == 1:
            dll_path = matches[0]
            print(f"\nUsing DLL: {dll_path.name}")
        else:
            print("\nMultiple DLLs matched:")
            for i, d in enumerate(matches, start=1):
                print(f"  {i}. {d.name}")
            di = choose("Select a DLL", matches, default_index=0)
            dll_path = matches[di]
    else:
        print("\nNo matching DLL found in 'dist\\libs'.")
        print(" - If you want GPU (OpenCL), be sure you compiled the model to a DLL in dist\\libs.")
        print(" - You can still run on CPU without --model-lib.")

    # 4) Device selection
    device = args.device
    if device == "auto":
        device = "opencl" if dll_path else "cpu"
    print(f"\nSelected device: {device.upper()}")

    # 5) Runner selection
    runner = args.runner
    if runner == "native":
        bin_dir = Path(args.native_bin) if args.native_bin else None
        if not bin_dir or not bin_dir.exists():
            print("\nERROR: --runner native selected but --native-bin not set or invalid.")
            print("Set --native-bin to the folder containing mlc_cli_chat.exe, or set env var MLC_UTILS_BIN.")
            sys.exit(1)
        exe = bin_dir / "mlc_cli_chat.exe"
        if not exe.exists():
            print(f"\nERROR: {exe} not found. Make sure mlc_cli_chat.exe is in {bin_dir}")
            sys.exit(1)

        # Compose command
        cmd = [str(exe), "--model", str(model_dir), "--device", device]
        if dll_path and device == "opencl":
            cmd += ["--model-lib", str(dll_path)]
        if args.oneshot:
            cmd += ["--with-prompt", args.oneshot]

    else:  # python runner
        # Ensure mlc_llm is importable
        try:
            import mlc_llm  # noqa: F401
        except Exception as e:
            print("\nERROR: Could not import mlc_llm in this environment. Activate your conda env (mlc-venv).")
            print(f"Details: {e}")
            sys.exit(1)

        cmd = [sys.executable, "-m", "mlc_llm", "chat", "--device", device]
        if dll_path and device == "opencl":
            cmd += ["--model-lib", str(dll_path)]
        cmd += [str(model_dir)]
        if args.oneshot:
            # python runner doesn't have a single-flag oneshot, so we just pass a prompt to stdin?
            # Instead, advise user to type their prompt interactively or switch to native runner with --oneshot.
            print("\nNOTE: --oneshot works best with the native runner. Launching interactive chat...")

    # 6) Run
    rc = run_subprocess(cmd, cwd=base)
    if rc != 0:
        print(f"\nProcess exited with code {rc}")
    else:
        print("\nDone.")

if __name__ == "__main__":
    main()
