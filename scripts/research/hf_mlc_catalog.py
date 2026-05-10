"""
Build a catalog of pre-compiled MLC models from the public HuggingFace
``mlc-ai`` organisation.

Uses only the unauthenticated HuggingFace Hub HTTP API (no token needed,
no third-party SDK required).

Usage::

    python hf_mlc_catalog.py ^
        --max-gb 9 ^
        --out-json mlc_ai_hf_catalog.json ^
        --out-md   mlc_ai_hf_catalog.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

HF_API = "https://huggingface.co/api"
DEFAULT_AUTHOR = "mlc-ai"
USER_AGENT = "tether-research/1.0 (+https://github.com/Lando-00/Tether)"

QUANT_PATTERNS = [
    r"q0f16", r"q0f32",
    r"q3f16_1",
    r"q4f16_0", r"q4f16_1", r"q4f32_1",
    r"q8f16_0", r"q8f16_1", r"q8f32_1",
    r"e5m2_e5m2_f16", r"e4m3_e4m3_f16",
]
QUANT_RE = re.compile(r"-(" + "|".join(QUANT_PATTERNS) + r")(?:-|$)", re.I)
PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")
INSTRUCT_HINTS = ("instruct", "chat", "it", "tool", "tools")


@dataclass
class CatalogEntry:
    repo_id: str
    family: Optional[str]
    params_b: Optional[float]
    quant: Optional[str]
    total_bin_bytes: int = 0
    n_bin_files: int = 0
    likes: int = 0
    downloads: int = 0
    last_modified: Optional[str] = None
    tags: list[str] = dataclasses.field(default_factory=list)
    error: Optional[str] = None

    @property
    def total_gb(self) -> float:
        return self.total_bin_bytes / (1024 ** 3)


def _http_get_json(url: str, retries: int = 3, backoff: float = 1.5) -> Any:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 504):
                last_err = e
                time.sleep(backoff ** attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            time.sleep(backoff ** attempt)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_err}")


def list_models(author: str) -> list[dict]:
    url = (
        f"{HF_API}/models?author={urllib.parse.quote(author)}"
        f"&limit=2000&full=true"
    )
    data = _http_get_json(url)
    if not isinstance(data, list):
        raise RuntimeError(f"unexpected payload type from {url}: {type(data)}")
    return data


def get_repo_tree(repo_id: str) -> list[dict]:
    url = f"{HF_API}/models/{urllib.parse.quote(repo_id, safe='/')}/tree/main?recursive=true"
    return _http_get_json(url)


def parse_repo_name(repo_id: str) -> tuple[Optional[str], Optional[float], Optional[str]]:
    name = repo_id.split("/", 1)[-1]

    quant_match = QUANT_RE.search(name)
    quant = quant_match.group(1).lower() if quant_match else None

    params_match = PARAM_RE.search(name)
    params_b: Optional[float] = None
    if params_match:
        try:
            params_b = float(params_match.group(1))
        except ValueError:
            params_b = None

    family = name
    if params_match:
        family = name[: params_match.start()].rstrip("-_")
    elif quant_match:
        family = name[: quant_match.start()].rstrip("-_")
    parts = family.split("-")
    while parts and parts[-1].lower() in INSTRUCT_HINTS:
        parts.pop()
    family = "-".join(parts) if parts else family

    return (family or None), params_b, quant


def total_weight_bytes(tree: Iterable[dict]) -> tuple[int, int]:
    total = 0
    n = 0
    for node in tree:
        if node.get("type") != "file":
            continue
        path = node.get("path", "")
        if path.endswith(".bin") and ("params_shard" in path or "shard" in path or path.startswith("params_")):
            total += int(node.get("size", 0) or 0)
            n += 1
    return total, n


def build_catalog(author: str, max_gb: float, *, fetch_trees: bool = True,
                  verbose: bool = False) -> list[CatalogEntry]:
    print(f"listing repos for author={author} ...", file=sys.stderr)
    repos = list_models(author)
    print(f"  {len(repos)} repos", file=sys.stderr)

    out: list[CatalogEntry] = []
    for i, repo in enumerate(repos):
        repo_id = repo.get("id") or repo.get("modelId")
        if not repo_id:
            continue
        family, params_b, quant = parse_repo_name(repo_id)
        entry = CatalogEntry(
            repo_id=repo_id,
            family=family,
            params_b=params_b,
            quant=quant,
            likes=int(repo.get("likes") or 0),
            downloads=int(repo.get("downloads") or 0),
            last_modified=repo.get("lastModified"),
            tags=list(repo.get("tags") or []),
        )

        if fetch_trees:
            try:
                tree = get_repo_tree(repo_id)
                if isinstance(tree, list):
                    entry.total_bin_bytes, entry.n_bin_files = total_weight_bytes(tree)
            except Exception as exc:
                entry.error = f"tree fetch failed: {exc}"
            if verbose and (i + 1) % 25 == 0:
                print(f"  fetched {i+1}/{len(repos)}", file=sys.stderr)

        out.append(entry)

    cap = int(max_gb * (1024 ** 3))
    kept = [e for e in out if e.total_bin_bytes == 0 or e.total_bin_bytes <= cap]
    print(
        f"  filtered to {len(kept)}/{len(out)} repos <= {max_gb} GB on disk",
        file=sys.stderr,
    )
    return kept


def render_markdown(entries: list[CatalogEntry], author: str, max_gb: float) -> str:
    families: dict[str, list[CatalogEntry]] = {}
    for e in entries:
        key = e.family or "(unknown)"
        families.setdefault(key, []).append(e)

    def family_sort_key(name: str) -> tuple[int, str]:
        peak = max((e.likes for e in families[name]), default=0)
        return (-peak, name.lower())

    md: list[str] = []
    md.append(f"# HuggingFace `{author}` MLC Catalog")
    md.append("")
    md.append(
        f"_Auto-generated by `hf_mlc_catalog.py`. "
        f"Filtered to repos with weight shards totalling <= {max_gb} GB "
        f"on disk (or unknown size)._"
    )
    md.append("")
    md.append(f"Total repos retained: **{len(entries)}**.")
    md.append("")

    for family in sorted(families.keys(), key=family_sort_key):
        rows = sorted(
            families[family],
            key=lambda e: (-(e.params_b or 0), e.quant or ""),
        )
        md.append(f"## {family}")
        md.append("")
        md.append("| Repo | Params | Quant | GB | Likes | DLs | Last modified |")
        md.append("|------|-------:|-------|---:|------:|----:|---------------|")
        for e in rows:
            params = f"{e.params_b:.1f}" if e.params_b is not None else "—"
            quant = e.quant or "—"
            gb = f"{e.total_gb:.2f}" if e.total_bin_bytes else "—"
            last_mod = (e.last_modified or "")[:10] if e.last_modified else "—"
            md.append(
                f"| [{e.repo_id}](https://huggingface.co/{e.repo_id}) | "
                f"{params} | {quant} | {gb} | {e.likes} | {e.downloads} | {last_mod} |"
            )
        md.append("")

    return "\n".join(md) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--author", default=DEFAULT_AUTHOR)
    parser.add_argument("--max-gb", type=float, default=9.0)
    parser.add_argument("--no-trees", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        entries = build_catalog(
            args.author,
            args.max_gb,
            fetch_trees=not args.no_trees,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps([dataclasses.asdict(e) for e in entries], indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.out_json}")

    md = render_markdown(entries, args.author, args.max_gb)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
        print(f"wrote {args.out_md}")
    elif not args.out_json:
        sys.stdout.write(md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
