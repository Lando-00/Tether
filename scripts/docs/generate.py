"""Regenerate docs/specs/* artefacts from current code.

Run:
    python -m scripts.docs.generate

This writes (overwriting existing files):
    docs/specs/events.schema.json
    docs/specs/settings.schema.json
    docs/specs/openapi.json
    docs/specs/tools.md

Used by the @pytest.mark.docs check to detect drift between code and
shipped specs (synthesis §4 Phase 8 steps 91, 92).

Determinism notes (load-bearing for the drift gate):

* All JSON output uses ``json.dumps(obj, indent=2, sort_keys=True,
  ensure_ascii=False)`` followed by a single trailing newline.
* The tools markdown listing is emitted in sorted order by tool name.
* No timestamps, random IDs, or environment-dependent absolute paths
  appear in the output. The OpenAPI spec includes the ``tether``
  package version from ``pyproject.toml``, which is deterministic from
  the repo state.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import TypeAdapter

from tether.app.http.api import create_app
from tether.config.settings import Settings
from tether.protocol.wire.events import WireEvent
from tether.tools.registration import discover


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "docs" / "specs"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _dump_json(obj: Any) -> str:
    """Deterministic JSON serialisation: indent=2, sorted keys, trailing newline."""
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# events.schema.json
# ---------------------------------------------------------------------------


def regenerate_events_schema() -> str:
    """Build the JSON Schema for the v2 wire-protocol event vocabulary.

    Bundles every concrete event subclass under the ``WireEvent``
    discriminated union from :mod:`tether.protocol.wire.events`.
    """
    adapter = TypeAdapter(WireEvent)
    schema = adapter.json_schema()
    return _dump_json(schema)


# ---------------------------------------------------------------------------
# settings.schema.json
# ---------------------------------------------------------------------------


def regenerate_settings_schema() -> str:
    """Build the JSON Schema for :class:`tether.config.settings.Settings`."""
    schema = Settings.model_json_schema()
    return _dump_json(schema)


# ---------------------------------------------------------------------------
# openapi.json
# ---------------------------------------------------------------------------


def _minimal_settings(db_path: Path) -> Settings:
    """A minimal Settings with DummyProvider + sqlite store at *db_path*.

    Mirrors the fixture pattern used by ``tests/unit/test_engine.py``
    so the FastAPI app can be constructed without booting the real
    MLC model or requiring secrets. The ``disabled`` list keeps the
    four bundled tools out of the registry — they don't affect the
    HTTP route surface, and dropping them avoids touching httpx /
    geocoding APIs at import time.
    """
    dsn = "sqlite:///" + str(db_path).replace(os.sep, "/")
    return Settings.model_validate(
        {
            "providers": {
                "model": {
                    "impl": "tether.providers.dummy.provider.DummyProvider",
                    "args": {},
                },
                "parser": {
                    "impl": "tether.protocol.parsers.sliding.SlidingParser",
                    "args": {},
                },
                "session_store": {
                    "impl": "tether.context.sqlite_store.SqliteSessionStore",
                    "args": {},
                },
            },
            "storage": {
                "sqlite": {"dsn": dsn},
            },
            "tools": {
                "registry": [],
                "enabled": [],
                "disabled": ["time", "weather", "forecast", "web_search"],
            },
        }
    )


def regenerate_openapi() -> str:
    """Build the OpenAPI spec for the FastAPI app via ``app.openapi()``.

    Constructs ``create_app(settings)`` with a minimal Settings using
    :class:`DummyProvider` and a throw-away sqlite DB so model boot
    is never required. The temporary DB file is created in a
    ``tempfile.TemporaryDirectory`` and discarded after schema export.
    """
    with tempfile.TemporaryDirectory(prefix="tether-docsgen-") as tmpdir:
        db_path = Path(tmpdir) / "docsgen.db"
        settings = _minimal_settings(db_path)
        app = create_app(settings)
        spec = app.openapi()
    return _dump_json(spec)


# ---------------------------------------------------------------------------
# tools.md
# ---------------------------------------------------------------------------


def _tool_description(cls: type) -> str:
    """Return the first non-empty paragraph of a tool's class docstring."""
    raw = (cls.__doc__ or "").strip()
    if not raw:
        return ""
    # First paragraph is everything up to the first blank line.
    first_para_lines: List[str] = []
    for line in raw.splitlines():
        if not line.strip():
            if first_para_lines:
                break
            continue
        first_para_lines.append(line.strip())
    return " ".join(first_para_lines)


def _format_default(value: Any) -> str:
    """Render a parameter default for the markdown table."""
    if value is None:
        return "`null`"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    if isinstance(value, (int, float)):
        return f"`{value}`"
    if isinstance(value, str):
        return f"`\"{value}\"`"
    return f"`{json.dumps(value, sort_keys=True)}`"


def _format_type(prop: Dict[str, Any]) -> str:
    """Render a parameter type for the markdown table."""
    nullable = bool(prop.get("nullable"))
    enum = prop.get("enum")
    if enum is not None:
        rendered = " \\| ".join(json.dumps(v) for v in enum)
        base = f"enum({rendered})"
    elif "type" in prop:
        base = str(prop["type"])
        if base == "array":
            items = prop.get("items") or {}
            inner = items.get("type", "any")
            base = f"array<{inner}>"
    else:
        base = "any"
    if nullable:
        base = f"{base} \\| null"
    return f"`{base}`"


def _tool_section(name: str, cls: type) -> str:
    """Build one ``## <name>`` markdown section for a discovered tool."""
    instance = cls()
    schema = instance.auto_schema
    fn = schema.get("function", {}) or {}
    params = fn.get("parameters", {}) or {}
    properties: Dict[str, Any] = params.get("properties", {}) or {}
    required = set(params.get("required", []) or [])

    description = _tool_description(cls)

    lines: List[str] = []
    lines.append(f"## `{name}`")
    lines.append("")
    if description:
        lines.append(description)
        lines.append("")
    lines.append("### Parameters")
    lines.append("")
    if not properties:
        lines.append("_This tool takes no parameters._")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Name | Type | Required | Default | Description |")
    lines.append("|---|---|---|---|---|")
    for param_name in sorted(properties):
        prop = properties[param_name]
        is_required = "yes" if param_name in required else "no"
        default = "—" if param_name in required else _format_default(prop.get("default"))
        type_str = _format_type(prop)
        desc = (prop.get("description") or "").replace("\n", " ").strip() or "—"
        lines.append(
            f"| `{param_name}` | {type_str} | {is_required} | {default} | {desc} |"
        )
    lines.append("")
    return "\n".join(lines)


def regenerate_tools_md() -> str:
    """Build a deterministic markdown listing of every discovered ``@tool``.

    Discovery walks ``tether.tools`` via
    :func:`tether.tools.registration.discover`. Sections are emitted in
    sorted order by tool name so re-running the generator produces
    byte-identical output.
    """
    registered = discover()
    header = [
        "# Built-in Tools",
        "",
        (
            "Auto-generated from `tether.tools` via `scripts/docs/generate.py`. "
            "Do not edit by hand."
        ),
        "",
        "",
    ]
    sections: List[str] = []
    for name in sorted(registered):
        sections.append(_tool_section(name, registered[name]))
    body = "\n".join(sections).rstrip("\n") + "\n"
    return "\n".join(header) + body


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_ARTIFACTS: Tuple[Tuple[str, Any], ...] = (
    ("events.schema.json", regenerate_events_schema),
    ("settings.schema.json", regenerate_settings_schema),
    ("openapi.json", regenerate_openapi),
    ("tools.md", regenerate_tools_md),
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, fn in _ARTIFACTS:
        content = fn()
        (OUTPUT_DIR / filename).write_text(content, encoding="utf-8", newline="\n")
        print(f"wrote {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
