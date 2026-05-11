"""Verify the [whatsapp] PIP extra metadata is well-formed.

Per Phase 2b · ADR-0018 D1. We don't actually install neonize in CI; we
just confirm the project metadata advertises it correctly so a
``pip install tether[whatsapp]`` on the user's machine works.
"""

from __future__ import annotations

from pathlib import Path


def test_whatsapp_extra_lists_neonize_and_phonenumbers():
    import tomllib

    pyproject = Path(__file__).resolve().parents[4] / "pyproject.toml"
    assert pyproject.exists(), f"pyproject.toml not at expected path: {pyproject}"
    cfg = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    extras = cfg["project"]["optional-dependencies"]
    assert "whatsapp" in extras, "[whatsapp] extra missing from pyproject.toml"
    deps = extras["whatsapp"]
    assert any("neonize==0.3.17.post0" in d for d in deps), (
        f"neonize pin not found in [whatsapp] extra: {deps}"
    )
    assert any(d.startswith("phonenumbers") for d in deps), (
        f"phonenumbers not found in [whatsapp] extra: {deps}"
    )


def test_whatsapp_default_yml_commented_example_present():
    """The default.yml should have a commented-out whatsapp example so users
    discover the connector option without enabling it by default."""
    yml = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "tether"
        / "config"
        / "default.yml"
    )
    content = yml.read_text(encoding="utf-8")
    assert "WhatsApp connector example" in content or "whatsapp:" in content, (
        "default.yml missing commented-out WhatsApp connector example"
    )
    # Confirm registry stays empty by default — the example must be
    # commented out so existing tests don't try to load the connector.
    import yaml

    parsed = yaml.safe_load(content)
    registry = parsed.get("connectors", {}).get("registry") or {}
    assert "whatsapp" not in registry, (
        f"whatsapp connector enabled in default.yml registry: {registry}"
    )
