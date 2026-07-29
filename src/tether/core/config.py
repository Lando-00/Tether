import os
from importlib import resources
from typing import Any, Dict

import yaml


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _yaml_load_text(path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_settings_legacy() -> Dict[str, Any]:
    """
    DEPRECATED (one-cycle coexistence): legacy ``dict``-returning loader.

    Use :func:`tether.config.settings.load_settings` for new code; that
    returns a typed :class:`tether.config.settings.Settings`. This
    function is preserved unchanged so that existing call sites continue to
    work; ``p2-cleanup`` will migrate them and delete this name.

    Loads default.yml, overlays dev.yml if present, then applies env overrides
    using ``TETHER__A__B=val`` -> ``cfg['a']['b']=parsed(val)``.
    """
    # base config dir within package
    pkg_root = resources.files("tether.config")
    cfg = _yaml_load_text(pkg_root / "default.yml")

    dev_file = pkg_root / "dev.yml"
    if dev_file.is_file():
        dev_cfg = _yaml_load_text(dev_file)
        # If dev config has `_replaces_default: true`, use it as the base
        if dev_cfg.pop("_replaces_default", False):
            cfg = dev_cfg
        else:
            # Otherwise, merge it over the default config
            cfg = deep_merge(cfg, dev_cfg)

    # Apply env overrides
    prefix = "TETHER__"
    for key, val in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].split("__")
        parts = [p.strip().lower() for p in parts if p.strip()]
        if not parts:
            continue
        sub = cfg
        for p in parts[:-1]:
            # create intermediate dicts
            sub = sub.setdefault(p, {})
        # parse value as YAML for numbers/bools/lists/dicts support
        try:
            parsed = yaml.safe_load(val)
        except Exception:
            parsed = val
        sub[parts[-1]] = parsed

    return cfg
