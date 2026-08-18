"""Emit the subset of Tether settings the one-command launcher needs.

``tether.ps1`` shells out to this module so the launcher never has to
duplicate — or drift from — the ports, provider ids and GenieX base URL that
live in ``src/tether/config/default.yml`` (and any ``TETHER__*`` environment
overrides layered on top of it).

Output is a single JSON object on stdout. Any diagnostic noise goes to stderr
so the launcher can parse stdout unconditionally::

    {
      "http": {"host": "127.0.0.1", "port": 8080},
      "api_base_url": "http://127.0.0.1:8080/api/v1",
      "readyz_url": "http://127.0.0.1:8080/api/v1/readyz",
      "default_provider": "geniex",
      "geniex": {
        "provider_id": "geniex",
        "base_url": "http://127.0.0.1:18181",
        "health_url": "http://127.0.0.1:18181/v1/",
        "model_id": "bartowski/Qwen_Qwen3-8B-GGUF:Q4_0"
      }
    }

``geniex`` is ``null`` when no GenieX provider is registered, which is the
launcher's signal to skip that service entirely.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional

_GENIEX_IMPL_MARKER = "geniex"


def _find_geniex(settings: Any) -> Optional[dict]:
    """Return launcher-relevant fields for the registered GenieX provider."""
    registry = getattr(settings.providers, "model_registry", {}) or {}
    default_id = getattr(settings.providers, "default_model_provider", None)

    # Prefer the default provider when it is a GenieX one, so a registry with
    # several entries still warms the provider the user will actually hit.
    ordered_ids = [pid for pid in (default_id,) if pid in registry]
    ordered_ids += [pid for pid in registry if pid not in ordered_ids]

    for provider_id in ordered_ids:
        spec = registry[provider_id]
        impl = (getattr(spec, "impl", "") or "").lower()
        if _GENIEX_IMPL_MARKER not in impl:
            continue
        args = getattr(spec, "args", {}) or {}
        base_url = str(args.get("base_url", "http://127.0.0.1:18181")).rstrip("/")
        return {
            "provider_id": provider_id,
            "base_url": base_url,
            "health_url": f"{base_url}/v1/",
            "model_id": args.get("model_id"),
        }
    return None


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # pragma: no cover - dotenv is optional for the probe
        pass

    from tether.config.settings import load_settings

    settings = load_settings()
    host = settings.http.host
    port = settings.http.port
    # 0.0.0.0 / :: are bind addresses, not dialable ones.
    dial_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    origin = f"http://{dial_host}:{port}"

    payload = {
        "http": {"host": host, "port": port},
        "origin": origin,
        "api_base_url": f"{origin}/api/v1",
        "readyz_url": f"{origin}/api/v1/readyz",
        "default_provider": getattr(
            settings.providers, "default_model_provider", None
        ),
        "geniex": _find_geniex(settings),
    }
    json.dump(payload, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
