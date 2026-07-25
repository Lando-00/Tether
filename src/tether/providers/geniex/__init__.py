"""GenieX provider — marker-only external SSE streaming provider.

Connects to an operator-managed GenieX CLI server (Qualcomm AI Runtime)
exposing OpenAI-compatible endpoints at a configured base URL.  No subprocess
management, model download, or SDK dependency — pure httpx HTTP client.

Provider kind: ``"geniex"``
"""
from tether.providers.geniex.provider import GenieXProvider

__all__ = ["GenieXProvider"]
