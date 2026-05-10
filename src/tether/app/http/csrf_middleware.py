"""CSRF token middleware.

Phase 7 step 79. Synthesis Section 3 (security), B5 steps 11-12.

When enabled, requires an ``X-Tether-CSRF`` header (configurable name) on
state-changing requests (POST/PUT/PATCH/DELETE). Token is matched in
constant time against a configured static value or a server-generated
session token printed to stderr once at startup.

GET/HEAD/OPTIONS requests are exempt (CSRF doesn't apply to safe methods).
Configured exempt paths (e.g., healthz) are also bypassed for both GETs
and POSTs (used for readyz that may need POST per future API design).
"""

from __future__ import annotations

import hmac
import logging
import secrets
import sys
from typing import TYPE_CHECKING, Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

if TYPE_CHECKING:
    from tether.config.settings import CSRFTokenSettings

logger = logging.getLogger(__name__)

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


class CSRFTokenMiddleware(BaseHTTPMiddleware):
    """ASGI middleware enforcing CSRF token on mutating /api/* requests."""

    def __init__(self, app, settings: CSRFTokenSettings) -> None:
        super().__init__(app)
        self._cfg = settings
        if settings.token:
            self._token = settings.token
            self._token_source = "configured"
        else:
            self._token = secrets.token_urlsafe(32)
            self._token_source = "generated"
            # Log existence (not the value) for ops auditing.  The token
            # itself must not reach log files — it's a long-lived secret and
            # the JSON log is append-only.  Print to stderr instead; stderr
            # is not captured by the file handlers in core/logging.py.
            logger.info(
                "csrf.token_generated",
                extra={
                    "source": "secrets.token_urlsafe(32)",
                    "token_chars": len(self._token),
                },
            )
            print(
                f"\n[Tether] CSRF token generated: {self._token}\n"
                f"[Tether] Pass it as the {settings.header_name!r} header"
                " on POST/PUT/PATCH/DELETE requests.\n",
                file=sys.stderr,
                flush=True,
            )

        self._exempt = {p.rstrip("/") for p in settings.exempt_paths}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.method in _SAFE_METHODS:
            return await call_next(request)

        path_norm = request.url.path.rstrip("/")
        if path_norm in self._exempt:
            return await call_next(request)

        provided = request.headers.get(self._cfg.header_name)
        if provided is None:
            return JSONResponse(
                status_code=403,
                content={"error": "csrf_token_missing", "header": self._cfg.header_name},
            )
        if not hmac.compare_digest(provided, self._token):
            return JSONResponse(
                status_code=403,
                content={"error": "csrf_token_invalid"},
            )

        return await call_next(request)
