"""Phase-9 P0-B2: require application/json on mutating routes.

Mitigates the browser CORS-simple-POST + DNS-rebinding vector by forcing a
preflight requirement (any Content-Type != application/json on POST/PUT/
PATCH/DELETE triggers preflight; no-Content-Type or text/plain etc are
simple requests and bypass CORS today). Tribunal §3 P0-04 / B3-P0-3.
"""
from __future__ import annotations

from typing import Awaitable, Callable, ClassVar

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.get_logger(__name__)

_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


class RequireJsonContentTypeMiddleware(BaseHTTPMiddleware):
    """Reject mutating requests without ``application/json`` content type."""

    # Routes that legitimately need other content types (e.g., OAuth callback
    # is GET; not mutating; skipped automatically).  File uploads etc. would
    # be skipped here if the app ever gains them.
    _EXEMPT: ClassVar[frozenset[str]] = frozenset({
        "/api/v1/connectors",  # OAuth callback subroute is GET only
    })

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.method not in _MUTATING_METHODS:
            return await call_next(request)

        ct = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        if ct != "application/json":
            logger.warning(
                "http.content_type.rejected",
                method=request.method,
                path=request.url.path,
                received_content_type=ct or "<missing>",
            )
            return JSONResponse(
                status_code=415,
                content={
                    "error": "unsupported_media_type",
                    "detail": (
                        "Mutating requests must use Content-Type: application/json. "
                        "This protects against browser CORS-simple-POST attacks."
                    ),
                },
            )
        return await call_next(request)


__all__ = ["RequireJsonContentTypeMiddleware"]
