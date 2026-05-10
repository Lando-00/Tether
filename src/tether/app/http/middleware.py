"""HTTP middleware: request_id binding for structured logging.

Phase 7 step 68. The middleware:
  - reads X-Request-ID from inbound request (or generates a new one)
  - binds it to a structlog contextvar so every log emitted during the
    request includes 'request_id' automatically
  - echoes it on the X-Request-ID response header
  - clears the contextvar at the end of the request

Synthesis §3, B3 step 3.
"""
from __future__ import annotations

import re
import uuid
from typing import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Conservative pattern: alphanumeric, hyphens, underscores. Length 8-128.
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9_\-]{8,128}$")


def _generate_request_id() -> str:
    """Generate a new request_id: 'req-' + 12 hex chars."""
    return f"req-{uuid.uuid4().hex[:12]}"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Binds request_id contextvar for the duration of each HTTP request."""

    HEADER_NAME = "X-Request-ID"

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        inbound = request.headers.get(self.HEADER_NAME, "").strip()
        if inbound and _VALID_REQUEST_ID.match(inbound):
            request_id = inbound
        else:
            request_id = _generate_request_id()

        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
            response.headers[self.HEADER_NAME] = request_id
            return response
        finally:
            structlog.contextvars.unbind_contextvars("request_id")


__all__ = ["RequestIdMiddleware"]
