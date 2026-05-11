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
import os
import secrets
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

if TYPE_CHECKING:
    from tether.config.settings import CSRFTokenSettings

logger = logging.getLogger(__name__)

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _atomic_write_token(path: Path, token: str) -> None:
    """Atomically write *token* to *path* with mode 0600.

    Recipe: open-with-mode (mkstemp in same dir) → fchmod 0600 → write →
    fsync → os.replace → best-effort directory fsync. Matches the A5-F2
    pattern in ``tether.core.secrets``. POSIX file-mode bits and dir
    fsync are best-effort: on Windows they no-op, which is acceptable
    because Windows ACLs default to user-only on ``%APPDATA%``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".csrf_token.", dir=str(path.parent))
    try:
        try:
            os.fchmod(fd, 0o600)
        except (AttributeError, NotImplementedError, OSError):
            pass
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(token)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_path, path)
        try:
            dfd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except (OSError, AttributeError):
            pass
    finally:
        if Path(tmp_path).exists():
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass


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
            # P0-B3 / Tribunal P1-10 / ADR-0012: persist the generated CSRF
            # token to a 0600 file so CLI clients can read it
            # deterministically. Stderr fallback retained for unwritable
            # filesystems — never lose the token.
            token_path = self._cfg.resolved_token_file()
            try:
                _atomic_write_token(token_path, self._token)
                logger.info(
                    "csrf.token_generated",
                    extra={
                        "source": "secrets.token_urlsafe(32)",
                        "token_chars": len(self._token),
                        "path": str(token_path),
                    },
                )
            except OSError as exc:
                logger.warning(
                    "csrf.token_persist_failed_falling_back_to_stderr",
                    extra={
                        "path": str(token_path),
                        "error": str(exc),
                    },
                )
                print(
                    f"\n[Tether] CSRF token generated: {self._token}\n"
                    f"[Tether] Token-file write failed ({exc}); pass it as "
                    f"the {settings.header_name!r} header on "
                    f"POST/PUT/PATCH/DELETE.\n",
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
