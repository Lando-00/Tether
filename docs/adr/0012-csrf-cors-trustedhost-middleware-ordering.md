# ADR-0012: CSRF token + CORS + TrustedHost middleware ordering

- **Status**: Accepted (Phase 7 of refactor)
- **Date**: 2026-05 (Phase 7)
- **Synthesis digest**: [`../refactor/synthesis-2026-05.md`](../refactor/synthesis-2026-05.md)
  `csrf_token`)

## Context

Even though Tether is a single-user local-first app with the documented non-goal of
authentication, it exposes a localhost HTTP API on a known port. A malicious web page in
the user's browser can trivially issue cross-origin POSTs to `localhost:8000` from
JavaScript and (without protection) cause Tether to take destructive actions on the user's
behalf — classic CSRF. CORS misconfiguration would compound this. Pre-refactor, a
"local_token" Settings field existed but was misnamed: it did not authenticate (there is
nothing TO authenticate, single-user) — it served exclusively as a CSRF token. The
middleware mounting order was also undocumented; FastAPI/Starlette runs middlewares in
**last-added = outermost** order, which is counter-intuitive enough to require codification.

## Decision

Adopt three optional middlewares, all default-off, with a **codified mounting order**, and
rename `local_token` → `csrf_token` per §11 R12:

- **`Settings.security.csrf_token.enabled = False` default**. When enabled, Tether writes
  a random token to `~/.tether/token` (mode 0600); the CLI reads the same file. State-
  changing requests (POST/PUT/PATCH/DELETE) require an `X-Tether-Token: <value>` header.
  This is **CSRF protection on the localhost API, not authentication** — naming reflects
  intent.
- **`Settings.security.cors.enabled = False` default**. When enabled, takes an explicit
  `allow_origins: list[str]` (no wildcards), `allow_credentials: bool`. No browser fronts
  Tether by default; opt-in is the only sensible posture.
- **`Settings.security.trusted_host.enabled = False` default**. When enabled, takes an
  `allowed_hosts: list[str]` (e.g. `["localhost", "127.0.0.1"]`) to reject Host-header
  spoofing.
- **Mounting order** in `create_asgi_app()` matters and is documented in code:
  ```python
  # FastAPI/Starlette: LAST add_middleware = OUTERMOST.
  # Inbound order: TrustedHost → CORS → RequestId → CSRF → routers.
  # Therefore the add_middleware calls happen in REVERSE order:
  app.add_middleware(CSRFMiddleware, ...)         # innermost (state-changing checks)
  app.add_middleware(RequestIdMiddleware, ...)    # binds correlation
  app.add_middleware(CORSMiddleware, ...)         # browser preflight handling
  app.add_middleware(TrustedHostMiddleware, ...)  # outermost (cheapest reject)
  ```
- **Why this order**: we want the cheapest, least-side-effecting rejection (host header)
  to fire first; CORS preflight before request-id binding (don't pollute logs with
  preflight noise); CSRF after request-id (so blocked requests still get a request_id
  in logs); routers last.
- **Independent of outbound allowlist** (ADR-0011): CSRF/CORS/TrustedHost protect the
  localhost API from external attackers; outbound URL safety protects the user from the
  model. Different threat models, both needed.
- **Documented**: `docs/runbooks/` to gain a "hardening localhost API" runbook when the
  first user enables it.

## Consequences

### Positive
- Mounting order is no longer folklore — it is documented in source and in this ADR.
- Single-user laptops stay frictionless (everything default-off); paranoid users get
  layered defense with three independent flips.
- Renaming `local_token` → `csrf_token` removes a misleading name that suggested
  authentication semantics where none existed.

### Negative
- Three independent settings flips for full hardening — no "secure-mode" preset. We
  considered a preset; rejected because the right allowlist is environment-specific.
- Renaming `local_token` is a config-schema break; mitigated by a one-cycle
  back-compat alias that emits `DeprecationWarning`.

### Trade-offs accepted
- Default-off security: a browser-CSRF attack against a default Tether install is
  blocked only by the browser's same-origin policy. We accept this for v1; the
  non-goal "no auth" makes any default-on token incompatible with frictionless CLI use.

## Alternatives considered

- **Default-on CSRF** — rejected: would force every CLI invocation to read the token
  file; for single-user the cost outweighs the benefit. Easy enable when user wants it.
- **Single "secure mode" preset** — rejected: every deployment needs different
  `allow_origins`/`allowed_hosts`. A preset would be wrong for everyone.
- **Bake middleware order into a private helper** — partially adopted: `create_asgi_app`
  centralizes the order, but the comment block is the canonical reference so newcomers
  don't reorder by accident.

## References

- Synthesis digest: [`../refactor/synthesis-2026-05.md`](../refactor/synthesis-2026-05.md)
- `src/tether/adapters/http/api.py` (`create_asgi_app` middleware mount block)
- ADR-0011 (outbound URL allowlist — complementary outbound-side defense)
- Starlette docs: `https://www.starlette.io/middleware/` (last-added = outermost)


## Implementation status (2026-05 Phase 9 P0-B2)

Tribunal Section 3 P0-04 (A4-F4 / B3-P0-3) flipped TrustedHost to default-on:

- `security.trusted_host.enabled` now defaults to `True`.
- `allowed_hosts` defaults expanded to include IPv6 loopback:
  `["localhost", "127.0.0.1", "[::1]", "::1"]`.
- New `RequireJsonContentTypeMiddleware` (in
  `src/tether/app/http/content_type_middleware.py`) rejects mutating
  requests (POST/PUT/PATCH/DELETE) lacking `Content-Type: application/json`
  with HTTP 415. This closes the browser CORS-simple-POST + DNS-rebinding
  read-primitive vector: any non-JSON CT now triggers a CORS preflight,
  and missing/text content types are rejected before reaching the handler.
- Runtime middleware order is now:
  `RequestId -> TrustedHost -> CORS -> RequireJsonContentType -> CSRF -> handler`.
  TrustedHost still fires first so host-spoofing 400s short-circuit; the 415
  check runs before CSRF so a missing CT is rejected before token validation.
