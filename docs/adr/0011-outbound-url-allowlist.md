# ADR-0011: Outbound URL allowlist + `assert_safe_url` for SSRF defense

- **Status**: Accepted (Phase 7 of refactor)
- **Date**: 2026-05 (Phase 7)
- **Synthesis digest**: [`../refactor/synthesis-2026-05.md`](../refactor/synthesis-2026-05.md)

## Context

Tools that issue outbound HTTP (today: `web_search` via Brave; tomorrow: arbitrary
URL-accepting tools, OAuth callbacks, `httpx`-based connectors) can be coerced via prompt
injection or malicious tool inputs into hitting metadata services (e.g. `169.254.169.254`,
`fd00:ec2::254`), localhost services on non-public ports, or RFC 1918 ranges. A single-user
local-first app on a developer laptop is a particularly attractive SSRF target because it
trivially has access to other localhost services. There is no universally-correct
allowlist, but there must be a configured seam to enforce one when needed and a helper to
make the check trivial to call.

## Decision

Adopt a **per-tool outbound `httpx` allowlist** plus an `assert_safe_url()` helper, both
configurable through Settings:

- **`assert_safe_url(url, *, allowed_hosts, deny_private_ranges=True)`** in
  `src/tether/core/url_safety.py`. Resolves the host, validates the resulting IP is NOT in
  loopback (`127.0.0.0/8`, `::1`), link-local (`169.254/16`, `fe80::/10`), private
  (`10/8`, `172.16/12`, `192.168/16`, `fc00::/7`), or unspecified (`0.0.0.0`, `::`)
  ranges by default. Optionally checks the host against an allowlist. Raises
  `UnsafeUrlError` on violation.
- **Default permissive, seam present**: `Settings.security.outbound_allowlist.enabled =
  False` default. When enabled, each tool that performs outbound HTTP must declare its
  allowlist in tool config (e.g. `tools.web_search.allowed_hosts: ["api.search.brave.com"]`).
  The `BaseTool` base provides a `self._make_http_client()` helper that wraps `httpx`
  with the allowlist.
- **Documented in the tool-author guide**: any tool that accepts a URL as a parameter
  (or interpolates one from model output) MUST call `assert_safe_url(url, ...)` before
  the request. The guide includes an example.
- **Wired through bundled tools**: `WebSearchTool` (Brave) calls `assert_safe_url` on its
  base URL. Future URL-accepting tools (e.g. `fetch_url`) inherit the helper.
- **Connector tools**: when a connector ships an HTTP-fetching tool, it uses the same
  helper and declares an allowlist scoped to its API host.
- **Independent of CSRF/CORS** (ADR-0012): outbound URL safety protects the user FROM
  the model; CSRF/CORS protects the localhost API from external attackers. They cover
  different threat models.

## Consequences

### Positive
- One helper, one rule — every tool author has a clear way to be safe.
- Default permissive avoids breaking existing single-user setups; opt-in lets paranoid
  users (or future per-connector-permissions UI) clamp down.
- The metadata-service exfiltration vector (a known SSRF favorite) is closed by the
  default-private-ranges deny.

### Negative
- Tool authors must remember to call `assert_safe_url` for URL-accepting tools — there is
  no automatic enforcement at the `httpx` layer. Mitigated by the helper API + a lint /
  documentation pass.
- DNS resolution at check-time may differ from request-time (TOCTOU). For higher-paranoia
  setups, a future enhancement is to re-check post-`getaddrinfo` and pin the IP for the
  request.

### Trade-offs accepted
- We accept TOCTOU risk in v1; the threat model (single-user local-first) makes it low
  enough to defer.

## Alternatives considered

- **`httpx` event hooks for global enforcement** — rejected for v1: hooks across all
  `httpx` clients in-process is invasive and connector clients with their own auth flows
  shouldn't share one global hook. Revisit if URL-accepting tools proliferate.
- **No allowlist, document don'ts** — rejected: prompt injection makes "remember to be
  safe" insufficient.
- **Network namespace / sandbox** — rejected: overkill for a desktop app; not
  cross-platform.

## References

- Synthesis digest: [`../refactor/synthesis-2026-05.md`](../refactor/synthesis-2026-05.md)
- `src/tether/core/url_safety.py` (`assert_safe_url`)
- `src/tether/tools/web_search_tool.py`, `src/tether/tools/brave_client.py`
- ADR-0012 (CSRF/CORS — complementary inbound-side defense)
