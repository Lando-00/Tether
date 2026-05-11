# ADR-0018: WhatsApp connector — neonize backend with `WhatsAppClientAdapter` seam

- **Status**: Proposed (will move to Accepted after `wa-REV-RECONCILE`)
- **Date**: 2026-05-11
- **Wave 0 inputs**: `wa-D-synthesis.md` · `wa-D-neonize.md` · `wa-D-openclaw.md`

## Status

Proposed (will move to Accepted after `wa-REV-RECONCILE`).

## Context

Phase 2b of the refactor plan adds a single-user WhatsApp connector to Tether,
implementing the locked Connectors-Spec §4 contract (`begin_login` / `complete_login`,
`start` / `stop`, `inbound_stream`, plus the nine locked `whatsapp_*` tools listed in
spec §4). The connector must fit ADR-0005's `ConnectorRegistry` contract, ADR-0009's
`SqliteInbox` drain task, and ADR-0015's outbound-send doctrine (draft + confirm).
Nothing in the spec dictates the underlying WhatsApp Web library; that choice is the
job of this ADR.

The Python landscape for personal-account WhatsApp Web has exactly one viable backing
implementation today: [`neonize`](https://github.com/krypton-byte/neonize), a Python
binding over the Go `whatsmeow` library. Alternatives are not viable for this
contract — `yowsup` is unmaintained, `whatsmeow` has no pure-Python port, and `pywa`
targets the Cloud API (Meta Business account required, the spec explicitly chooses
WhatsApp Web for single-user use). OpenClaw's `extensions/whatsapp/` is the closest
prior art but is a TypeScript implementation over Node's Baileys; its design patterns
transfer, its runtime does not.

An adapter seam is non-negotiable. The Wave 0 neonize audit (`wa-D-neonize.md`
BLOCKERS §1, §2) surfaced **two upstream-level instability points** that must not be
exposed to the rest of Tether: a hardware-arch-vs-process-arch DLL mis-selection
(`neonize/utils/platform.py:36`) that breaks `import neonize` on Snapdragon X Elite,
and a module-global asyncio loop singleton (`neonize/aioze/events.py:52-60`) that
silently drops events on restart. Both are fixable at the adapter layer, but only if
there is one. A third concern — neonize/whatsmeow exposes no arbitrary history fetch
— forces the `whatsapp_get_thread` tool to read from `SqliteInbox` rather than the
client. The combined effect: the rest of Tether must talk to an interface, not to
`neonize` directly. ADR-0018 ratifies that interface.

## Decision

### D1. Library: pin `neonize==0.3.17.post0`

- Pin **exactly** (`==`, not `~=` or `>=`). The 0.3.x series is the production-stable
  line; `.post0` releases are DLL-only bumps that ship a new pre-compiled `goneonize`
  binary without Python API changes. Adding the dependency to a `[whatsapp]` extra in
  `pyproject.toml` keeps the base install lean.
- Use **`neonize.aioze` exclusively**, never `neonize.client`. Every Go call in the
  async path is wrapped in `asyncio.to_thread` upstream
  (`neonize/aioze/client.py:210-229`), so the adapter never blocks the event loop.
- Two upstream defects are patched **at the adapter boundary**, not by forking neonize:
  1. **Platform-machine patch** (`wa-D-neonize.md` BLOCKER-1). Monkey-patch
     `platform.machine` to return `"AMD64"` before the first `import neonize` on
     Windows when hardware arch reports `ARM64`. The `win_amd64` wheel already
     bundles both DLLs side-by-side; only the selection logic at
     `neonize/utils/platform.py:36` is broken. Patch lives in
     `src/tether/connectors/whatsapp/__init__.py` so it fires regardless of import
     order.
  2. **Event-loop reset on stop** (`wa-D-neonize.md` BLOCKER-2).
     `NeonizeWhatsAppClientAdapter.stop()` sets
     `neonize.aioze.events.event_global_loop = None` before returning, so a
     subsequent `start()` can re-register against the new loop.

### D2. Adapter pattern: `WhatsAppClientAdapter` ABC

- The seam lives at `src/tether/connectors/whatsapp/adapter.py`. The concrete
  neonize-backed implementation is `NeonizeWhatsAppClientAdapter` in
  `neonize_adapter.py` (sibling module).
- The method surface mirrors what `WhatsAppConnector` needs — **not** what neonize
  exposes. Per `wa-D-synthesis.md` §D9 the contract is:

  | Method | Shape | Notes |
  |---|---|---|
  | `start(auth_dir)` | async I/O | Opens the neonize session and supervises the long-lived connect task. |
  | `stop()` | async I/O | Bounded by the connector stop budget. |
  | `logout()` | async I/O | Drops the linked-device session and deletes local creds. |
  | `pair_qr()` | async I/O | Waits for the first QR payload. |
  | `await_paired(timeout_sec)` | async I/O | Returns `PairStatus.PAIRED`, `QR_ROTATED`, `LOGGED_OUT`, `FAILED`, or `TIMEOUT`. |
  | `send_text(...)` | async I/O | Sends plain text. |
  | `send_media(...)` | async I/O | Sends adapter-ready bytes; tool-layer input is a file path. |
  | `send_read_receipt(...)` | async I/O | Backing call for the `whatsapp_mark_platform_read` tool. |
  | `get_contacts()` | async I/O | Returns `Contact` dataclasses. |
  | `fetch_history_sync()` | async I/O | Explicit hook for the initial historical drain. |
  | `subscribe_inbound()` | sync factory | Returns an `AsyncIterator[InboundEvent]`. |
  | `health()` | sync snapshot | Returns a cheap cached `AdapterHealthSnapshot`. |

  All I/O methods are async; `subscribe_inbound` returns an `AsyncIterator` and
  `health` returns a cheap snapshot.
- **No neonize types leak through the contract.** JIDs are `str`, message IDs are
  `str`, contacts are `Contact` dataclasses, inbound events are Tether's
  `InboundEvent`. The
  mapping from protobuf types (`Contact`, `SendResponse`, `MessageEv`,
  `ReceiptType.READ`) to Python primitives happens at the adapter boundary; see
  `wa-D-neonize.md` §11 for the full table.
- A future `BaileysSidecarWhatsAppClientAdapter` (Node subprocess + IPC channel, the
  Plan-§4 fallback) becomes a **single-class swap**: `WhatsAppConnector`, the tools,
  the drain task, and every test fixture are unchanged.
- `MockWhatsAppClientAdapter` satisfies the same ABC for tests and ships with **zero
  neonize imports** (`wa-D-neonize.md` §11 verdict).

### D3. `fetch_thread` is served from `SqliteInbox`, not a live client call

- WhatsApp Web (whatsmeow) exposes **no arbitrary history fetch**
  (`wa-D-neonize.md` §1, fetch_thread row). Historical messages arrive exactly once,
  via `HistorySyncEv` (event code 13, `neonize/events.py:59-101`) during initial
  pair, and via live `MessageEv` (code 17) thereafter.
- The adapter's `subscribe_inbound()` yields **both** `MessageEv` and
  `HistorySyncEv` payloads through the same `InboundEvent` mapper. The connector's
  drain task (per ADR-0009) appends them to `SqliteInbox` via `append_many`.
- `whatsapp_get_thread(peer, limit)` queries
  `SqliteInbox.list_recent(connector_id="whatsapp", limit=...)` and filters the
  result by peer JID in Python. Index optimisation (`idx_inbound_events_peer`) is a
  documented v1 follow-up (`fu-wa-peer-index`); single-user volume (<10k rows)
  makes the un-indexed scan acceptable.
- Injection seam: `WhatsAppConnector.__init__` accepts `inbox: InboundInbox | None`.
  `ConnectorRegistry` injects its shared inbox handle after connector validation by
  assigning `_inbox`, parallel to the registry-owned drain-task wiring. Tools that
  need the archive read this connector-owned handle; they do not reach through the
  Engine or HTTP app state.
- Tool docstring **states the limitation explicitly**: *"Returns messages Tether
  has seen since first connect. Earlier history is unavailable."* The model needs
  to see this in its tool schema to avoid promising the user something neonize
  cannot deliver.

Locked tool surface:

| Tool | v1 contract |
|---|---|
| `whatsapp_prepare_send` | Creates a pending text-send draft and returns `draft_id`. |
| `whatsapp_confirm_send` | Sends only when `user_confirmed_send` is true and `draft_id` exists. |
| `whatsapp_list_unread` | Lists unread WhatsApp inbox events. |
| `whatsapp_get_thread` | Reads messages Tether has seen from `SqliteInbox`. |
| `whatsapp_inbox_mark_seen` | Marks Tether inbox events seen. |
| `whatsapp_mark_platform_read` | Marks messages read on WhatsApp itself; this is a visible blue-check side effect. |
| `whatsapp_send_media` | Takes a file path string; after size + MIME validation the connector reads bytes for the adapter. |
| `whatsapp_get_contacts` | Takes `query: str` and `limit: int = 20`; returns contacts whose name or E.164 contains `query` case-insensitively. Empty query is rejected. |
| `whatsapp_resolve_contact` | Resolves an exact display name, E.164, or JID to a WhatsApp JID. |

### D4. State mapping and QR login payload

`WhatsAppConnector` maps adapter events and pair outcomes onto
`ConnectorState` as follows:

| Trigger / outcome | Target |
|---|---|
| No creds on disk | `UNCONFIGURED` |
| `begin_login()` called, QR pending | `AUTHENTICATING` |
| Creds on disk, `start()` returned, `ConnectedEv` not yet seen | `DEGRADED` with `detail="connecting_on_resume"` |
| neonize `ConnectedEv` after pair or resume | `READY` |
| `KeepAliveTimeoutEv` / `DisconnectedEv` / `ConnectFailureEv` | `DEGRADED` (Go auto-reconnects) |
| `KeepAliveRestoredEv` / `ConnectedEv` after `DEGRADED` | `READY` |
| `LoggedOutEv` | `LOGGED_OUT` |
| `StreamReplacedEv` | `ERROR` with `detail="session_conflict"` |
| `TemporaryBanEv` / `ClientOutdatedEv` | `ERROR` |
| `logout()` called explicitly | `LOGGED_OUT` |

Pair-poll outcomes map to `LoginContinueResult` this way: `PairStatus.PAIRED` →
`state=READY`; `QR_ROTATED` → `state=AUTHENTICATING` with a refreshed
`next_prompt`; `LOGGED_OUT` → `state=LOGGED_OUT,
detail="logged_out_during_pair"`; `FAILED` → `state=ERROR` with adapter detail;
`TIMEOUT` → `state=AUTHENTICATING, detail="qr_scan_timeout"`.

`begin_login()` returns `LoginPrompt(kind="qr_code", payload=qr_raw_string,
extra={"png_b64": ...})`: `payload` is the raw QR text to encode, while the PNG is
optional rendering metadata for clients that cannot render QR text natively.

## Consequences

### Positive

- **Upstream-instability containment.** The two known neonize blockers (and any
  future API drift in the 0.3.x line) are absorbed by exactly one file
  (`neonize_adapter.py`). Nothing else in Tether imports `neonize` directly.
- **Testability with zero neonize dependency.** A `MockWhatsAppClientAdapter` at
  `tests/fixtures/whatsapp/mock_adapter.py` satisfies the same ABC. All nine tool
  unit tests, the draft/confirm flow, and the inbound-stream integration test
  run without a live WhatsApp account or the neonize wheel installed.
- **Future-proof library swap.** If neonize regresses or is abandoned (the
  Connectors Spec §4 footnote already names the contingency — "ship a thin Python
  wrapper around the Node Baileys process"), the swap is one new class. The
  alternative — direct neonize use throughout the tools and connector — would
  require rewriting every consumer.
- **Lean default install.** `pip install tether` does not pull neonize; users
  install `tether[whatsapp]` to opt in. The dependency adds `phonenumbers` for
  JID/E.164 normalisation (per `wa-D-synthesis.md` §D5) but no other transitive
  Python-level adds.

### Negative

- **One more abstraction layer to maintain.** The mapping from neonize protobuf
  types to plain Python dicts must be kept in sync with neonize releases. Mitigated
  by:
  - the exact version pin (`==0.3.17.post0`),
  - adapter unit tests that exercise the mapping with frozen protobuf-shaped
    fixtures (`tests/fixtures/whatsapp/neonize_events.py`),
  - the `wa-REV-RECONCILE` reviewer pass before bumps to 0.3.18 or 0.4.x.
- **Two-blocker workaround code lives in our tree forever-ish.** Even if neonize
  upstream fixes BLOCKER-1 and BLOCKER-2, the workarounds are conditional no-ops
  on a fixed neonize (BLOCKER-1 is gated on the hardware-vs-process arch
  mismatch; BLOCKER-2 just sets a global that the fixed code would also reset).
  Net cost: ~20 lines, well-documented.
- **Inbox-as-archive is a soft expansion of `SqliteInbox`'s mandate.** Originally
  the inbox stored unread events; with this ADR it also serves as the local
  message archive for thread queries. Schema (migration 004) is sufficient; only
  the tool-layer queries change. Documented in ADR-0009's implementation status.

### Operational

- **Session DB path.** `data/connectors/whatsapp/auth/neonize.db` is managed by
  whatsmeow's Go runtime (`wa-D-neonize.md` §3). The adapter's `start()` creates
  the directory (`Path(...).mkdir(parents=True, exist_ok=True)`) before the first
  `client.connect()`. `logout()` attempts `client.logout()` and then deletes the
  directory unconditionally, even if stop/logout failed; the stop budget does not
  apply to local credential deletion.
- **`auth_status()` is cheap pre-start.** Before the adapter is started,
  `WhatsAppConnector.auth_status()` checks credential-file existence on disk only
  and MUST NOT import neonize. After start, it reads the cached
  `self._adapter.health()` snapshot.
- **`stop()` budget.** The connector's 2 s stop budget (per
  `src/tether/connectors/base.py:97-108` and ADR-0005) is enforced via
  `asyncio.wait_for(client.stop(), timeout=1.8)`
  (`neonize/aioze/client.py:3182-3186`). On timeout, the adapter applies the
  daemon-thread + force-exit pattern from
  `src/tether/providers/mlc/provider.py::shutdown_provider_with_timeout` — abandon
  the thread-pool worker, let the OS reap it on process exit. The Go runtime does
  not hold Python interpreter state, so an abandoned stop is safe
  (`wa-D-neonize.md` §8).
- **HistorySync flood.** A busy account fires hundreds of `HistorySyncEv` messages
  on first pair. The adapter's queue uses `asyncio.Queue(maxsize=10000)`; if full,
  history-sync entries are dropped (live `MessageEv` arrives on a separate
  callback and has effective priority). Documented in the adapter docstring;
  spec §3.4 `max_backfill_events` is the eventual knob (deferred).
- **`connect()` is not idempotent** (`neonize/aioze/client.py:3267-3310`, anti-pattern
  #3 in `wa-D-neonize.md` §12). The adapter guards `start()` with an `_started`
  flag; the registry's idempotent-start contract is preserved.
- **Structured adapter logs.** Every adapter log line includes
  `connector_id="whatsapp"` and `adapter="neonize"`, plus `chat_jid` when a chat is
  in scope, `message_id` when a message is in scope, `event_kind` for `MessageEv` /
  `HistorySyncEv` mappings, and the standard structlog context
  (`correlation_id`, error fields, etc.) from ADR-0010.

## Alternatives considered

1. **Direct neonize use, no adapter ABC.** Rejected. The two BLOCKER fixes in
   `wa-D-neonize.md` are exactly the kind of upstream instability that an adapter
   should absorb. We have already paid the design cost of identifying them; the
   engineering cost is one ABC file plus a mock impl. The savings of skipping the
   ABC (one fewer class, one fewer indirection in stack traces) do not justify
   coupling every tool and the drain task to a third-party Go-backed library that
   has demonstrably shipped a 0.3.x release with a Windows-on-ARM platform bug.

2. **Node Baileys sidecar process from day 1.** Rejected for v1. Substantially
   more engineering (IPC channel design, process supervision, separate language
   runtime, lifecycle inversion) for v1 functional parity with neonize. OpenClaw
   has demonstrated that Baileys-on-Node works, but their `connection-controller.ts:250`
   and `login-qr.ts:288` patterns transfer to neonize-on-Python without needing
   the Node runtime. ADR-0018's adapter seam keeps the door open: if neonize
   becomes untenable, the sidecar lands as `BaileysSidecarWhatsAppClientAdapter`
   without disturbing the rest of the connector.

3. **`yowsup` / pure-Python `whatsmeow` port / `pywa` (Cloud API).** Rejected.
   `yowsup` is unmaintained (last meaningful release 2020) and uses the obsolete
   pre-2021 WhatsApp protocol. No pure-Python port of `whatsmeow` exists; the
   protocol is large and changes are upstream-driven. `pywa` targets the WhatsApp
   Business Cloud API, which requires a Meta Business account and is not the
   single-user personal-account experience the locked spec describes. None of
   these satisfies the Connector Spec §4 contract on day one.

4. **Skip `[whatsapp]` extra; require neonize for all users.** Rejected. The
   neonize wheel is ~30 MB (bundled Go DLLs) and pulls a `goneonize` download at
   first run if the bundled DLL is missing (`neonize/_binder.py:46-47`). Users
   who never enable the WhatsApp connector should not pay that cost. The extra
   also makes the WhatsApp blockers' blast radius explicit: only people who
   opted in to `[whatsapp]` can hit BLOCKER-1's `WinError 193`.

## References

### Wave 0 inputs

Wave 0 produced three discovery reports (neonize audit, OpenClaw pattern
extraction, intent classifier failure-mode catalogue) feeding a synthesis
digest. The canonical record in the repo is
[`../refactor/synthesis-2026-05.md`](../refactor/synthesis-2026-05.md);
specific decisions cited above (library pin, fetch_thread routing, adapter
contract) are summarised in its §D1, §D2, §D9 respectively. The neonize
internals and OpenClaw patterns below were extracted from those reports
and are the authoritative external-source citations for this ADR.

### neonize internals cited (audited at 0.3.17.post0)

- `neonize/utils/platform.py:36` — `generated_name()` DLL selection (BLOCKER-1).
- `neonize/aioze/events.py:52-60` — `event_global_loop` module global (BLOCKER-2).
- `neonize/aioze/client.py:210-229` — `GoCode` async wrapper (`asyncio.to_thread`).
- `neonize/aioze/client.py:3267-3310` — `connect()` non-idempotent Task creation.
- `neonize/aioze/client.py:3312-3316` — `disconnect()`.
- `neonize/aioze/client.py:3182-3186` — `stop()`.
- `neonize/aioze/client.py:591-651` — `send_message()`.
- `neonize/aioze/client.py:2262-2298` — `mark_read()`.
- `neonize/aioze/client.py:321-336` — `ContactStore.get_all_contacts()`.
- `neonize/aioze/client.py:3319-3408` — `ClientFactory`.
- `neonize/events.py:59-101` — `EVENT_TO_INT` (including code 13 `HistorySyncEv`,
  code 17 `MessageEv`).
- `neonize/_binder.py:46-47` — DLL download fallback.

### OpenClaw patterns adopted (TypeScript, Baileys-backed)

- `extensions/whatsapp/src/connection-controller.ts:250` — state-machine entry point.
- `extensions/whatsapp/src/login-qr.ts:288` — `startWebLoginWithQr` (Phase 1 QR).
- `extensions/whatsapp/src/login-qr.ts:448` — `waitForWebLogin` (Phase 2 long-poll).
- `extensions/whatsapp/src/login-qr.ts:48` — `ACTIVE_LOGIN_TTL_MS = 3 * 60_000`.
- `extensions/whatsapp/src/reconnect.ts:8` — `DEFAULT_RECONNECT_POLICY` defaults.
- `extensions/whatsapp/src/normalize-target.ts:676` — `normalizeWhatsAppTarget`.

### OpenClaw patterns rejected (multi-channel framework concerns)

- `extensions/whatsapp/src/outbound-base.ts:391` — `createWhatsAppOutboundBase`
  channel-adapter abstraction (multi-channel routing).
- `extensions/whatsapp/src/inbound-policy.ts:22` — `dmAllowFrom` / `groupAllowFrom`
  (multi-user gating).
- `extensions/whatsapp/src/group-session-key.ts:14` — per-account group session
  scoping.

### Tether internals referenced

- Connector spec §4 (locked single-user outbound-send + inbound-read
  doctrine): encoded in this repo as ADR-0015. The doctrine drives the
  draft+confirm pattern used by the WhatsApp tools.
- Plan §13 (Phase 2b WhatsApp work) — the implementation plan that this
  ADR records; see [`../refactor/synthesis-2026-05.md`](../refactor/synthesis-2026-05.md)
  for the consolidated decision record.
- `src/tether/connectors/base.py:97-108` — `stop()` 2 s budget contract.
- `src/tether/connectors/types.py` — `LoginPrompt(kind="qr_code", payload=...)`.
- `src/tether/core/connector_registry.py` — drain-task wiring, 2 s stop budget.
- `src/tether/providers/mlc/provider.py::shutdown_provider_with_timeout` —
  daemon-thread + force-exit pattern (mirrored in adapter `stop()`).
- `src/tether/context/inbox_store.py` — `InboundInbox` ABC + `SqliteInbox` impl
  (the back-end for `whatsapp_get_thread`).

### Related ADRs

- ADR-0005 — `ConnectorRegistry` + mandatory `{connector_id}_` prefix (the
  contract this connector must satisfy).
- ADR-0009 — `SqliteInbox` in shared `data/tether.db` (the back-end for
  `fetch_thread`).
- ADR-0015 — Single-user outbound-send doctrine (mandates the draft + confirm
  pattern that `whatsapp_prepare_send` / `whatsapp_confirm_send` implement).
- ADR-0003 — GC-disabled daemon-thread shutdown (the force-exit pattern reused
  for `stop()`).

### External

- neonize on PyPI: <https://pypi.org/project/neonize/0.3.17.post0/>
- neonize source: <https://github.com/krypton-byte/neonize>
- whatsmeow (Go backend): <https://github.com/tulir/whatsmeow>
- OpenClaw whatsapp extension:
  <https://github.com/openclaw/openclaw/tree/main/extensions/whatsapp>
