# Making fact-based orchestration the default

> **Status**: Implemented. Triage + direct-answer path landed and the default is
> switched. Remaining open items are in §7.
>
> **Date**: 2026-07 · **Scope**: `src/tether/protocol/orchestration/`,
> `src/tether/protocol/intent/`, `orchestrator.*` in `src/tether/config/default.yml`,
> `StreamRequest.mode` in `src/tether/app/http/routers/chat.py`

---

## 1. Goal

Make the **fact-based** orchestration (`NotebookOrchestrator`, ADR-0020, derived
from Steve Hanov's Ralph-Loop / Graph-Reader article) work smoothly and become
the **default** mechanism, because Tether will primarily run **small models**
(currently Qwen3-1.7B on GenieX/NPU).

The motivation is sound and worth stating precisely, because it is the thing
actually worth preserving:

> A chatty agent that issues five `web_search` calls accumulates **~6 k tokens**
> of raw tool-result text in its history. The Notebook orchestrator at the same
> point sees **~1.6 k tokens** (40-fact notebook + one fresh snippet batch),
> because each LLM call sees only `(question, notebook, current tool result)` and
> the raw search text is **thrown away after extraction**.
> — [ADR-0020](../adr/0020-notebook-orchestrator-algorithm.md)

Bounded per-call context is exactly what a small model needs. That principle is
the real prize — not the web-research loop that currently carries it.

---

## 2. What works today (measured, on-device)

Tested live against GenieX + Qwen3-1.7B on the NPU:

- **The Plan phase performs well on a 1.7 B model.** From *"Who won the 2022 FIFA
  World Cup?"* it produced five sensible sub-queries (`2022 FIFA World Cup
  winner`, `… results`, `… final match`, …). This is strong evidence the
  decompose-then-extract approach genuinely suits small models.
- **The turn lifecycle is robust.** Phase events, heartbeats
  (`notebook_phase_progress` every ~2 s), cancellation checks, `notebook_no_facts`,
  and a clean `message_stop(complete)` all behave correctly even in total failure.
- **Prompt-injection posture is good.** The orchestrator deliberately bypasses
  `SlidingParser`, so tool markers inside search snippets are never executed.

## 3. What blocks "make it the default"

### 3.1 It is a *web-research* loop, not a general conversation loop

`NotebookOrchestrator.run()` unconditionally does Plan → Explore(`web_search`) →
Extract → Refine → Synthesize. There is no path for a turn that needs no
research. Consequences if it became the literal default:

| Prompt | What would happen |
|---|---|
| "hello" | `unsearchable_input` clarification, or 5 web searches for "hello" |
| "write me a poem" | plans web queries about poems |
| "what did I just say?" | searches the web instead of reading history |
| "what is 25 + 50" | ✅ handled — local `Decimal` shortcut already exists |

The only non-research escape hatches today are the arithmetic shortcut
(`source_kind: local_deterministic`) and the `unsearchable_input` clarification,
which asks the user to rephrase rather than just answering.

### 3.2 It hard-depends on a search backend that is not configured

`web_search` is Brave-only and `BRAVE_API_KEY` is unset on this machine, so
**every** research turn currently gathers zero facts. Making research the default
would make the product unusable out of the box for anyone without a Brave key.

### 3.3 Boot-time coupling

The engine **fails loud at boot** if `research` is registered while `web_search`
is not in `tools.enabled` (ADR-0020 §D6). Any default-switch work must keep that
invariant coherent — and it interacts directly with the tool enable/disable work
queued next.

---

## 4. Implemented now: explain an empty notebook

**Problem found by testing.** When every search failed, the client saw five
`explore` phases flash past in ~6 ms, then `notebook_no_facts` with
`note: null`, then a synthesis that said it lacked evidence. The failures were
logged server-side (`notebook.explore_tool_error`) but **never surfaced on the
wire**. "The web had no answer" and "you have not configured a search key" were
indistinguishable.

**Fix.** Explore failures are now collected and summarised into the existing
(previously always-null) `NotebookNoFacts.note` field. Live output today:

```json
{"type":"notebook_no_facts","queries_attempted":4,"iterations":4,
 "note":"every search failed — web_search not initialised — BRAVE_API_KEY is not
 configured. Set the env var or write the value to <data_dir>/secrets/BRAVE_API_KEY
 and restart."}
```

Additive and non-breaking: the field already existed on the wire, the text is
drawn from the tool's structured `error` (never raw snippets), and it is capped
to the field's 256-char limit. Two unit tests that pinned `note is None` were
updated — their original intent ("don't say *empty plan* when queries were
attempted") is preserved and now asserted more precisely.

---

## 5. What was built

Option **B** was implemented: triage first, then flip the default. Option A alone
would have shipped a regression, and Option C remains a follow-up (§7).

### 5.1 The `TurnTriage` seam

`src/tether/protocol/intent/turn_triage.py` defines a `TurnTriage` ABC returning
`TurnKind.DIRECT` or `TurnKind.RESEARCH`, mirroring the `ConfirmIntentClassifier`
seam (ADR-0019). Two implementations ship:

- `RulesTurnTriage` (`rules_turn_triage.py`) — pure string work, microseconds, no
  model call.
- `AlwaysResearchTriage` — the pre-triage behaviour, used by explicit research.

**Biased toward `DIRECT`.** The failure directions are not symmetric: wrongly
routing a research question to `DIRECT` merely reproduces the old chat-mode
behaviour, whereas wrongly routing "hello" to `RESEARCH` costs several web
searches, seconds of latency, and an error when no backend is configured. A turn
therefore becomes `RESEARCH` only on a *positive* signal (evidence markers like
"latest"/"current"/"price"/"who won", or an interrogative plus a proper noun or
year). Back-references are checked **before** evidence markers, so "what did you
say about the latest release?" reads history instead of searching.

### 5.2 `AutoOrchestrator` — the new default

`AutoOrchestrator` subclasses `NotebookOrchestrator` and supplies
`RulesTurnTriage`. On a `DIRECT` turn it skips Plan/Explore/Extract entirely and
streams `_direct_answer_stream()`, which — unlike synthesis — includes prior
session history so back-references resolve. Like synthesis it bypasses
`SlidingParser`, so a stray `<<function_call>>` marker stays inert (ADR-0020 §D1).

Its constructor is spelled out rather than `**kwargs`-forwarded, because `Engine`
threads constructor arguments via `inspect.signature` (ADR-0020 §D5) — a
`**kwargs`-only constructor advertises no parameters and gets called with none.

The registry now reads:

| Mode | Orchestrator | Behaviour |
|---|---|---|
| `auto` **(default)** | `AutoOrchestrator` | Triage per turn: answer directly or research |
| `chat` | `ChattyAgentOrchestrator` | Legacy tool loop; explicit opt-out |
| `research` | `NotebookOrchestrator` | Always research; never downgraded by triage |

### 5.3 `orchestrator.default` was inert — fixed

`StreamRequest.mode` was `Literal["chat","research"] = "chat"`, so **every HTTP
request hard-coded chat mode** and the configured `orchestrator.default` was
silently ignored. `mode` is now `Optional[...] = None`, and the route resolves an
`effective_mode` once from the engine's configured default, threading it through
provider resolution, reasoning-effort validation, orchestrator lookup, and the
engine call.

### 5.4 Measured result (GenieX + Qwen3-1.7B, no search backend)

| Prompt | Phases | Behaviour |
|---|---|---|
| `hello` | 0 | direct answer |
| `Write a haiku about the sea` | 0 | direct answer |
| `What is the latest version of Python?` | 5 | full research loop |
| `My favourite colour is teal.` → `What did I just say...?` | 0 | *"You said your favorite color is teal."* |

## 6. Decision record for the three options considered

### Option A — Flip `orchestrator.default` to `research`
*Rejected alone.* Every turn becomes a web-research turn; requires a search key;
breaks "hello".

### Option B — Triage, then flip (**implemented**)
Cheap first decision routes conversational/creative/self-referential turns to a
direct answer and only evidence-seeking turns to the research loop. Degrades
correctly with no search backend: conversation works, and research turns report
the missing key (via §4).

### Option C — Port fact-bounding into the chat path (**follow-up**)
Make tool results become compact facts instead of raw JSON in history. Captures
the small-model context benefit for *all* tool use, not just web research, and
needs no search backend.

---

## 7. Remaining work

| Step | Work | Notes |
|---|---|---|
| 1 | Configure a search backend, or add a non-Brave option | Research turns still gather zero facts without one |
| 2 | Option C: fact-extraction for general tool results | Generalises the small-model benefit beyond web research |
| 3 | Tune the triage keyword sets against real usage | Rules are deliberately conservative; add signals as gaps appear |
| 4 | Consider an LLM triage fallback for ambiguous turns | Only if rules prove insufficient — it taxes every turn |

**Instrumentation note.** Per the telemetry plan, `async_span` should land
**after** this work so the new orchestrator is instrumented once in its final
shape rather than twice.

## 8. Open questions

1. **Is a search backend going to be configured?** Without one, genuine research
   turns cannot produce facts. Brave key, or an alternative provider?
2. **How aggressive should triage be?** The current rules are conservative and
   favour direct answers. If research questions are being answered from
   parametric memory too often, widen `_EVIDENCE_MARKERS`.
