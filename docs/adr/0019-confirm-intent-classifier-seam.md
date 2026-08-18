# ADR-0019: `ConfirmIntentClassifier` ABC for draft+confirm send gates

- **Status**: Accepted
- **Date**: 2026-05 (Phase 2b, Wave 1)
- **Synthesis digest**: [`../refactor/synthesis-2026-05.md`](../refactor/synthesis-2026-05.md)
- **Wave 0 input**: synthesis digest at [`../refactor/synthesis-2026-05.md`](../refactor/synthesis-2026-05.md)

## Context

ADR-0015 locks Tether's outbound-send doctrine: every connector that can send
MUST implement a two-phase `*_prepare_send` / `*_confirm_send` pair, and
`*_confirm_send` MUST refuse unless the user has affirmed in their last turn.
The mechanism is `ToolExecutionContext.user_confirmed_send` — a boolean
threaded from `Orchestrator → ToolRunner → BaseTool.invoke`. The connector
spec §4 footer proposed a regex (`^(yes|send|go ahead|confirm|do it)\b`) to
flip this flag; `tests/fixtures/echo_connector.py::EchoConfirmSendTool` is
the reference consumer.

The plumbing is already in place. `ToolExecutionContext` is constructed in
`chatty.py` at the per-tool-call boundary, and `EchoConfirmSendTool` reads
the flag and refuses unless it is `True`. The deferred piece is the
classifier itself: `chatty.py:1041` currently hardcodes
`user_confirmed_send=False`, so every `*_confirm_send` tool refuses every
call. This is intentional for the refactor (synthesis §10.8 #4 deferred the
regex to ship with the first real send-capable connector), but it makes the
draft+confirm pattern unusable in practice — exactly the situation Phase 2b
is about to hit when `whatsapp_confirm_send` lands.

The minimal fix is to inline the spec footer regex at the assignment site.
We reject that for two reasons. First, the literal proposal misclassifies
the dangerous "yes-prefixed refusal" class — `yes but cancel`, `yes,
actually wait`, `confirm — no hold on` — as confirmation; the Wave 0 audit
(`wa-D-intent.md`) catalogues six such false positives. Hardening the regex
in place buries the test corpus inside `chatty.py`. Second, this v1 is
almost certainly not the final implementation: an LLM-based or multilingual
classifier is a foreseeable v2 (non-English affirmatives like `sí` / `oui` /
`はい` are accepted-but-flagged misses). Hardcoding the regex inside the
orchestrator makes that swap a cross-cutting change touching every consumer.
A swappable seam keeps the swap to one new class.

## Decision

Introduce a `ConfirmIntentClassifier` ABC plus a v1
`RegexConfirmIntentClassifier` implementation, configured via the same
dotted-path-import pattern Tether already uses for `providers.parser.impl`.

### D1. `ConfirmIntentClassifier` ABC

The contract lives at `src/tether/protocol/intent/classifier.py`:

```python
from abc import ABC, abstractmethod


class ConfirmIntentClassifier(ABC):
    """Binary classifier: did the user just say 'yes, send the draft'?

    Consumed by the orchestrator at the per-tool-call boundary to set
    ``ToolExecutionContext.user_confirmed_send``. The classifier is the
    gate on a destructive action (sending a WhatsApp message, dispatching
    a Gmail draft); default policy is **safe-hold** — when in doubt
    return ``False`` and let the user re-confirm. Impls MUST be pure
    (no side effects beyond logging) and reentrant.
    """

    @abstractmethod
    def classify(self, last_user_message: str | None) -> bool: ...
```

Notes:

- **ABC, not Protocol.** A future LLM-based classifier will need
  construction-time dependency injection (provider handle, prompt
  template, latency budget). The ABC's `__init__` is a contract surface
  we want — `Protocol` is too thin.
- **Single method.** No chat history, no locale, no draft text. A future
  classifier needing richer input introduces a v2 ABC; v1 consumers stay
  stable.
- **The argument is `last_user_message`**, not the full turn record.
  Whatever the orchestrator already has is what gets passed; the
  classifier does not reach into stores or context.
- **Return type is `bool`**, not a confidence score. The flag downstream
  is binary; encoding uncertainty here would be cargo-cult.

### D2. `RegexConfirmIntentClassifier` v1 impl

Lives at `src/tether/protocol/intent/regex_classifier.py`. The full
implementation is in Appendix A. Salient design points:

- **Normalization pipeline**: lowercase → strip → drop one optional
  salutation prefix (`tether,` / `@bot ` / `ai:` / `assistant:`) → strip
  leading wrapping punctuation → collapse internal whitespace → truncate
  to 2048 chars. The salutation peel ensures `tether, yes` and
  `@tether send it` are accepted; the punctuation peel ensures `"yes"`
  and `'yes,'` are accepted. The truncation is an O(n) DoS guard.
- **Affirmative vocabulary**: 33 canonical tokens grouped into nine
  families — yes (`yes`/`yeah`/`yep`/`yup`/`ya`/`yas`),
  sure-ok (`sure`/`sure thing`/`ok`/`okay`/`okey`/`okie`/`k`/`kk`),
  imperative-send (`send`/`send it`/`send them`/`send away`/`fire
  away`/`fire it off`/`ship it`), imperative-go (`go`/`go ahead`/`go for
  it`/`go on`/`proceed`/`continue`), confirm (`confirm`/`confirmed`/
  `confirming`/`do it`), correctness (`correct`/`right`/`that's
  right`/`that's correct`), approval (`looks good`/`looks fine`/`looks
  great`/`sounds good`/`lgtm`/`sgtm`/`approve`/`approved`), military
  (`affirmative`/`roger`/`roger that`/`copy`/`copy that`/`10-4`), and
  bare emoji (`👍`/`✅`/`👌`/`🆗`/`☑️`/`✔️`). Multi-word phrases bind
  longest-first inside the alternation so `go ahead` is not partially
  consumed by `go`.
- **Boundary correctness**: `^…(?=\W|$)` rejects `yesterday` and
  `confirmation bias` automatically while still accepting bare emoji. A
  trailing `\b` requires word-character adjacency and therefore fails for
  `\W`-category emoji such as `👍`. The `re.match` (not `re.search`) anchors at
  the start of the normalized string; embedded affirmatives (`please yes`, `the
  answer is yes`) return `False` by design (corpus case #10).
- **Key hardening: deny-list early-exit.** Before the affirmative match,
  the normalized message is scanned for negation / hesitation /
  modification substrings: `no`, `not`, `n't`, `wait`, `hold on`,
  `stop`, `cancel`, `abort`, `undo`, `redo`, `scrap`, `scrub`, `delete`,
  `remove`, `drop`, `discard`, `throw out`, `wrong`, `incorrect`, `typo`,
  `mistake`, `ignore`, `skip`, `revert`, `rollback`, `nvm`, `never mind`,
  `actually`, `but`, `instead`, `except`, `rephrase`, `reword`, `change`,
  `edit`, `modify`, `fix`, `tweak`, `in spanish`, `in french`, `in german`,
  `add`, `replace`, `first`, … The deny-list is substring-matched (with
  boundary padding for short tokens like `no`/`not`) — this is the
  defense against `yes, but change the wording`, `yes please cancel`,
  `confirm — no hold on`. **This is the single biggest deviation from
  the spec footer regex.**
- **Adversarial logging**: if the message matches an affirmative AND
  contains a digit-run of length ≥ 8, the classifier emits a structured
  `confirm_intent.digit_run_co_occurrence` warning. The classifier still
  returns `True`; the tool layer validates `draft_id` against its own
  pending-drafts store, so smuggled IDs simply don't exist. The log is
  ops-audit signal.
- **English-only, by design.** Non-English affirmatives return `False`
  and the user re-confirms. The ABC seam permits a future
  `LLMConfirmIntentClassifier` to drop in without touching consumers.

A 84-case test corpus (Appendix B) lives at
`tests/unit/protocol/intent/test_regex_classifier.py` and is the v1
acceptance set.

### D3. Configuration + wiring

Wave 1 (this PR) ships only the ABC plus `NullConfirmIntentClassifier` as the seam
shape. `IntentSettings` is added in Wave 2 (IMP-C) alongside
`RegexConfirmIntentClassifier`; that avoids a Wave-1 settings default pointing at a
module that has not shipped yet.

In Wave 2, the classifier is constructed once in `Engine.from_settings`, stored on
the engine, and passed into the chatty orchestrator via the same DI seam as
`provider`, `parser`, and `tool_runner` (classifier is a singleton, not a per-turn
factory; the DI seam is shared). The configuration key added in Wave 2 is:

```yaml
intent:
  classifier_impl: "tether.protocol.intent.regex_classifier.RegexConfirmIntentClassifier"
```

Loading rules:

- Dotted-path import, mirroring `providers.parser.impl` (no kwargs in v1;
  the impl's `__init__` takes no required args).
- **Default after Wave 2 IMP-C**: the regex impl. The classifier is on by default
  once `regex_classifier.py` exists.
- **Empty / unset**: fall back to a `NullConfirmIntentClassifier` whose
  `classify(...)` always returns `False`. This preserves the current refactor
  behaviour exactly — every `*_confirm_send` tool refuses every call. The null impl
  is the safe "feature off" switch.
- `ChattyAgentOrchestrator.__init__` gains a keyword-only
  `confirm_intent_classifier: ConfirmIntentClassifier` kwarg. `Engine` adds it to
  `_orch_kwargs` conditionally via `inspect.signature`, matching the existing
  `audit_store_args` pattern, so `NotebookOrchestrator` and other orchestrators that
  do not accept the kwarg are unaffected.
- The wiring site in `chatty.py` (currently lines 1037–1042) becomes:

  ```python
  tool_ctx = ToolExecutionContext(
      session_id=session_id,
      turn_id=turn_id,
      last_user_message=prompt,
      user_confirmed_send=self.confirm_intent_classifier.classify(prompt),
  )
  ```

  `self.confirm_intent_classifier` is set in the chatty orchestrator's
  `__init__` from the engine; the existing `prompt` variable (the user's
  last message text at this turn) is already the right input.

## Consequences

### Positive

- Unblocks `whatsapp_confirm_send` (Phase 2b) and any future
  `gmail_confirm_send` (Phase 3): the gate finally flips from `False` to
  the classifier's verdict.
- The 84-case test corpus gives high confidence in the regex's behaviour
  on the dangerous false-positive class — every `yes, but cancel`-style
  input has a row.
- An LLM-based classifier (v2) is a future ADR plus one new class behind
  the ABC. No consumer changes. `chatty.py`, `EchoConfirmSendTool`,
  WhatsApp/Gmail `confirm_send` tools all keep their current shape.
- The `NullConfirmIntentClassifier` fallback gives operators a one-line
  config switch after Wave 2 to disable the gate entirely (e.g., for an
  environment where `*_confirm_send` is wrapped by an external approval system).
  ADR-0015's doctrine is unchanged: turning off the classifier means the tool
  refuses every call, not that it sends without confirmation.

### Negative

- One new module (`tether.protocol.intent`) — small surface area, two
  files.
- The "what counts as 'yes'" definition is now a single source of truth
  in one place. A user-facing config override (e.g., the user disables
  the gate, accepts the risk) is a follow-up; Wave 2 ships with the classifier on
  and configurable only by impl-swap.
- Test maintenance: every new corpus row touches one parametrised test.
  The trade is intentional — the regex is the gate on a destructive
  action and deserves the heavy test coverage.

### Known v1 false negatives (accepted)

Per `wa-D-intent.md` §10. Each is one extra user round-trip; the cost is
asymmetrically lower than a false positive.

- **Any non-English affirmative**: `sí`, `oui`, `ja`, `はい`, `好的`,
  `да`, `evet`. Mitigation: future `LLMConfirmIntentClassifier`.
- **Affirmative not in vocabulary**: `green light`, `let's do this`,
  `pull the trigger`, `fire when ready`, `aye`, `aye aye`, `ack`,
  `acknowledged`. v1 chooses a curated list; expansion is cheap if a
  miss is observed.
- **Affirmative buried mid-message**: `please yes`, `the answer is yes`,
  `i'd say yes`. v1 chooses strict-anchor (`re.match`, not `re.search`);
  see corpus case #10 for the justification.
- **Affirmative with on-deny-list-but-benign tail**: `yes, send it
  first` (the user means "send this first, then the others"). The word
  `first` is on the deny-list and returns `False`. v1 accepts this miss
  as the price of catching `yes, please rephrase first`.
- **Notify-style imperatives** containing `not`: `notify John yes`
  contains the `not` substring and returns `False`. Acceptable: this
  isn't a confirmation in the first place.

### Known v1 false positives (accepted)

- **`yes` as a generic discourse marker**: if the user says `yes I
  agree with your analysis` while a draft is pending, the classifier
  returns `True` and the tool fires. The mitigation is layered: the LLM
  only calls `*_confirm_send` when its turn-level reasoning believes the
  user wants to send. The classifier is the second check, not the first.
- **`confirm 12345678901234` (draft-id smuggling)**: classifier returns
  `True` and emits the digit-run warning log. Mitigation: the tool
  validates `draft_id` against its pending-drafts store; smuggled IDs do
  not exist.
- **Sarcasm** (`oh yeah, sure, send it 🙄`): classifier returns `True`.
  Out of scope for v1; LLM classifier needed.

The asymmetry of costs — a false negative is one extra round-trip; a
false positive is a sent message — drove the safe-hold default. The
deny-list early-exit is what shifts the boundary toward safe-hold.

## Alternatives considered

1. **Regex inlined directly in `chatty.py`.** Rejected: makes the future
   LLM swap a cross-cutting change touching every consumer, buries the
   test corpus inside orchestrator tests, and reuses the spec footer's
   misclassification of yes-prefixed refusals.
2. **LLM-based classifier from day one.** Rejected for v1: tail latency
   (an extra forward pass per tool call), cost, complexity. The
   advantage — multilingual coverage, sarcasm detection, paraphrase
   handling — is real but not worth the dependency on a model call to
   gate every send. A future ADR can switch.
3. **Proof-of-context arg** (model echoes the draft text as a second
   argument to `*_confirm_send`). Rejected per spec §4 footer
   ("cleaner shape is the context") and ADR-0015. The classifier is
   the cleaner shape.
4. **`Protocol` instead of `ABC`.** Rejected: future impls
   (LLM-based) need `__init__` dependency injection. ABC permits a
   richer contract surface; `Protocol` is structural-typing only.
5. **`Settings.intent.regex_overrides`** (let the user extend the
   affirmative vocabulary). Rejected for v1: increases the failure
   surface (user types a regex with catastrophic backtracking; user
   accidentally accepts `n` as affirmative); follow-up if observed in
   practice.

## References

- Wave 0 inputs (intent failure-mode catalogue + reconciled v1 algorithm):
  see synthesis digest [`../refactor/synthesis-2026-05.md`](../refactor/synthesis-2026-05.md).
  Appendix A (the regex pattern) and Appendix B (the 84-case acceptance corpus)
  in this ADR are the authoritative copies — the synthesis is the design
  background, not a separate canonical source.
- Connector spec §4 footer (the original `^(yes|send|go ahead|confirm|do it)\b`
  proposal that this ADR hardens) — single-user outbound-send doctrine; encoded
  in this repo as ADR-0015.
- `src/tether/core/types.py` lines 70–100 — `ToolExecutionContext`
  dataclass; the `user_confirmed_send` docstring explicitly says the
  regex classifier ships with the WhatsApp/Gmail connectors.
- `src/tether/protocol/orchestration/chatty.py` lines 1037–1042 — the
  current `user_confirmed_send=False` wiring site that this ADR
  replaces.
- `tests/fixtures/echo_connector.py` lines 176–225 — the
  `EchoConfirmSendTool` reference consumer of the flag.
- ADR-0004: Tool v2 `BaseTool` + `ToolExecutionContext` — defines the
  context dataclass this classifier writes into.
- ADR-0015: Single-user outbound-send + inbound-read doctrine — this
  classifier is what makes the doctrine enforceable in practice.

---

## Appendix A: `RegexConfirmIntentClassifier` v1 reference implementation

Verbatim from the Wave 0 design synthesis (see [`../refactor/synthesis-2026-05.md`](../refactor/synthesis-2026-05.md)).
Lives at `src/tether/protocol/intent/regex_classifier.py`.

The trailing lookahead uses `(?=\W|$)` instead of `\b` because bare emoji are
non-word characters; `\b` has no word-character edge to match on a standalone emoji.

```python
from __future__ import annotations

import re
from typing import Final

from tether.core.logging import logger
from tether.protocol.intent.classifier import ConfirmIntentClassifier

_MAX_LEN: Final[int] = 2048

# 33-token affirmative vocabulary, ordered longest-first inside each
# alternation group so multi-word phrases bind before their prefixes.
_CONFIRM_RE: Final[re.Pattern[str]] = re.compile(
    r"^("
    # multi-word (must come first so they bind before single-word prefixes)
    r"that(?:'s| is) (?:right|correct)"
    r"|sounds good|looks (?:good|fine|great)"
    r"|go (?:ahead|for it|on)"
    r"|sure thing|yeah sure|yes please"
    r"|send (?:it|them|away)|fire (?:away|it off)|ship it"
    r"|do it|copy that|roger that|10-4"
    r"|never mind"            # present only to be SHADOWED by deny-list
    # single-word
    r"|yes|yeah|yep|yup|yas|ya"
    r"|sure|ok|okay|okey|okie|kk|k"
    r"|send|go|proceed|continue|confirm(?:ed|ing)?"
    r"|correct|right|approved?|lgtm|sgtm"
    r"|affirmative|roger|copy"
    # bare emoji (must be alone; anchored)
    r"|\U0001F44D|\u2705|\U0001F44C|\U0001F197|\u2611\ufe0f?|\u2714\ufe0f?"
    r")(?=\W|$)"
)

# Substring deny-list. Matched against the normalized message
# (case-folded, whitespace-collapsed). Order doesn't matter; any hit
# forces False.
_DENY_SUBSTRINGS: Final[tuple[str, ...]] = (
    " no", "no ", "nope", "nah", "naw", " not", "not ", "n't",
    "wait", "hold on", "stop", "cancel",
    "abort", "undo", "redo", "scrap", "scrub",
    "delete", "remove", "drop", "discard", "throw out",
    "wrong", "incorrect", "typo", "mistake",
    "ignore", "skip", "revert", "rollback",
    "don't", "do not", "never", "nvm", "never mind", "nevermind",
    "forget it", "let me think", "maybe", "actually", "hmm",
    "on second thought", "second thoughts",
    " but ", "but,", "instead", "except",
    "rephrase", "reword", "rewrite", "change",
    "edit", "modify", " fix", "tweak",
    "in spanish", "in french", "in german",
    " add ", " remove ", " replace ", " first",
)

_SALUTATION_RE: Final[re.Pattern[str]] = re.compile(
    r"^(tether|ai|bot|assistant|@\w+)[,:\s]+"
)
_LEADING_PUNCT_RE: Final[re.Pattern[str]] = re.compile(
    r"^[\"'`\(\[\s,\.!\?]+"
)
_WS_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_DIGIT_RUN_RE: Final[re.Pattern[str]] = re.compile(r"\d{8,}")


class RegexConfirmIntentClassifier(ConfirmIntentClassifier):
    """v1 regex impl of the confirm-intent gate.

    Connector spec §4 footer + wa-D-INTENT failure-mode catalogue.
    English-only, context-free, pure. Safe-default: returns False on
    any ambiguity. The ABC seam permits a future LLM-based impl
    without touching consumers (chatty.py orchestrator, connector
    confirm_send tools).
    """

    def classify(self, last_user_message: str | None) -> bool:
        if not last_user_message:
            return False
        s = last_user_message.strip().lower()
        if not s:
            return False
        if len(s) > _MAX_LEN:
            s = s[:_MAX_LEN]
        s = _SALUTATION_RE.sub("", s)
        s = _LEADING_PUNCT_RE.sub("", s)
        s = _WS_RE.sub(" ", s)
        if not s:
            return False

        # Pad with spaces so " no" / "no " substring tests behave
        # correctly at boundaries.
        padded = f" {s} "
        for token in _DENY_SUBSTRINGS:
            if token in padded:
                return False

        m = _CONFIRM_RE.match(s)
        if not m:
            return False

        if _DIGIT_RUN_RE.search(s):
            logger.warning(
                "confirm_intent.digit_run_co_occurrence",
                extra={"normalized_len": len(s)},
            )
        return True
```

---

## Appendix B: 84-case test corpus

Verbatim from `wa-D-intent.md` §8. This is the v1 acceptance set;
`tests/unit/protocol/intent/test_regex_classifier.py` parametrises over it.

| # | Input | Expected | Why |
|---|---|---|---|
| 1 | `yes` | True | Canonical, spec seed |
| 2 | `Yes` | True | Case-insensitive after normalization |
| 3 | `YES` | True | Case-insensitive |
| 4 | `  yes  ` | True | Whitespace stripped |
| 5 | `"yes"` | True | Wrapping punctuation stripped |
| 6 | `yes!` | True | Trailing punctuation does not block `\b` |
| 7 | `yes.` | True | Same |
| 8 | `yes please` | True | Affirmative + polite suffix, no deny tokens |
| 9 | `yes please send` | True | Same-action reinforcement |
| 10 | `please yes` | **False** | Not anchored at start; v1 chooses strict-anchor (justification: avoids accepting `please yes do not send` and similar embedded affirmatives) |
| 11 | `yesterday i said yes` | False | `^yes\b` fails on `yesterday` (`\b` between `s`/`t` is not a boundary); even if it slipped, deny-list catches nothing — but the message simply doesn't start with an affirmative |
| 12 | `yeah` | True | Yes-family |
| 13 | `yep` | True | Yes-family |
| 14 | `yup` | True | Yes-family |
| 15 | `ya` | True | Yes-family |
| 16 | `sure` | True | Sure-family |
| 17 | `sure thing` | True | Sure-family |
| 18 | `ok` | True | OK-family |
| 19 | `okay` | True | OK-family |
| 20 | `k` | True | Terse OK |
| 21 | `kk` | True | Terse OK |
| 22 | `send` | True | Spec seed |
| 23 | `send it` | True | Imperative send |
| 24 | `ship it` | True | Imperative send |
| 25 | `go` | True | Spec seed (extended) |
| 26 | `go ahead` | True | Spec seed |
| 27 | `go for it` | True | Imperative go |
| 28 | `proceed` | True | Imperative go |
| 29 | `confirm` | True | Spec seed |
| 30 | `confirmed` | True | Confirm-family |
| 31 | `confirmation bias` | False | `\b` between `m`/`a` fails — does not match `^confirm\b` |
| 32 | `correct` | True | Correctness |
| 33 | `that's right` | True | Correctness |
| 34 | `looks good` | True | Approval |
| 35 | `lgtm` | True | Approval |
| 36 | `approved` | True | Approval |
| 37 | `affirmative` | True | Military / radio |
| 38 | `roger that` | True | Military / radio |
| 39 | `do it` | True | Spec seed |
| 40 | `👍` | True | Bare emoji |
| 41 | `✅` | True | Bare emoji |
| 42 | `👌` | True | Bare emoji |
| 43 | `sure 👍` | True | Affirmative + emoji, no deny tokens |
| 44 | `no` | False | Refusal |
| 45 | `nope` | False | Refusal |
| 46 | `nah` | False | Refusal |
| 47 | `not yet` | False | Hesitation (`not` deny token) |
| 48 | `wait` | False | Hesitation |
| 49 | `hold on` | False | Hesitation |
| 50 | `stop` | False | Hesitation |
| 51 | `cancel` | False | Refusal |
| 52 | `nvm` | False | Refusal |
| 53 | `never mind` | False | Refusal |
| 54 | `actually` | False | `actually` deny token, no affirmative head anyway |
| 55 | `i guess` | False | Not in affirmative list |
| 56 | `whatever` | False | Not in affirmative list |
| 57 | `if you say so` | False | Not in affirmative list |
| 58 | `yes, but change the wording` | False | `but` + `change` deny tokens — KEY false-positive defense |
| 59 | `yes, in Spanish though` | False | `in spanish` deny token |
| 60 | `yes please rephrase first` | False | `rephrase` + `first` deny tokens |
| 61 | `yes I want you to NOT send` | False | `not` deny token — KEY false-positive defense |
| 62 | `yes please cancel` | False | `cancel` deny token |
| 63 | `yes, abort that` | False | Destructive verb after yes |
| 64 | `yes please undo` | False | Destructive verb after yes |
| 65 | `yes - delete that` | False | Destructive verb after yes |
| 66 | `yes that was wrong` | False | Incorrect after yes |
| 67 | `ok, scrub it` | False | `scrub` deny token |
| 68 | `confirm, but discard the body` | False | `discard` deny token |
| 69 | `yep, ignore that draft` | False | `ignore` deny token |
| 70 | `yes but no` | False | `but` + `no` deny tokens |
| 71 | `confirm — no hold on` | False | `no` + `hold on` deny tokens |
| 72 | `send to John instead` | False | `instead` deny token |
| 73 | `Hello John, lunch tomorrow?` (echoed draft text) | False | No affirmative head; safe hold |
| 74 | `tether, yes` | True | Salutation stripped, then `yes` matches |
| 75 | `@tether send it` | True | Salutation stripped |
| 76 | `ai: confirm` | True | Salutation stripped |
| 77 | `Yes. Now send the second one too` | True | Affirmative head, no deny tokens — accepted (scope-creep tolerated; §5) |
| 78 | `Yes! Also change the time` | False | `change` deny token |
| 79 | `sí` | False | Non-English; v1 English-only |
| 80 | `oui` | False | Non-English |
| 81 | `はい` | False | Non-English |
| 82 | `` (empty) | False | Guard clause |
| 83 | `   ` (whitespace only) | False | Guard clause |
| 84 | `confirm 12345678901234` | True (+ warn log) | Adversarial digit-run heuristic logs warning but does not block; tool layer validates draft_id |
