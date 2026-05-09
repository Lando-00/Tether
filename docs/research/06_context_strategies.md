# Context Strategies — Ralph Loops & Graph-Reader for Tether

> **Inbox item, May 2026.** Captured from
> [Steve Hanov's "A Ralph Loop for reading"](https://stevehanov.ca/blog/a-ralph-loop-for-reading-beating-gpt-52-with-a-4k-context-window-and-4-gpus)
> (May 2026) and [Geoffrey Huntley's original
> Ralph Loop](https://ghuntley.com/ralph/) (Jul 2025).
>
> This is **architecture/orchestration research**, not model research.
> It belongs alongside the model picks in
> [`04_candidate_shortlist.md`](./04_candidate_shortlist.md) because the
> two interact: Ralph-style orchestration makes the small-practical-ctx
> models on our hardware *much* more useful for multi-step agentic
> tasks. Conversely, the better the orchestration, the less we need
> bigger models.

## TL;DR

- Two community techniques worth absorbing into Tether's orchestrator
  long term:
  1. **Ralph Loop** (Geoffrey Huntley, code generation): "the context
     window is a liability." Each loop iteration starts with a *fresh*
     context, does *one* atomic task, and persists state to the file
     system / git rather than to the chat trace.
  2. **Ralph Loop for reading / Graph-Reader** (Steve Hanov):
     same philosophy but for research / multi-step retrieval. State
     lives in a **Notebook of atomic facts**, not in chat history.
     Implemented in [`smhanov/laconic`](https://github.com/smhanov/laconic)
     (Go), inspired by the GraphReader paper.
- Both are direct counters to ReAct-style "chatty agent" loops where
  every tool result fattens the chat trace until the model loses its
  way.
- Tether's current `protocol/orchestration/orchestrator.py` is a
  classic chatty agent. With our practical context windows
  (Phi-3.5-mini ~16–20k, Qwen3-4B ~32k under the 9 GB cap), 4–5 tool
  iterations on rich tools (e.g. `web_search` returning 5+ snippets)
  will already eat measurable context.
- **Recommendation:** keep the existing chatty-agent flow as the
  default for short turns; **add a second orchestration mode**
  ("research mode") that uses the Notebook-of-facts pattern for
  multi-step research-style queries. Implement it after the refactor
  alongside the `NexaProvider` work — both pull on the same provider
  abstraction.

## What "Ralph" actually is

> *"Ralph is a Bash loop. `while :; do cat PROMPT.md | claude-code; done`"* — Huntley

The minimal claim is that **discarding the LLM's context between
iterations is a feature, not a bug.** Each loop:

1. Reads a prompt from the file system.
2. Spins up a fresh agent with no memory.
3. Tells it to do **one** thing and commit to git when done.
4. Kills the process.

State lives in the file system (and the spec files / `progress.txt`
the loop reads each turn). The model doesn't need to remember
anything because the file system does. This sidesteps the
context-window-bloat problem entirely at the cost of one fresh-load
penalty per iteration.

## "Ralph Loop for reading" / Graph-Reader

Hanov's adaptation for *research* tasks (paraphrasing the post):

| Phase    | What happens |
|----------|--------------|
| Plan     | LLM breaks the user's question into Key Elements ("Revenue", "CEO", "Competitors"). |
| Explore  | LLM enqueues search queries (graph nodes). |
| Extract  | For every search result, the LLM extracts **Atomic Facts** (e.g. `"Apple CEO is Tim Cook"`) into a Notebook. The raw search snippets are *thrown away*. |
| Refine   | Fuzzy facts spawn new search queries on the queue. |
| Answer   | When the Notebook has enough facts, LLM synthesizes the answer from the Notebook only. |

Crucially: **the LLM never sees the full history.** Each LLM call
sees only `(current question, current Notebook, current search result)`.
That keeps the context per-call small even when the overall task is
deep — exactly the property our 4–8k *practical* contexts need.

Hanov demonstrates the technique on `qwen3:4b` (effectively the same
model class as our `Qwen3-4B-q4f16_0-MLC`) successfully answering a
question whose answer is *outside the model's training data* — purely
through tool use plus the Notebook compression.

## Why this matters for Tether specifically

Look at the current orchestration shape in
[`tether_service/protocol/orchestration/orchestrator.py`](../../tether_service/protocol/orchestration/orchestrator.py)
and the history-handling in
[`tether_service/context/sqlite_store.py`](../../tether_service/context/sqlite_store.py):

- Each tool-call loop iteration adds the tool *result* to history as
  a `tool_result` message, then sends the full history to the model
  on the next iteration.
- With `max_tool_loops=5` (per `config/default.yml`) and a tool like
  `web_search` returning a 5-result Brave Search payload, even a
  single research query can push 4–10 KB of tool-result text into
  the chat trace.
- At Tether's 9 GB RAM budget on Snapdragon X Elite, the practical
  context windows are:

  | Model | Advertised ctx | Practical (per `04_candidate_shortlist.md`) |
  |-------|---------------:|--------------------------------------------:|
  | Qwen3-4B-q4f16_0-MLC | 40 960 | ~32 k |
  | Qwen2.5-7B-q4f16_0-MLC | 12 288 | ~12 k |
  | Phi-3.5-mini-instruct (deleted) | 40 960 | ~16–20 k |

  None of these is "infinite." A multi-step research task with rich
  tool output will hit the wall.

The Ralph-for-reading pattern is the structural answer.

## A sketch: a Notebook-mode orchestrator for Tether

This is *not* a proposal for the immediate post-refactor work. It's a
sketch so the idea is concrete.

```
NotebookOrchestrator(question):
    notebook = []                       # list of atomic facts
    queue    = plan(question)           # initial sub-queries

    while queue and len(notebook) < MAX_FACTS:
        sub_q = queue.pop_left()
        result = run_tool(sub_q)        # web_search, time, weather, etc.
        new_facts, new_queries = extract_facts(question, sub_q, result, notebook)
        notebook.extend(new_facts)
        queue.extend(new_queries)

    return synthesize(question, notebook)
```

Each call to `extract_facts` and `synthesize` is a *fresh* LLM
invocation that sees only what it strictly needs:

- `extract_facts(question, sub_q, result, notebook_so_far)` — small
  prompt, bounded by `len(result)` + `len(notebook_so_far)`. Notebook
  stays compact because facts are atomic strings.
- `synthesize(question, notebook)` — the *only* call that sees the
  whole notebook. Still bounded by `len(notebook)` rather than by the
  cumulative tool-output history.

This maps onto Tether's existing pieces cleanly:

- **`ModelProvider.stream(messages, tools)`** stays as-is. The
  Notebook orchestrator just builds smaller `messages` than today.
- **`Tool` / `ToolRunner`** stay as-is. The orchestrator is what
  changes.
- **`SqliteSessionStore`** would gain a `notebook` table or a JSON
  column on the session, persisting facts across turns the same way
  it persists messages.

The orchestrator becomes a swappable strategy:

```yaml
# default.yml (sketch — not real config)
protocol:
  orchestrator:
    impl: tether_service.protocol.orchestration.notebook.NotebookOrchestrator
    args:
      max_facts: 40
      planner_model: <model_name>
      synthesizer_model: <model_name>
```

…with the existing chatty-agent orchestrator left as the default for
short single-turn chat where the Notebook overhead doesn't pay off.

## When to use which mode

| Scenario | Recommended mode | Why |
|----------|------------------|-----|
| One-shot question, no tools | Direct (no orchestrator loop) | No iteration, no overhead. |
| Single tool call (e.g. "what time is it?") | Existing chatty agent | The chatty trace is shorter than the Notebook scaffolding. |
| 2–3 tool calls touching different topics | Existing chatty agent | Still fits in practical ctx. |
| Research workflow (5+ tool calls, large tool payloads, multi-source synthesis) | **Notebook orchestrator** | Where the savings show up. Specifically: anything that pulls multiple `web_search` results, anything that touches files/emails/calendars in future tools. |

The mode could be selected per-message via a flag on the
`/api/v1/chat/stream` request body, or auto-detected from the
question shape (heuristic: > N tool calls expected → Notebook).

## Don't confuse this with bigger context

A common reaction is "just buy more context." Two reasons not to:

1. The 9 GB RAM cap on Snapdragon X Elite is the **real** constraint;
   bigger ctx eats KV cache faster than it eats anything else (see the
   estimator findings in
   [`04_candidate_shortlist.md`](./04_candidate_shortlist.md)).
2. Even in big-ctx models, the *needle-in-the-haystack* problem
   degrades reasoning quality beyond a certain context fill ratio.
   Hanov's post calls this out: "even if I enabled the long contexts,
   the models struggle to reason over them." The Ralph technique
   sidesteps the quality issue, not just the budget issue.

## Cross-references

- Hardware constraints: [`02_adreno_mlc_landscape.md`](./02_adreno_mlc_landscape.md)
- KV-cache math: `scripts/research/estimate_ram.py`
- Practical context per model: [`04_candidate_shortlist.md`](./04_candidate_shortlist.md)
- Future runtime providers (where a new orchestrator could plug in):
  [`03_npu_hexagon_landscape.md`](./03_npu_hexagon_landscape.md) — a
  `NexaProvider` + a `NotebookOrchestrator` pair-up would be a
  particularly potent combination for energy-efficient research
  workflows.

## Sources

- [Steve Hanov, "A Ralph Loop for reading: beating GPT-5.2 with a 4k context window and 4 GPUs"](https://stevehanov.ca/blog/a-ralph-loop-for-reading-beating-gpt-52-with-a-4k-context-window-and-4-gpus)
- [Geoffrey Huntley, "Ralph Wiggum as a Software Engineer"](https://ghuntley.com/ralph/)
- [`smhanov/laconic`](https://github.com/smhanov/laconic) — Go reference implementation of the Notebook strategy.
- GraphReader paper (cited by Hanov) — the original "atomic facts" formulation for long-context reading.
