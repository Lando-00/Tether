# Recursive Language Models (RLMs) — Orchestration Research

**Date:** 2026-08
**Hardware context:** Snapdragon X Elite, 16 GB UMA, `bartowski/Qwen_Qwen3-8B-GGUF:Q4_0` via GenieX (NPU)
**Status:** Research note — no implementation decision taken yet

> **Correction (2026-08-18).** Sections 6.1–6.2 below were written against an
> assumed **Qwen3-1.7B** default. That is out of date: GenieX now ships
> **Qwen3-8B at Q4_0** (`src/tether/config/default.yml`, `context_window: 4096`).
> The root-capability verdict softens accordingly but does not reverse — see
> **§6.5**, which supersedes the 1.7B framing and covers the tuned RLM weights
> the authors actually released.

---

## TL;DR

RLMs (arXiv:2512.24601) are a real, published, open-source inference strategy from MIT OASYS
(Zhang, Kraska, Khattab, Dec 2025 / May 2026 revised). The core idea — offload context into a
Python REPL as a variable and let the model recursively grep, chunk, and sub-call itself over it
— is genuinely novel and shows dramatic benchmark improvements over frontier models on hard
long-context tasks. **The mechanism is, however, deeply dependent on a capable root model that
can write reliable Python code**, and every result in the paper uses GPT-5 or GPT-5-mini as the
root. There is *no* published evidence that the approach works with a 1–2 B parameter model as
root, and strong reasons to expect it will not. The REPL requirement is also a hard incompatibility
with Tether's current architecture and project scope. Individual RLM *ideas* — treating context as
an addressable variable, recursive decomposition without a REPL, per-fragment sub-calls — map
cleanly onto Tether's existing Notebook orchestrator and the scoped "Option C" fact-extraction
work, and those connections are the actionable takeaway.

---

## 1. What "RLM" Is — Disambiguation

"RLM" is an overloaded initialism. For this document it means **Recursive Language Models**,
not "RLM" in the sense of regularised linear models, rocket lifecycle management, or any other
domain. The orchestration-relevant paper is unambiguous:

| Field | Value |
|---|---|
| **Title** | Recursive Language Models |
| **Authors** | Alex L. Zhang, Tim Kraska, Omar Khattab |
| **Affiliation** | MIT OASYS Lab / MIT DSG |
| **Venue / status** | arXiv preprint (cs.AI, cs.CL) — **not peer-reviewed as of 2026-08** |
| **arXiv ID** | [2512.24601](https://arxiv.org/abs/2512.24601) — first submitted Dec 2025, revised May 2026 |
| **Blog post** | [alexzhang13.github.io/blog/2025/rlm/](https://alexzhang13.github.io/blog/2025/rlm/) — Oct 2025 (predates the arXiv submission) |
| **Reference code** | [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm) (MIT license, actively maintained by authors) |
| **Minimal demo** | [github.com/alexzhang13/rlm-minimal](https://github.com/alexzhang13/rlm-minimal) |

> ⚠️ **Status caveat.** The paper is an arXiv preprint. It has not been peer-reviewed or published
> at a venue as of this writing. The blogpost precedes the paper and is the primary public
> description of the results; the arXiv submission expands and formalises it. Both are primary
> sources here.

---

## 2. Mechanism — Precise Description

### 2.1 The Core Idea

A standard LLM call is `M(query, context) → str`. An RLM wraps this as
`RLM_M(query, context) → str`, with the same external signature. Internally, instead of passing
the entire context to the model in its prompt, the RLM:

1. Stores the (potentially enormous) **context as a Python string variable** in a REPL
   environment (a notebook-style Python `exec` loop, similar to Jupyter).
2. Passes the **root LM** (depth = 0) only the *query* plus metadata about the variable's
   existence, size, and structure — but **not the context text itself**.
3. The root LM writes Python code cells in an iterative loop. It can:
   - Peek at slices of the context (`context[:2000]`)
   - Run regex/grep queries over it
   - Partition it into chunks
   - Call `llm_query(sub_query, sub_context_slice)` — a sub-LM call (depth = 1) that runs
     an isolated, fresh LLM invocation over a fragment
   - Call `rlm_query(sub_query, sub_context)` — a recursive sub-RLM call (not used at
     depth=1 in the published experiments)
   - Collect and aggregate results in REPL variables
4. When confident, the root LM emits `FINAL(answer)` or `FINAL_VAR(var_name)` to terminate.

Source: [blog post §"Recursive Language Models (RLMs)"](https://alexzhang13.github.io/blog/2025/rlm/),
[`rlm/core/rlm.py`](https://github.com/alexzhang13/rlm/blob/main/rlm/core/rlm.py).

### 2.2 Termination and Depth Control

Termination is controlled by hard limits exposed as constructor arguments
([`rlm/core/rlm.py`](https://github.com/alexzhang13/rlm/blob/main/rlm/core/rlm.py)):

| Parameter | Default | Purpose |
|---|---|---|
| `max_depth` | 1 | Maximum recursion depth. At `depth >= max_depth`, falls back to a plain LM call. |
| `max_iterations` | 30 | Maximum REPL iterations (code cells) per RLM call. |
| `max_budget` | None | USD cost ceiling (requires a cost-tracking backend like OpenRouter). |
| `max_timeout` | None | Wall-clock seconds. Stops and returns best partial answer. |
| `max_tokens` | None | Total input+output token ceiling. |
| `max_errors` | None | Consecutive execution errors before stopping. |

The authors' experiments used **depth = 1 only** (root can call sub-LMs but sub-LMs cannot
themselves spawn further RLMs). They note enabling depth > 1 "is a relatively easy change" but
was not needed for the benchmarks tested. Infinite recursion is therefore avoided by the depth
cap and iteration cap, not by any structural guarantee from the LM itself.

### 2.3 Sub-result Aggregation

Sub-call results are returned as strings into the REPL's variable namespace. The root LM is
responsible for aggregating them — typically by collecting them into a list, then either
summarising over them in another sub-call or reasoning over the collected variables directly.
The authors identify four emergent strategies the root LM has adopted autonomously:
**Peek**, **Grep**, **Partition+Map** (map sub-calls over chunks in parallel via
`rlm_query_batched`), and **Summarisation**.

The parallelism is controlled by `max_concurrent_subcalls` (default 4); batched sub-calls use
Python threads. Source: [`rlm/core/rlm.py`](https://github.com/alexzhang13/rlm/blob/main/rlm/core/rlm.py).

### 2.4 Model Choices in the Published Experiments

| Role | Model used | Notes |
|---|---|---|
| Root LM (OOLONG experiment) | GPT-5-mini | Root writes REPL code and orchestrates sub-calls |
| Sub-LM (OOLONG experiment) | GPT-5-mini | Same model, fresh context, shorter sub-prompt |
| Root LM (BrowseComp-Plus) | GPT-5 | Larger model for harder multi-hop task |
| Sub-LM (BrowseComp-Plus) | GPT-5-mini | Cheaper model for fragment processing |
| Post-trained variant | RLM-Qwen3-8B | Fine-tuned on REPL interaction data via prime-rl/verifiers |

There is **no experiment in the paper or blog post using a model smaller than GPT-5-mini as
root**. RLM-Qwen3-8B is fine-tuned for the role and functions as the root, but at 8B parameters
it is still ~5× the size of Qwen3-1.7B and substantially more capable at code generation.

### 2.5 REPL Environment Options

The authors ship multiple sandboxing tiers
([README](https://github.com/alexzhang13/rlm/blob/main/README.md)):

| Environment | Isolation | Notes |
|---|---|---|
| `local` (default) | None — `exec()` in host process | Not for production use |
| `ipython` | Subprocess or in-process IPython kernel | Adds timeout enforcement |
| `docker` | Full container isolation | Recommended for untrusted content |
| `modal`, `prime`, `daytona`, `e2b` | Cloud sandboxes | For production-grade isolation |

The default `local` environment runs generated code via Python's `exec()` in the same process and
virtual environment as the host. The README states: "Using this REPL is generally safe, but
**should not be used for production settings**."

---

## 3. Benchmark Results — What the Authors Report

> ⚠️ **Framing note.** All results below are from the authors' own paper and blog post —
> i.e., the same group that proposed the method. No independent replication is known as of
> 2026-08. The OOLONG benchmark data was shared with the authors by the benchmark authors
> specifically for these experiments. These are promising early results, but treat them as
> **author-reported**, not independently verified.

### 3.1 OOLONG (`trec_coarse` split)

Task: distributional queries over semantically labelled rows; contexts of 132k–263k tokens.
Full context fits in GPT-5/GPT-5-mini context window — degradation is not due to truncation.

| Method | OOLONG@132k score | OOLONG@263k score | Avg cost/query |
|---|---|---|---|
| GPT-5 (plain) | ~29.7 | ~30.0 | baseline |
| GPT-5-mini (plain) | ~19.8 | ~15.0 | lower |
| RLM(GPT-5) without sub-calls | ~39.5 | ~36.0 | similar to GPT-5 |
| **RLM(GPT-5-mini)** | **~63.7** | **~45.0** | **≈ GPT-5** |
| ReAct + GPT-5 + BM25 | lower | lower | — |

*RLM(GPT-5-mini) outperforms GPT-5 by >34 raw points (~114% increase) at 132k; by >15 points
(~49% increase) at 263k.* Source: [blog post §"Exciting Result #1"](https://alexzhang13.github.io/blog/2025/rlm/).
Exact numeric scores are read from the blog post's figures; the blog post does not tabulate them
precisely, so these values are approximate ±1–2 points.

### 3.2 BrowseComp-Plus (20-query sample)

Task: multi-hop retrieval over corpora of 10–1000 documents (up to 11M tokens).
Results are on **20 randomly sampled queries** — a very small sample.

| Method | @10 docs | @100 docs | @1000 docs | Avg cost/query |
|---|---|---|---|---|
| GPT-5 (full context) | ~100% | drops | truncated | high |
| GPT-5 + BM25 (top 40) | ~100% | ~60% | ~40% | moderate |
| RLM(GPT-5) w/o sub-calls | ~100% | ~90% | ~90% | moderate |
| **RLM(GPT-5)** | **~100%** | **~100%** | **~100%** | **$0.99 avg** |
| ReAct + GPT-5 + BM25 | ~100% | ~70% | ~50% | moderate |

At 1000 documents (~6–11M tokens), RLM(GPT-5) is the only method maintaining near-perfect
performance. Estimated linear cost for GPT-5-mini ingesting 6–11M tokens: $1.50–$2.75; RLM
averaged $0.99 while outperforming all baselines by ≥29%.
Source: [blog post §"Exciting Result #2"](https://alexzhang13.github.io/blog/2025/rlm/),
[rlm.md/research.html](https://rlm.md/research.html).

> ⚠️ **20-query sample caveat.** 20 queries is too small for statistical confidence. The authors
> themselves describe these as "preliminary results" and "early results". The cost numbers
> especially should be treated as illustrative, not definitive.

### 3.3 RLM-Qwen3-8B Post-trained Variant

A fine-tuned version (`RLM-Qwen3-8B`) trained with RL on REPL interaction data via the
[prime-rl / verifiers](https://github.com/PrimeIntellect-ai/prime-rl) framework.
Reported result: **+28.3% average across four long-context benchmarks** (S-NIAH, OOLONG,
BrowseComp-Plus, CodeQA) relative to base Qwen3-8B.
On three of the four benchmarks it approaches vanilla GPT-5 performance.
Source: search result citing arXiv:2512.24601; **I could not directly verify this figure
from the blog post alone** — it appears in the arXiv paper body, which I could not parse (PDF).

### 3.4 Cost and Latency Characteristics

The authors state explicitly:
> "We did not optimise our implementation of RLMs for speed, meaning each recursive LM call is
> both blocking and does not take advantage of any kind of prefix caching."
> — [blog post §"Limitations"](https://alexzhang13.github.io/blog/2025/rlm/)

Depending on the root LM's chosen partitioning strategy, queries ranged from "a few seconds to
several minutes." No specific token-count-per-query figures are published in the blog post.
The cost data above suggest that a BrowseComp-Plus query at 1000 documents uses roughly $0.99
of GPT-5 API calls, which at GPT-5 pricing implies hundreds of thousands of tokens across all
calls — consistent with many (tens to hundreds) of sub-LM invocations.

---

## 4. Critical Evaluation — Failure Modes and Limitations

### 4.1 Author-Stated Limitations

From the [blog post §"Limitations"](https://alexzhang13.github.io/blog/2025/rlm/):

1. **No asynchrony or prefix caching.** Each recursive LM call is blocking. Latency can be
   "seconds to several minutes." This is called out as "low-hanging fruit" for systems optimisation,
   implying it is a known gap, not a solved problem.
2. **No strong guarantees on cost or runtime.** "We do not currently have strong guarantees
   about controlling either the total API cost or the total runtime of each call."
3. **Counting problems degrade at long contexts.** On OOLONG@263k, RLM performance drops on
   numerical/counting queries even though these were already weak for GPT-5 at 132k.
4. **Performance degrades with context length.** Even RLM(GPT-5-mini) drops from ~63.7 to ~45.0
   between 132k and 263k — significant degradation, though still better than baselines.

### 4.2 What the Authors Do Not Discuss That Is Important

- **Small root model viability.** The paper contains no experiment with a sub-8B model as root.
  The code-writing requirement is demanding: the root must generate syntactically valid Python
  that manipulates string variables, calls `llm_query()` with appropriate arguments, and
  terminates cleanly with `FINAL(...)`. This is qualitatively harder than instruction following
  or fact retrieval, and known to be a weak point of small models.
- **Independent replication.** No independent group has published reproduced results as of 2026-08.
  The benchmark data for OOLONG was shared privately with the authors. Community adoption (DSPy,
  Prime Agent, Ax) is real, but all of that is inferential about adoption, not independent
  replication of the paper's specific numerical claims.
- **Prompt injection into the REPL.** The default `local` environment runs arbitrary generated
  Python code via `exec()`. If the context variable contains adversarially crafted content (e.g.,
  a web page that includes Python string literals designed to manipulate the LM's REPL code),
  the model could be induced to execute malicious code. The paper discusses mitigation options
  (filtered namespaces, Docker isolation, cloud sandboxes) but the vulnerability exists by
  construction in the core design. The README acknowledges: "should not be used for production
  settings" for the default environment.

### 4.3 Where RLM is Likely Worse Than Simpler Approaches

| Scenario | Better alternative | Reason |
|---|---|---|
| Short context (< 32k tokens), single-hop question | Plain LLM call | RLM overhead (code generation, REPL loop) is unnecessary |
| Single-entity retrieval (NIAH) | BM25 or dense retrieval | Retrieval is O(1) cost; RLM is overkill |
| Weak root model (< 7B, no code fine-tuning) | Notebook-style orchestration | Root model cannot reliably write correct REPL code |
| Latency-sensitive single-turn response | Plain call or RAG | RLM latency is "seconds to minutes" per the authors |
| No sandbox available, untrusted input | Any non-REPL approach | Security risk from code execution over untrusted text |

---

## 5. Prior and Adjacent Art — Situating RLMs

| Method | Year | Core idea | Similarity to RLM | Key difference |
|---|---|---|---|---|
| [ReAct](https://arxiv.org/abs/2210.03629) | 2022 | Interleave reasoning traces and tool calls | Iterative, context-shrinking | Problem-centric decomposition; no recursive self-calls; context grows |
| [Least-to-most prompting](https://arxiv.org/abs/2205.10625) | 2022 | Decompose problem into sub-problems, solve sequentially | Recursive decomposition | Fixed structure, not context-centric; no REPL |
| [Map-Reduce / Refine summarisation](https://python.langchain.com/docs/modules/chains/document/map_reduce) | 2023 | Chunk → summarise → combine | Partition+Map is an RLM sub-strategy | Fixed pipeline; model doesn't choose partition strategy |
| [MemGPT](https://arxiv.org/abs/2310.08560) | 2023 | Paged context management with LM control | LM controls what enters context | Single context window managed by the LM; no recursive sub-calls |
| [GraphReader](https://arxiv.org/abs/2406.14550) | 2024 | Graph-of-nodes over document; atomic fact extraction | Graph-based decomposition | Prescribed structure; facts extracted into notebook, not REPL variables |
| **Ralph Loop / Hanov** | 2025 | Atomic fact extraction per iteration, notebook of facts, context discarded | Bounded per-call context | No REPL; no sub-LM calls; decomposition is structured not model-chosen |
| [CodeAct](https://arxiv.org/abs/2402.01030) | 2024 | LLM executes actions via code in a REPL | Direct inspiration for RLM's REPL | CodeAct is problem-centric (tools/actions); RLM is context-centric |
| **RLM** | 2025 | Context as REPL variable; model chooses decomposition; recursive self-calls | — | Unique: context-centric, model-chosen strategy, recursive sub-LM calls |
| [MemWalker](https://arxiv.org/abs/2310.05029) | 2023 | Tree-structured summarisation of long context | Hierarchical decomposition | Fixed tree structure; no REPL; no recursion |

The key novelty claim is: **prior work decomposes by problem structure (agent decides what to do);
RLM decomposes by context structure (model decides how to look at data)**. This framing is
the authors' own ([blog post §"Related Works"](https://alexzhang13.github.io/blog/2025/rlm/));
I found it substantiated by the comparison to CodeAct and ReAct.

The closest prior art is probably **MemGPT** (paged context under model control) but MemGPT
does not spawn recursive sub-LM instances — the same single model manages everything in one
rolling context. The RLM's recursive sub-calls are genuinely distinguishing.

---

## 6. Applicability to Tether — The Part That Matters Most

### 6.1 Does RLM's Core Mechanism Depend on a Frontier Root Model?

**Yes, critically and unambiguously.**

The root LM in an RLM must:
1. Write syntactically correct Python code cells in a REPL loop.
2. Choose an appropriate partitioning / grepping strategy for an arbitrary context.
3. Compose `llm_query()` calls with well-formed sub-queries and context slices.
4. Aggregate sub-results and reason over them.
5. Terminate cleanly by emitting `FINAL(...)`.

All of these require reliable code generation and complex multi-step planning under minimal
supervision. GPT-5-mini already sits at the frontier of small capable models; the smallest
model tested as root is GPT-5-mini, which is a gating model with reasoning capabilities far
beyond any 1–2B model currently available.

**Qwen3-1.7B as root: extremely unlikely to work.** At 1.7B parameters, Qwen3 is a capable
chat model for its size but is not a strong Python code generator, and it has not been
fine-tuned for REPL interaction. The RLM failure mode for a weak root model is not graceful
degradation — it is **infinite REPL loops, malformed code that crashes the executor, and
nonsense answers** that are harder to detect than a plain bad answer. The `max_iterations=30`
and error caps help, but they result in wasted compute, not a useful answer.

**RLM-Qwen3-8B as a reference point.** Even the fine-tuned 8B model required explicit RL
training on REPL interaction data to perform well. The authors provide a training harness
precisely because zero-shot REPL orchestration at small scale is hard.

**Conclusion (direct):** The full RLM mechanism as published is **not viable with Qwen3-1.7B
as the root model**. This is the single most important finding for Tether's evaluation.

### 6.2 Latency Arithmetic on an NPU at Low-Tens of Tokens/Second

Assumptions (stated explicitly):
- Decode speed: **20 tok/s** on Snapdragon X Elite NPU (Qwen3-1.7B via GenieX; this is
  consistent with the `~18–20 tok/s` measured in `10_xllamacpp_experimentation.md` for a
  comparable model).
- Prefill: faster but comparable in total time for medium prompts; ignore for order-of-magnitude.
- Root model generates ~100–200 tokens per REPL iteration (code cells tend to be short).
- Sub-calls: each sub-LM invocation processes a context fragment (~1000–5000 tokens prefill)
  and generates ~100–300 tokens.
- A "light" RLM query: 5 root iterations + 3 sub-calls.
- A "heavy" RLM query (BrowseComp-Plus-style): 20 root iterations + 20 sub-calls.

**Light query estimate:**
- Root iterations: 5 × (100 tok output / 20 tok/s) = 5 × 5 s = **25 s root decode**
- Sub-calls (sequential, as per authors): 3 × (200 tok output / 20 tok/s) = 3 × 10 s = **30 s**
- Plus prefill overhead (rough 2× multiplier): **~110 s total** (~2 min)

**Heavy query estimate:**
- Root iterations: 20 × 5 s = 100 s
- Sub-calls: 20 × 10 s = 200 s
- Prefill overhead: **~600 s total** (~10 min)

The authors themselves report queries ranging from "seconds to several minutes" on API-backed
frontier models with fast servers. On a local NPU at 20 tok/s, the latency floor is 5–10× higher.
**A "light" RLM query would take on the order of 2 minutes; a "heavy" query, 10+ minutes.**
This is unusable for interactive single-user chat.

Even with aggressive parallelism (the `max_concurrent_subcalls=4` option), batched sub-calls
would require concurrent NPU access, which the current GenieX / single-process serving model
does not support.

### 6.3 Which RLM Ideas Are Portable to Tether

Even though the full mechanism is not viable, several ideas transfer cleanly.

#### Idea 1: Context as Addressable Variable (Not a Monolithic Blob)

RLM's key framing: the context is an *object to be understood*, not a prompt to be stuffed.
The Notebook orchestrator already embodies this — each LLM call sees only `(question, notebook,
current tool result)`, not the accumulated history. The Notebook is the "addressed" version of
the context.

**Connection to Tether:** This idea is already implemented in `NotebookOrchestrator`
([`src/tether/protocol/orchestration/notebook.py`](../../src/tether/protocol/orchestration/notebook.py)).
ADR-0020 explicitly states that per-call context stays at ~1.6k tokens vs. ~6k for Chatty
after five web searches. The Notebook is Tether's production-ready version of this idea.

#### Idea 2: Recursive Decomposition Without a REPL

The Map+Summarise pattern RLM uses (chunk context → sub-call each chunk → aggregate) does not
strictly require a REPL. It can be implemented as deterministic orchestration logic:

```python
async def _extract_from_long_result(question, tool_result, notebook):
    # If tool_result is short, extract directly (existing behaviour)
    if len(tool_result) < DIRECT_EXTRACT_THRESHOLD:
        return await _extract_facts(question, tool_result, notebook)
    # If long, chunk and extract per chunk
    chunks = chunk(tool_result, size=CHUNK_SIZE)
    facts = []
    for chunk in chunks:
        facts += await _extract_facts(question, chunk, notebook)
    return facts
```

This is precisely the scoped "Option C" work: *generalise fact-extraction so all tool results
become compact facts, not just web-search results*. RLM's Map step validates the approach; the
key insight is that you do not need the root model to decide to do this — **the orchestrator
decides structurally**, which is exactly what small models need.

#### Idea 3: Progressive Peek Before Full Processing

RLM's root LM commonly starts by "peeking" at the first N characters of a large context before
deciding how to decompose it. The analogue in Tether is the **Planner phase** of
`NotebookOrchestrator`, which already decomposes the question into sub-queries before any tool
call is made. A natural extension is a lightweight pre-inspection pass on long tool results
before deciding whether to chunk them.

#### Idea 4: Separate Sub-Model for Extraction

RLM uses GPT-5-mini as the sub-LM (cheaper, faster) and GPT-5 as the root. In Tether's
context, if a second endpoint becomes available (e.g., the CPU path + NPU path simultaneously),
running a cheaper/faster model for per-fragment extraction while the primary model handles
synthesis is a direct adoption of this idea. It is a future option contingent on multi-model
routing, which is currently out of scope.

### 6.4 Does RLM-ish Recursion Conflict with Tether's "No Multi-Agent Routing" Boundary?
**It depends on how strictly "multi-agent" is defined.**

Tether's boundary as described: "no autonomous/multi-agent routing." The concern is autonomous
agents spawning other agents without user visibility or control.

The NotebookOrchestrator is already a multi-*call* design (Plan, Extract × N, Synthesize) but
is not multi-*agent* — there is one session, one user, one orchestrator instance; the LLM calls
are structured sub-steps, not autonomous agents with independent goals.

**RLM full mechanism** would push into the "multi-agent" zone if the sub-calls are
model-chosen, unbounded, and asynchronous. The root model deciding at runtime to spawn 20
parallel sub-agents over 1000 document fragments, without the user knowing this is happening,
is architecturally equivalent to multi-agent routing.

**RLM ideas applied to the Notebook** (Option C-style structural chunking, deterministic
partition+extract) remain single-agent: the orchestration logic decides the strategy, the LLM
only executes extraction steps. This is fully compatible with the existing boundary.

**Conclusion:** The full RLM mechanism violates the "no autonomous multi-agent routing" boundary
in spirit. The extracted ideas, implemented as deterministic orchestrator logic, do not.

---

### 6.5 Are there tuned RLM weights we could actually run on GenieX?

**Short answer: the weights exist and the base model matches ours exactly — but no
NPU-compatible build exists, and the model is not a drop-in chat model.**

The authors released exactly one post-trained model. Verified against the HF API 2026-08-18:

| Repo | Format | Size | NPU-capable? |
|---|---|---|---|
| [`mit-oasys/rlm-qwen3-8b-v0.1`](https://huggingface.co/mit-oasys/rlm-qwen3-8b-v0.1) | safetensors (4 shards) | — | n/a — needs GGUF conversion |
| [`mitkox/rlm-qwen3-8b-v0.1-Q4_K_M-GGUF`](https://huggingface.co/mitkox/rlm-qwen3-8b-v0.1-Q4_K_M-GGUF) | **Q4_K_M** | 4.68 GB | ❌ K-quant → CPU/GPU fallback |
| [`cameronbergh/rlm-qwen3-8b-v0.1-gguf`](https://huggingface.co/cameronbergh/rlm-qwen3-8b-v0.1-gguf) | f16 / q8_0 | 15.26 / 8.11 GB | ❌ wrong quant *and* too large |

**No `Q4_0` build exists anywhere.** Per `docs/runbooks/geniex-provider.md`, `Q4_0` is not a
preference but the Hexagon NPU requirement — every other quantisation silently falls back off
the NPU. So nothing here is pullable into GenieX as-is.

Three further observations, in descending order of importance:

1. **The base model is already ours.** `rlm-qwen3-8b-v0.1` is a fine-tune of `Qwen/Qwen3-8B`,
   and GenieX's shipped default is `bartowski/Qwen_Qwen3-8B-GGUF:Q4_0` (4.46 GB). A self-made
   `Q4_0` of the RLM fine-tune should land at essentially the same size and memory profile —
   which the runbook already records as stable on its own (~1.3 s/response warm). The
   conversion path is the standard llama.cpp one: `convert_hf_to_gguf.py` then
   `llama-quantize … Q4_0`. This is the only genuinely promising route.

2. **It is not a general chat model.** The model card is explicit: *"trained on trajectories
   produced using a fixed system prompt. It assumes the environment/scaffold from our RLM
   repo,"* with vLLM plus their inference code recommended. Swapping it in as a general Tether
   model would very likely perform **worse** than base Qwen3-8B, because it is specialised for
   emitting REPL code against a scaffold Tether does not have.

3. **Context budget fights the premise.** The GGUF advertises `context_length: 40960`, but
   GenieX runs `--nctx 4096` and `context_window: 4096`. RLM exists to exploit long context;
   at 4 k the root has little room for REPL transcripts on top of the task.

**What this does to §6.1.** The root-capability verdict softens but does not reverse. This
tuned 8B *is* precisely the artefact that makes small-model RLM plausible — the authors built
it because zero-shot REPL orchestration fails at this scale. But it still needs (a) a Q4_0
conversion nobody has published, (b) the RLM scaffold including a code sandbox Tether does not
have, and (c) more context than GenieX currently serves. Each is surmountable alone; together
they are a project, not an experiment.

---

## 7. Recommendation
### Verdict: **Adapt — do not adopt in full; do not reject the ideas**

| Aspect | Assessment |
|---|---|
| Full RLM (REPL + recursive self-calls) | ❌ **Reject for Tether now.** Root model capability gap (1.7B vs frontier), REPL security requirement, and latency (~2–10 min/query on NPU) are hard blockers. |
| RLM as an ideas source | ✅ **Adopt the framing.** Context-as-variable, per-fragment extraction, progressive peeking are directly actionable. |
| Monitoring RLM progress | ✅ **Watch.** The tuned `mit-oasys/rlm-qwen3-8b-v0.1` exists and shares our exact base model, but ships no `Q4_0` build and assumes the RLM scaffold — see §6.5. Revisit if someone publishes a `Q4_0` repack, or if a sandbox lands. |

### Concrete Next Experiments (Smallest Steps That Settle Open Questions)

1. **Option C fast-follow (highest value, lowest risk):** Extend `NotebookOrchestrator`'s
   `_extract_facts` to handle long tool results by chunking and iterating. No REPL. No new
   model. Directly answers "does Notebook degrade on large single tool results?" Implement as
   a feature flag on `ResearchSettings`.

2. **Qwen3-8B REPL smoke test (cheap, diagnostic):** Prompt the shipped
   `Qwen3-8B:Q4_0` (via GenieX) with a minimal RLM system prompt and a simple REPL task (find a
   number in a 10k-character string stored as a Python variable). Measure: does it produce valid
   Python? Does it terminate? How many iterations? This settles the root-model viability
   question for the model we actually run, with one afternoon of experiment — and needs no new
   weights. Only if this passes is a `Q4_0` conversion of the tuned model (§6.5) worth the effort.

3. **Latency profiling on multi-call Notebook queries:** Instrument `NotebookOrchestrator` to
   measure per-phase wall-clock time for real research queries. Establish the baseline before
   adding any RLM-inspired complexity. Currently uncharacterised.

4. **REPL security audit (before any REPL work):** If REPL execution is ever reconsidered,
   commission a threat model for code execution over web-search results. The prompt-injection
   surface is real and non-trivial; this should precede any implementation work.

---

## Sources

- [Zhang, Kraska, Khattab — "Recursive Language Models," arXiv:2512.24601, Dec 2025 / May 2026](https://arxiv.org/abs/2512.24601)
- [Alex Zhang — "Recursive Language Models" (blog post), Oct 2025](https://alexzhang13.github.io/blog/2025/rlm/)
- [alexzhang13/rlm — reference implementation (MIT license)](https://github.com/alexzhang13/rlm)
- [alexzhang13/rlm-minimal — stripped implementation](https://github.com/alexzhang13/rlm-minimal)
- [rlm.md/research.html — independent benchmark summary](https://rlm.md/research.html)
- [BrowseComp-Plus paper, arXiv:2508.06600](https://arxiv.org/abs/2508.06600)
- [CodeAct, arXiv:2402.01030](https://arxiv.org/abs/2402.01030)
- [GraphReader, arXiv:2406.14550](https://arxiv.org/abs/2406.14550)
- [Steve Hanov — "A Ralph Loop for reading"](https://stevehanov.ca/blog/a-ralph-loop-for-reading-beating-gpt-52-with-a-4k-context-window-and-4-gpus)
- [Geoffrey Huntley — Ralph Loop](https://ghuntley.com/ralph/)
- Tether ADR-0020: `docs/adr/0020-notebook-orchestrator-algorithm.md`
- Tether context strategies note: `docs/research/06_context_strategies.md`
