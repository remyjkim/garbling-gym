Below is a practical, “production-minded” manual for **PydanticAI** (aka **Pydantic AI**, `pydantic-ai`). It focuses on *best practices, usage patterns, runtime configuration*, and *extension mechanisms* (tools, toolsets/“skills”, MCP, durable execution, UI adapters, evals).

---

## 1) What PydanticAI is optimized for

**PydanticAI is an agent framework that treats types and validation as first-class runtime guardrails**: you define structured outputs (and tool signatures) as Python types / Pydantic models, and the framework will validate and (optionally) prompt the model to correct itself when validation fails. ([Pydantic AI][1])

Conceptually, an **Agent** is a container for:

* a model (and provider),
* instructions,
* dependency injection (deps),
* tools/toolsets,
* output spec + validation,
* and (optionally) message-history processing and streaming hooks. ([Pydantic AI][2])

---

## 2) Installation and package layout

### `pydantic-ai` vs `pydantic-ai-slim`

* `pydantic-ai` includes batteries like Logfire integration dependencies out of the box.
* `pydantic-ai-slim` is modular: you add extras for providers and features (OpenAI, Google, Logfire, web UI, durable execution, etc.). ([Pydantic AI][3])

Examples of feature/provider extras (non-exhaustive):

* `pydantic-ai-slim[openai]` ([Pydantic AI][4])
* `pydantic-ai-slim[google]` ([Pydantic AI][5])
* `pydantic-ai-slim[logfire]` ([Pydantic AI][6])
* `pydantic-ai-slim[web]` (web chat UI) ([Pydantic AI][7])
* `pydantic-ai-slim[prefect]`, `...[dbos]` etc. ([Pydantic AI][3])

---

## 3) Core concepts and how to use them well

### 3.1 Agents, models, providers, profiles

PydanticAI aims for **portability across vendors**: “Model” classes wrap vendor SDKs behind a unified interface so you can swap providers without rewriting your app logic. ([Pydantic AI][8])

Providers are responsible for authenticated clients; multiple providers can implement the same interface (e.g., OpenAI-compatible). ([Pydantic AI][9])

There’s also an internal idea of **model profiles** describing capabilities like tool support and JSON-schema output support—useful when you’re writing custom model integrations or debugging “why tools don’t show up”. ([Pydantic AI][10])

### 3.2 Dependencies (`deps`) and `RunContext`

PydanticAI strongly encourages **dependency injection** (credentials, DB handles, per-user configuration) via an explicit deps object, accessible from tools and other hooks through `RunContext`. ([Pydantic AI][11])

**Best practice**

* Put anything secret or environment-specific in `deps`, not in prompt text.
* Keep deps serializable-ish where possible (or at least easy to construct per request).
* Prefer `async` when deps will do I/O; sync functions are run in a thread pool. ([Pydantic AI][11])

### 3.3 Output specification: `result_type`, output functions, validators

PydanticAI supports a few ways to “shape” agent output:

1. **Structured result types** (e.g., Pydantic models) so the “final answer” is guaranteed to parse/validate (with retries when it doesn’t). ([Pydantic AI][1])

2. **Output functions**: like tools, but *the model must call one*, the call ends the run, and the result is *not* passed back to the model. Output functions can validate arguments and can raise `ModelRetry`. ([Pydantic AI][12])

3. **Output validators**: validate (and optionally retry) output; when streaming, validators run on partial output too, so you often gate on `RunContext.partial_output`. ([Pydantic AI][12])

**Best practice**

* Use a structured `result_type` for anything you will store, execute, or display in a UI.
* Add business-rule validation in Pydantic validators (or output validators) so failures are actionable and can be reflected back to the model.

### 3.4 Message history (memory) and history processors

You can intercept/transform message history before each model request using `history_processors`. This is a major lever for:

* privacy filtering,
* token/cost control,
* summarization,
* removing tool chatter, etc. ([Pydantic AI][13])

**Best practice**

* Treat history processors as part of your security boundary (redaction, PII removal).
* Copy history before mutating if you need the original (processors replace the history). ([Pydantic AI][13])

### 3.5 Tools and toolsets (your “skills” layer)

PydanticAI tools are type-safe functions callable by the model (with validated args). For advanced use, you group tools into **toolsets**, which can be:

* registered at agent construction,
* passed at run-time,
* built dynamically from context,
* temporarily overridden. ([Pydantic AI][14])

Toolsets also power most extension patterns: MCP servers, third-party tool libraries, and “skills frameworks”. ([Pydantic AI][14])

**Key advanced capabilities**

* Dynamic tool availability (filter/rename/prefix tools; build toolsets dynamically)
* Tool approval workflows
* Changing tool execution behavior
* Timeouts and retry behavior per tool/toolset ([Pydantic AI][15])

### 3.6 Streaming and event hooks

PydanticAI supports streaming output via `run_stream()` and exposes event hooks (e.g. `event_stream_handler`) to observe what’s happening mid-run. ([Pydantic AI][2])

When streaming:

* output functions/validators can be called multiple times; guard side effects with `RunContext.partial_output`. ([Pydantic AI][12])

---

## 4) Runtime options you should actually care about

### 4.1 ModelSettings: sampling, timeouts, stop sequences, parallel tools

Common cross-provider knobs live in `ModelSettings` (a `TypedDict`). Important fields include:

* `max_tokens`, `temperature`, `top_p` (choose temperature *or* top_p, not both),
* `timeout`,
* `seed`,
* `stop_sequences`,
* `parallel_tool_calls`,
* plus penalties and other advanced options. ([Pydantic AI][16])

**Best practice**

* For deterministic-ish workflows: `temperature=0` + `seed` where supported, and keep prompts stable.
* For tool-heavy agents: prefer low temperature and stronger schema constraints.
* Only enable `parallel_tool_calls` if your tools are safe to run concurrently and your business logic tolerates reordering. ([Pydantic AI][16])

### 4.2 Tool execution timeouts and retries

At the agent level you can set `tool_timeout`; per-tool timeouts override it. When a tool times out, PydanticAI treats it as a failure and issues a retry prompt to the model (consuming retry budget). ([Pydantic AI][17])

### 4.3 HTTP request retries

There’s dedicated guidance for HTTP retry strategies—watch latency blowups, align retry policy with your end-to-end timeout, and monitor retry rates. ([Pydantic AI][18])

### 4.4 Overriding configuration for tests / experiments

There’s an `agent.override()` context manager to temporarily override:

* agent name,
* deps,
* model,
* toolsets/tools,
* instructions, etc. ([Pydantic AI][17])

Also, there’s a context manager to override `ALLOW_MODEL_REQUESTS`—useful for preventing accidental live calls in unit tests. ([Pydantic AI][19])

### 4.5 Usage tracking

Run results expose usage accounting; in multi-agent delegation you typically forward usage so child runs count toward parent totals. ([Pydantic AI][20])

---

## 5) Best-practice patterns (the stuff that prevents 80% of pain)

### 5.1 Treat schemas as product interfaces

**Rule**: if another system consumes it, it must be a typed/validated schema.

* Define output as a Pydantic model.
* Use field constraints and validators for business rules.
* Keep the schema stable; version it if needed.

Why: PydanticAI will keep prompting until it gets valid output (within retry limits), which is exactly what you want for production pipelines. ([Pydantic AI][1])

### 5.2 Tool design: small, deterministic, and explicit

**Good tools**

* single responsibility (fetch X, compute Y),
* deterministic given inputs,
* return structured results (Pydantic models / dataclasses) when possible,
* validate inputs strictly (avoid “stringly typed” tools).

**Retry pattern**

* If a tool fails due to model-chosen bad args, raise `ModelRetry` with a short, corrective message so the model can call the tool again correctly. ([Pydantic AI][21])

**Timeout pattern**

* Prefer short tool timeouts plus retries for transient dependency issues, but long timeouts for truly long I/O is usually a durable-execution problem (see 5.7). ([Pydantic AI][17])

### 5.3 Dynamic tool availability (“capability routing”)

Use `prepare_tools` / `prepare_output_tools` (or dynamic toolsets) to:

* disable risky tools for certain users,
* swap implementations by tenant,
* reduce prompt/tool clutter by only exposing relevant tools. ([Pydantic AI][15])

### 5.4 Human-in-the-loop approvals and deferred tool execution

For dangerous operations (payments, deletions, code execution, external writes):

* Require approval at tool/toolset level.
* Use deferred tools where tool calls can pause pending approval or external execution. ([Pydantic AI][22])

This yields a clean control plane:

* model proposes action → system approves/executes → model continues with results.

### 5.5 Message-history hygiene (privacy + cost)

Use history processors to:

* redact secrets/PII,
* summarize older turns,
* drop irrelevant tool transcripts,
* enforce data-minimization. ([Pydantic AI][13])

### 5.6 Multi-agent patterns: delegation vs handoff vs graphs

PydanticAI documents multiple coordination styles:

* **Delegation**: a parent agent spawns a specialized sub-agent and then resumes control; forward usage. ([Pydantic AI][20])
* **Programmatic handoff**: often modeled with output functions (handoff ends current run). ([Pydantic AI][20])
* **Graph-based control flow**: for complex, stateful workflows and auditability; Pydantic Graph provides typed nodes and step-by-step iteration when you need fine control. ([Pydantic AI][23])

**Best practice**

* Start with single-agent + tools.
* Add delegation for specialization.
* Only move to graphs when the workflow truly needs explicit state machines / branching / step-level observability. ([Pydantic AI][23])

### 5.7 Reliability at scale: durable execution

If your workflows are long-running, must survive restarts, or have human pauses, use **Durable Execution** integrations (Temporal, DBOS, Prefect). They preserve progress and support streaming + MCP with fault tolerance. ([Pydantic AI][24])

### 5.8 Observability: Logfire tracing as default posture

PydanticAI can emit detailed traces/spans for:

* agent runs,
* model requests,
* tool calls, etc.,
  when Logfire is installed/configured and instrumentation is enabled. ([Pydantic AI][6])

**Best practice**

* Instrument early in development; it shortens the “why did the agent do that?” loop dramatically.

---

## 6) Extensions: tools, toolsets (“skills”), MCP, third-party ecosystems

### 6.1 Toolsets as the “skills” abstraction

Toolsets let you package capabilities as reusable modules (search, DB access, ticketing, planning, etc.). They can be combined, filtered, renamed/prefixed, and approval-wrapped. ([Pydantic AI][14])

Toolsets can also require IDs for durable execution contexts. ([Pydantic AI][25])

### 6.2 MCP (Model Context Protocol)

PydanticAI supports MCP in multiple ways:

1. act as an MCP client connecting directly to MCP servers,
2. use FastMCP client toolsets,
3. or use provider-native MCP tools (where supported). ([Pydantic AI][26])

There are also docs for running MCP servers (and even sample servers like a Python-sandbox MCP server). ([Pydantic AI][27])

**Best practice**

* Use MCP when you want a standardized, language-agnostic tool boundary and shared tool infrastructure across agents/apps.

### 6.3 Built-in tools vs “common tools”

PydanticAI distinguishes:

* **Built-in tools**: executed by the model provider’s infrastructure (e.g., web search, code execution, file search/RAG, memory, etc.). ([Pydantic AI][28])
* **Common tools**: implemented and executed by your PydanticAI runtime (example: DuckDuckGo search tool), installed via extras. ([Pydantic AI][29])

This affects security, latency, and deployment:

* provider-executed tools can be faster and leverage provider caching/context optimizations, but may reduce your control plane. ([Pydantic AI][28])

### 6.4 Third-party tools: LangChain and ACI.dev

PydanticAI offers integrations for third-party tool ecosystems, including wrapping LangChain tools; note the tradeoff: argument validation may not be enforced by PydanticAI in that path. ([Pydantic AI][30])

### 6.5 Embeddings + RAG

PydanticAI includes embeddings support (and example RAG apps). Use this when you need vector search and grounded context injection. ([Pydantic AI][31])

### 6.6 UI event streams and web chat UI

There’s built-in support for streaming UI protocols via adapters (SSE/event stream patterns), and a “Web Chat UI” you can run with extras. ([Pydantic AI][7])

### 6.7 Agent2Agent (A2A) protocol

A2A support includes context storage, conversation persistence, and representing results as “artifacts” (text/data parts). ([Pydantic AI][32])

---

## 7) Production checklists

### 7.1 Agent hardening checklist

* **Structured outputs everywhere** (Pydantic models for outputs and tool results)
* **Explicit tool allowlists**; dynamic tool exposure
* **Approval gates** for destructive or expensive actions ([Pydantic AI][14])
* **History processors** for redaction + summarization ([Pydantic AI][13])
* **Timeouts + retries** aligned with end-to-end SLA ([Pydantic AI][18])
* **Observability** enabled (Logfire tracing) ([Pydantic AI][6])
* **Test mode** that blocks live calls (`ALLOW_MODEL_REQUESTS` override) ([Pydantic AI][19])

### 7.2 When to choose what

* Need deterministic pipelines / ETL: structured result types + low temperature + strict validators. ([Pydantic AI][16])
* Need long-running workflows / pauses: durable execution (Temporal/DBOS/Prefect). ([Pydantic AI][24])
* Need many external tools across teams/languages: MCP toolsets. ([Pydantic AI][26])
* Need grounding over internal docs: embeddings + RAG tools. ([Pydantic AI][31])

---

## 8) Reference templates (copy/paste starting points)

### 8.1 Minimal typed agent + deps + tool retry

```python
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry

@dataclass
class Deps:
    weather_api_key: str

class Forecast(BaseModel):
    city: str
    celsius: float = Field(..., ge=-80, le=80)
    summary: str

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    result_type=Forecast,
)

@agent.tool
async def get_forecast(ctx: RunContext[Deps], city: str) -> dict:
    if not city.strip():
        raise ModelRetry("city must be a non-empty string")
    # call your API here using ctx.deps.weather_api_key
    return {"city": city, "celsius": 18.0, "summary": "mild"}
```

`ModelRetry` is the canonical way to say “bad args, try again”. ([Pydantic AI][21])

### 8.2 Streaming without duplicated side effects

```python
async with agent.run_stream("Give me a forecast for Paris") as result:
    async for partial in result.stream_output():
        # render partial output
        ...

    final = result.output  # validated final output
```

When you use output functions/validators with streaming, gate side effects on `RunContext.partial_output`. ([Pydantic AI][12])

### 8.3 Human approval / deferred tools

* Use deferred tools when tool calls need approval or external execution, and resume with `DeferredToolResults`. ([Pydantic AI][33])
* For toolset-level dynamic approval decisions, wrap with `ApprovalRequiredToolset`. ([Pydantic AI][14])

### 8.4 Override config for tests

```python
from pydantic_ai.models import override_allow_model_requests

with override_allow_model_requests(False):
    # any accidental run will fail fast instead of calling a real provider
    ...
```

([Pydantic AI][19])

---

## 9) Versioning and upgrade hygiene

PydanticAI hit **v1 in September 2025** with a stated commitment to API stability until v2, and the upgrade guide tracks breaking changes. ([Pydantic AI][34])

**Best practice**

* Pin minor versions in production.
* Track `changelog` for streaming/event/tool API changes and deprecations (e.g., streamed result method deprecations). ([Pydantic AI][35])

---

## If you want this even more “manual-like”

Tell me your intended deployment shape (CLI / web app / async worker), which model providers you’ll target, and whether you need MCP + durable execution. I can then produce:

* a recommended reference architecture,
* a “golden path” folder structure,
* and opinionated defaults for retries/timeouts/history processing/tool approval that match your constraints.

[1]: https://ai.pydantic.dev/?utm_source=chatgpt.com "Pydantic AI - Pydantic AI"
[2]: https://ai.pydantic.dev/agents/?utm_source=chatgpt.com "Agents"
[3]: https://ai.pydantic.dev/install/?utm_source=chatgpt.com "Installation"
[4]: https://ai.pydantic.dev/models/openai/?utm_source=chatgpt.com "OpenAI - Pydantic AI"
[5]: https://ai.pydantic.dev/models/google/?utm_source=chatgpt.com "Google"
[6]: https://ai.pydantic.dev/logfire/?utm_source=chatgpt.com "Debugging & Monitoring with Pydantic Logfire"
[7]: https://ai.pydantic.dev/web/?utm_source=chatgpt.com "Web Chat UI"
[8]: https://ai.pydantic.dev/models/overview/?utm_source=chatgpt.com "Overview - Pydantic AI"
[9]: https://ai.pydantic.dev/api/providers/?utm_source=chatgpt.com "pydantic_ai.providers"
[10]: https://ai.pydantic.dev/api/profiles/?utm_source=chatgpt.com "pydantic_ai.profiles"
[11]: https://ai.pydantic.dev/dependencies/?utm_source=chatgpt.com "Dependencies"
[12]: https://ai.pydantic.dev/output/?utm_source=chatgpt.com "Output"
[13]: https://ai.pydantic.dev/message-history/?utm_source=chatgpt.com "Messages and chat history"
[14]: https://ai.pydantic.dev/toolsets/?utm_source=chatgpt.com "Toolsets"
[15]: https://ai.pydantic.dev/tools-advanced/?utm_source=chatgpt.com "Advanced Tool Features"
[16]: https://ai.pydantic.dev/api/settings/?utm_source=chatgpt.com "pydantic_ai.settings"
[17]: https://ai.pydantic.dev/api/agent/?utm_source=chatgpt.com "pydantic_ai.agent"
[18]: https://ai.pydantic.dev/retries/?utm_source=chatgpt.com "HTTP Request Retries"
[19]: https://ai.pydantic.dev/api/models/base/?utm_source=chatgpt.com "pydantic_ai.models"
[20]: https://ai.pydantic.dev/multi-agent-applications/?utm_source=chatgpt.com "Multi-Agent Patterns"
[21]: https://ai.pydantic.dev/api/exceptions/?utm_source=chatgpt.com "pydantic_ai.exceptions"
[22]: https://ai.pydantic.dev/deferred-tools/?utm_source=chatgpt.com "Deferred Tools"
[23]: https://ai.pydantic.dev/graph/?utm_source=chatgpt.com "Graph"
[24]: https://ai.pydantic.dev/durable_execution/overview/?utm_source=chatgpt.com "Durable Execution"
[25]: https://ai.pydantic.dev/api/toolsets/?utm_source=chatgpt.com "pydantic_ai.toolsets"
[26]: https://ai.pydantic.dev/mcp/overview/?utm_source=chatgpt.com "Model Context Protocol (MCP)"
[27]: https://ai.pydantic.dev/mcp/server/?utm_source=chatgpt.com "Server"
[28]: https://ai.pydantic.dev/builtin-tools/?utm_source=chatgpt.com "Built-in Tools"
[29]: https://ai.pydantic.dev/common-tools/?utm_source=chatgpt.com "Common Tools"
[30]: https://ai.pydantic.dev/third-party-tools/?utm_source=chatgpt.com "Third-Party Tools"
[31]: https://ai.pydantic.dev/embeddings/?utm_source=chatgpt.com "Embeddings"
[32]: https://ai.pydantic.dev/a2a/?utm_source=chatgpt.com "Agent2Agent (A2A) Protocol"
[33]: https://ai.pydantic.dev/api/tools/?utm_source=chatgpt.com "pydantic_ai.tools"
[34]: https://ai.pydantic.dev/changelog/?utm_source=chatgpt.com "Upgrade Guide"
[35]: https://ai.pydantic.dev/api/result/?utm_source=chatgpt.com "pydantic_ai.result - Pydantic AI"
