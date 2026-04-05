# agentflow

A production-grade Go framework for building agentic AI systems.

An agentic AI system is one where a language model autonomously decides to invoke tools, observes results, and continues reasoning in a loop until the task is complete. agentflow provides the core abstractions and orchestration to build such systems with any AI provider that supports tool use (function calling).

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Core Concepts](#core-concepts)
- [Execution Modes](#execution-modes)
- [Sub-Agent System](#sub-agent-system)
- [Session Persistence](#session-persistence)
- [Token Budget Management](#token-budget-management)
- [Tool Result Management](#tool-result-management)
- [Context Compaction](#context-compaction)
- [Input Validation](#input-validation)
- [Rate Limiting](#rate-limiting)
- [Structured Logging](#structured-logging)
- [Trace Context Propagation](#trace-context-propagation)
- [Multi-Provider Fallback](#multi-provider-fallback)
- [Multimodal Support](#multimodal-support)
- [Built-in Tools](#built-in-tools)
- [Middleware](#middleware)
- [Extension Packages](#extension-packages)
- [HTTP Server Deployment](#http-server-deployment)
- [Examples](#examples)
- [Testing](#testing)
- [License](#license)

## Features

- **Provider-agnostic** -- works with OpenAI, Anthropic, Google Gemini, Groq, OpenRouter, or any custom provider
- **Streaming-first** -- real-time event delivery via Go channels (17 event types)
- **Native tool calling** -- uses provider-native function calling APIs with JSON validation
- **Fuzzy tool name matching** -- handles models that shorten tool names (e.g. "search" matches "web_search")
- **Tool call repair** -- validation errors return full schema with field descriptions so the model can self-correct
- **Guaranteed tool results** -- every tool call gets a result, even if execution is cancelled or interrupted
- **Loop detection** -- SHA-256 signature tracking detects repetitive tool calling patterns (configurable threshold)
- **Early tool emission** -- tool calls emitted mid-stream when JSON arguments are valid, reducing latency
- **Auto-compaction** -- automatically compacts conversation on "context too large" errors and retries
- **Graceful tool fallback** -- models that don't support native tool calling (e.g. Groq compound) automatically retry without tools
- **Critical error handling** -- tool panics terminate the loop; validation errors are non-critical and sent to model for self-correction
- **Execution modes** -- ModeLocal for full access, ModeRemote for server-safe tools only
- **Input validation** -- automatic JSON Schema validation of tool inputs before execution
- **Rate limiting** -- token bucket rate limiter for provider API calls
- **Structured logging** -- optional slog integration for model calls, retries, compaction, budget
- **Trace context propagation** -- W3C traceparent headers propagated to provider HTTP requests
- **Tool timeout and retry** -- configurable per-tool timeout and retry on failure
- **Circuit breaker** -- middleware that blocks tools after consecutive failures
- **Error strategy** -- configurable error handling (transform errors, abort loop)
- **Anthropic thinking** -- native `thinking_delta` support for extended thinking/reasoning
- **Sub-agent system** -- spawn child agents, parallel orchestration, delegate_task tool
- **Agent cloning** -- derive specialized agents from a common base via `Clone()`
- **Session persistence** -- save/resume conversations with file-based or in-memory stores
- **Token budget** -- enforce consumption limits with warning thresholds
- **Concurrent tool execution** -- safe tools run in parallel, unsafe tools run in isolation
- **Hook system** -- intercept any lifecycle phase, multi-phase hooks supported
- **Permission control** -- AllowAll, DenyList, AllowList, ReadOnly, Chain, or custom checkers
- **Result size management** -- automatic truncation of oversized tool outputs
- **Context compaction** -- sliding window, token-aware, AI-powered summarization, staged
- **Multi-provider fallback** -- automatic failover across providers with health checking
- **Multimodal** -- send images (base64 or URL) alongside text
- **14 built-in tools** -- bash, file ops, search, HTTP, deep search, web search, calculator, date/time, URL reader, and more
- **Generic typed tools** -- `tools.NewTyped[I]()` with auto schema generation from struct tags
- **Event filtering** -- `FilterEvents()` helper to consume only specific event types
- **Panic recovery** -- tool panics are caught, marked critical, and terminate the loop safely
- **Smart retry logic** -- checks `x-should-retry` header, `io.ErrUnexpectedEOF`, status 408/409/429/5xx
- **Context overflow detection** -- `IsContextTooLarge()` detects token limit errors across providers
- **Zero external dependencies** -- core uses only the Go standard library
- **Comprehensive test suite** -- provider unit tests, OpenRouter integration tests, race tests, benchmarks

## Installation

```bash
go get github.com/CanArslanDev/agentflow
```

Requires Go 1.23 or later.

## Quick Start

```go
package main

import (
    "context"
    "fmt"

    "github.com/CanArslanDev/agentflow"
    "github.com/CanArslanDev/agentflow/provider/groq"
    "github.com/CanArslanDev/agentflow/tools"
)

type CalcInput struct {
    Expression string `json:"expression" description:"Math expression to evaluate"`
}

func main() {
    provider := groq.New("gsk-...", "llama-3.3-70b-versatile")

    calculator := tools.NewTyped[CalcInput](
        "calculator", "Evaluate a math expression",
        []string{"expression"},
        func(_ context.Context, input CalcInput, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
            return &agentflow.ToolResult{Content: "42"}, nil
        },
    )

    agent := agentflow.NewAgent(provider,
        agentflow.WithTools(calculator),
        agentflow.WithSystemPrompt("You are a helpful assistant."),
        agentflow.WithMaxTurns(10),
    )

    for ev := range agent.Run(context.Background(), []agentflow.Message{
        agentflow.NewUserMessage("What is 6 * 7?"),
    }) {
        switch ev.Type {
        case agentflow.EventTextDelta:
            fmt.Print(ev.TextDelta.Text)
        case agentflow.EventToolStart:
            fmt.Printf("\n[tool: %s]\n", ev.ToolStart.ToolCall.Name)
        case agentflow.EventTurnEnd:
            fmt.Printf("\n(done: %s)\n", ev.TurnEnd.Reason)
        }
    }
}
```

## Architecture

```
                        Agent.Run(ctx, messages)
                                |
                          <-chan Event  (17 event types)
                                |
    +---------------------------+---------------------------+
    |                    AGENTIC LOOP                       |
    |                                                       |
    |   +----------+     +-----------+     +----------+     |
    |   | PreModel |---->| Rate Limit|---->| Provider |     |
    |   |  Hooks   |     | + Retry   |     | Stream   |     |
    |   +----------+     +-----------+     +----+-----+     |
    |                                           |           |
    |                  Tool Calls?              |           |
    |                  /          \              |           |
    |                Yes          No --> return  |           |
    |                /                          |           |
    |   +-------------------+                   |           |
    |   |  Tool Pipeline:   |                   |           |
    |   |  1. Validate      |                   |           |
    |   |  2. PreHooks      |                   |           |
    |   |  3. Permission    |                   |           |
    |   |  4. Schema Check  |                   |           |
    |   |  5. Execute+Retry |                   |           |
    |   |  6. Error Strategy|                   |           |
    |   |  7. Size Limit    |                   |           |
    |   |  8. PostHooks     |                   |           |
    |   +--------+----------+                   |           |
    |            |                              |           |
    |   Append Results --> continue (next turn) |           |
    +-------------------------------------------------------+
```

## Core Concepts

### Provider

Abstracts the AI model API. Implement the Provider interface to add new backends:

```go
type Provider interface {
    CreateStream(ctx context.Context, req *Request) (Stream, error)
    ModelID() string
}
```

Optionally implement `HealthChecker` for proactive health checking:

```go
type HealthChecker interface {
    HealthCheck(ctx context.Context) error
}
```

Built-in providers:

| Provider | Package | Models | Auth |
|----------|---------|--------|------|
| OpenAI | `provider/openai` | gpt-4o, gpt-4o-mini, gpt-4-turbo | `Authorization: Bearer sk-...` |
| Anthropic | `provider/anthropic` | claude-sonnet-4, claude-haiku-4.5, claude-opus-4 | `x-api-key: sk-ant-...` |
| Google Gemini | `provider/gemini` | gemini-2.0-flash, gemini-2.5-pro | API key in URL param |
| Groq | `provider/groq` | llama-3.3-70b, mixtral-8x7b | `Authorization: Bearer gsk_...` |
| OpenRouter | `provider/openrouter` | All models via unified API | `Authorization: Bearer sk-or-...` |
| Ollama | `provider/ollama` | Any Ollama model (llama3, mistral, etc.) | None (URL-based) |
| Fallback | `provider/fallback` | Cascading failover across any providers | N/A |
| Mock | `provider/mock` | Deterministic testing without API calls | N/A |

### Tool

Three ways to define tools:

**Generic typed (recommended -- type-safe, auto schema):**
```go
type SearchInput struct {
    Query      string `json:"query" description:"Search query"`
    MaxResults int    `json:"max_results,omitempty" description:"Max results"`
}

tool := tools.NewTyped[SearchInput](
    "search", "Search the web",
    []string{"query"},
    func(ctx context.Context, input SearchInput, p agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
        return &agentflow.ToolResult{Content: "results for: " + input.Query}, nil
    },
)
```

**Builder pattern:**
```go
tool := tools.New("my_tool", "Description").
    WithSchema(map[string]any{...}).
    ConcurrencySafe(true).ReadOnly(true).RemoteSafe().
    WithExecute(fn).Build()
```

**Interface:**
```go
type Tool interface {
    Name() string
    Description() string
    InputSchema() map[string]any
    Execute(ctx context.Context, input json.RawMessage, progress ProgressFunc) (*ToolResult, error)
    IsConcurrencySafe(input json.RawMessage) bool
    IsReadOnly(input json.RawMessage) bool
}
```

### Hook

Intercept the execution pipeline at five lifecycle phases:

| Phase | When | Capabilities |
|-------|------|-------------|
| `HookPreToolUse` | Before tool execution | Block execution, modify input |
| `HookPostToolUse` | After tool execution | Log results, inject messages |
| `HookPreModelCall` | Before API call | Block call, inject context |
| `HookPostModelCall` | After API response | Log usage, inspect response |
| `HookOnTurnEnd` | Loop would terminate | Force continuation via injected messages |

Hooks can implement `MultiPhaseHook` to fire at multiple phases without duplicate registration.

### Event Stream

17 event types covering the full agent lifecycle:

| Event | Description |
|-------|-------------|
| `EventTextDelta` | Streaming text chunk from the model |
| `EventThinkingDelta` | Reasoning/thinking content from the model |
| `EventToolStart` | Tool execution beginning |
| `EventToolProgress` | Incremental progress from a running tool |
| `EventToolEnd` | Tool execution completed (with result and duration) |
| `EventTurnStart` | New agentic loop iteration starting |
| `EventTurnEnd` | Loop iteration or entire agent run finished |
| `EventMessage` | Complete message added to conversation history |
| `EventUsage` | Token usage statistics for the turn |
| `EventBudgetWarning` | Token consumption crossed warning threshold |
| `EventCompaction` | Context history was compacted |
| `EventRetry` | Provider call being retried |
| `EventPermissionDenied` | Tool blocked by permission checker |
| `EventHookBlocked` | Hook blocked execution |
| `EventSubAgentStart` | Sub-agent spawned |
| `EventSubAgentEnd` | Sub-agent completed |
| `EventError` | Recoverable error (with retry indicator) |

Filter events for simpler consumption:

```go
for ev := range agentflow.FilterEvents(agent.Run(ctx, msgs), agentflow.EventTextDelta, agentflow.EventTurnEnd) {
    // only text deltas and turn ends
}
```

### Agent Cloning

Derive specialized agents from a common base:

```go
base := agentflow.NewAgent(provider, agentflow.WithMaxTurns(10), agentflow.WithTools(tools...))
researcher := base.Clone(agentflow.WithSystemPrompt("You are a researcher."))
writer := base.Clone(agentflow.WithSystemPrompt("You are a writer."))
```

## Execution Modes

```go
// Local mode (default) -- all tools available
agent := agentflow.NewAgent(provider,
    agentflow.WithTools(builtin.All()...),
    agentflow.WithExecutionMode(agentflow.ModeLocal),
)

// Remote mode -- only remote-safe tools
agent := agentflow.NewAgent(provider,
    agentflow.WithTools(builtin.All()...),
    agentflow.WithExecutionMode(agentflow.ModeRemote),
)
```

| Locality | ModeLocal | ModeRemote | Examples |
|----------|-----------|------------|----------|
| `ToolLocalOnly` | Allowed | Blocked | bash, read_file, write_file, edit_file, glob, grep |
| `ToolRemoteSafe` | Allowed | Allowed | http_request, web_search, deep_search |
| `ToolAny` | Allowed | Allowed | sleep, ask_user |

## Sub-Agent System

```go
// Single child
events := agent.SpawnChild(ctx, agentflow.SubAgentConfig{
    SystemPrompt: "You are a research specialist.",
    MaxTurns:     5,
}, "Research Go concurrency patterns")

// Parallel orchestration
results := agentflow.Orchestrate(ctx, agent, agentflow.SubAgentConfig{
    MaxTurns: 3,
}, []string{"Topic A", "Topic B", "Topic C"})
```

## Session Persistence

```go
import "github.com/CanArslanDev/agentflow/session/filestore"

store, _ := filestore.New("./sessions")
agent := agentflow.NewAgent(provider, agentflow.WithSessionStore(store))

session := &agentflow.Session{Metadata: map[string]any{"user": "alice"}}
for ev := range agent.RunSession(ctx, session, messages) { ... }

// Resume later
events, _ := agent.Resume(ctx, session.ID, "Continue please")
```

## Token Budget Management

```go
agent := agentflow.NewAgent(provider,
    agentflow.WithTokenBudget(agentflow.TokenBudget{
        MaxTokens:        100000,
        WarningThreshold: 0.8,
    }),
)
```

## Tool Result Management

```go
agent := agentflow.NewAgent(provider,
    agentflow.WithMaxResultSize(2000),
)
```

| Limiter | Behavior |
|---------|----------|
| `TruncateLimiter` (default) | Keeps 80% head + 20% tail |
| `HeadTailLimiter` | Configurable head/tail ratio |
| `NoLimiter` | Pass-through |

## Context Compaction

```go
import "github.com/CanArslanDev/agentflow/compactor"

// Sliding window: keep last N messages
agentflow.WithCompactor(compactor.NewSlidingWindow(20, 0))

// Token-aware: trigger based on estimated token count
agentflow.WithCompactor(compactor.NewTokenWindow(8000, 20))

// AI-powered: summarize older messages
agentflow.WithCompactor(compactor.NewSummary(provider, 20, 0))

// Staged: try light compaction first, then heavier
agentflow.WithCompactor(compactor.NewStaged(
    compactor.NewSlidingWindow(30, 40),
    compactor.NewSummary(provider, 10, 0),
))
```

## Input Validation

Tool inputs are automatically validated against their `InputSchema()` before execution. Supports type checking, required fields, enum values, nested objects, and array items.

```go
// Disable if schemas are informational only
agentflow.WithDisableInputValidation()
```

## Rate Limiting

```go
// 10 requests per second with burst of 10
agentflow.WithRateLimiter(agentflow.NewTokenBucketLimiter(10, time.Second))

// 60 requests per minute
agentflow.WithRateLimiter(agentflow.NewTokenBucketLimiter(60, time.Minute))
```

## Structured Logging

```go
agentflow.WithLogger(slog.Default())
```

Logs model calls (start/end with duration), retries, compaction, budget warnings, validation failures, and tool retries.

## Trace Context Propagation

The `observability.Tracer` automatically propagates W3C `traceparent` headers to provider HTTP requests. All providers forward `Request.Metadata` entries as HTTP headers.

```go
import "github.com/CanArslanDev/agentflow/observability"

tracer := observability.NewTracer()
for _, h := range tracer.Hooks() {
    agent = agentflow.NewAgent(provider, agentflow.WithHook(h))
}
// after run:
trace := tracer.Finish()
```

## Multi-Provider Fallback

```go
import "github.com/CanArslanDev/agentflow/provider/fallback"

provider := fallback.New(
    groq.New(groqKey, "llama-3.3-70b-versatile"),
    openrouter.New(orKey, "anthropic/claude-sonnet-4-20250514"),
)
```

## Multimodal Support

```go
msg := agentflow.NewImageMessage("What do you see?",
    agentflow.ImageContent{MediaType: "image/png", Data: base64String},
)
msg := agentflow.NewImageURLMessage("Describe this", "https://example.com/photo.jpg")
```

## Built-in Tools

12 ready-to-use tools in `tools/builtin`:

| Tool | Description | Locality | Concurrent | ReadOnly |
|------|-------------|----------|------------|----------|
| `bash` | Execute shell commands | LocalOnly | No | No |
| `read_file` | Read file contents with offset/limit | LocalOnly | Yes | Yes |
| `write_file` | Create or overwrite files | LocalOnly | No | No |
| `edit_file` | String replacement in files | LocalOnly | No | No |
| `list_dir` | List directory contents | LocalOnly | Yes | Yes |
| `glob` | Find files by pattern (supports **) | LocalOnly | Yes | Yes |
| `grep` | Search file contents | LocalOnly | Yes | Yes |
| `http_request` | HTTP requests (GET/POST/PUT/DELETE) | RemoteSafe | Yes | Yes |
| `web_search` | Web search via DuckDuckGo | RemoteSafe | Yes | Yes |
| `deep_search` | Multi-step web research (search + fetch + extract) | RemoteSafe | Yes | Yes |
| `sleep` | Pause execution | Any | Yes | Yes |
| `ask_user` | Prompt user for input (callback) | Any | No | Yes |

Registry presets:

```go
builtin.All()      // all 11 tools (ask_user requires separate registration)
builtin.Local()    // alias for All()
builtin.Remote()   // remote-safe: http_request, web_search, deep_search, sleep
builtin.ReadOnly() // only read-only tools
```

## Middleware

```go
import "github.com/CanArslanDev/agentflow/middleware"

// Structured logging
for _, h := range middleware.Logging(slog.Default()) {
    opts = append(opts, agentflow.WithHook(h))
}

// Metrics collection
metrics := middleware.NewMetrics()
for _, h := range metrics.Hooks() {
    opts = append(opts, agentflow.WithHook(h))
}

// Circuit breaker: block tool after 3 consecutive failures, reset after 30s
cb := middleware.NewCircuitBreaker(3, 30*time.Second)
for _, h := range cb.Hooks() {
    opts = append(opts, agentflow.WithHook(h))
}

// Tool timeout and retry (via config, not middleware)
agentflow.WithToolTimeout(30 * time.Second)
agentflow.WithToolRetries(2)

// Error strategy
agentflow.WithErrorStrategy(agentflow.ErrorStrategyFunc(
    func(call *agentflow.ToolCall, result *agentflow.ToolResult) (*agentflow.ToolResult, agentflow.ErrorAction) {
        if call.Name == "critical_tool" {
            return result, agentflow.ErrorActionAbort
        }
        return result, agentflow.ErrorActionDefault
    },
))
```

## Extension Packages

| Package | Import Path | Purpose |
|---------|------------|---------|
| `compactor` | `agentflow/compactor` | SlidingWindow, TokenWindow, Summary, Staged, ContextCollapser |
| `team` | `agentflow/team` | Multi-agent coordination with mailbox and shared memory |
| `observability` | `agentflow/observability` | Tracer (spans) and CostTracker (token pricing) |
| `trigger` | `agentflow/trigger` | Scheduled agent execution on intervals |
| `plan` | `agentflow/plan` | Plan(), PlanAndExecute(), ExtractMemories() |
| `skill` | `agentflow/skill` | Skill registry, execution, and YAML parsing |
| `task` | `agentflow/task` | Task store for agent work tracking |

## HTTP Server Deployment

The streaming example shows how to deploy agentflow as an HTTP API with SSE streaming:

```go
agent := agentflow.NewAgent(provider,
    agentflow.WithTools(builtin.Remote()...),
    agentflow.WithExecutionMode(agentflow.ModeRemote),
    agentflow.WithRateLimiter(agentflow.NewTokenBucketLimiter(10, time.Second)),
    agentflow.WithToolTimeout(30 * time.Second),
    agentflow.WithLogger(slog.Default()),
)

// SSE streaming endpoint
http.HandleFunc("/chat", func(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "text/event-stream")
    flusher := w.(http.Flusher)

    for ev := range agent.Run(r.Context(), messages) {
        data, _ := json.Marshal(eventToJSON(ev))
        fmt.Fprintf(w, "data: %s\n\n", data)
        flusher.Flush()
    }
})
```

See [`_examples/streaming/`](_examples/streaming/) for the full implementation with both SSE and synchronous JSON endpoints.

## Examples

| Example | Description |
|---------|-------------|
| [`basic`](_examples/basic/) | Minimal agent with a calculator tool |
| [`chat`](_examples/chat/) | Interactive chat loop with built-in tools |
| [`custom_tools`](_examples/custom_tools/) | File system agent with metrics and permissions |
| [`streaming`](_examples/streaming/) | HTTP SSE server with remote-safe tools |

## Testing

```bash
# Unit tests (no API key)
go test ./... -run "Test[^I]" -timeout 60s

# Integration tests (requires Groq API key)
GROQ_API_KEY=gsk-... go test ./... -run "TestIntegration_" -timeout 180s

# Benchmarks
go test -bench=. -benchmem -run="^$"

# Race detector
go test -run "TestRace_" -race -timeout 30s
```

## Project Structure

```
agentflow/
    -- Core (17 source files) --
    agent.go              Agent, Run(), RunSync(), RunSession(), Resume(), Clone()
    config.go             Config, 22 WithXxx() options
    tool.go               Tool interface, ExecutionMode, ToolLocality
    message.go            Message, ContentBlock, multimodal, documents
    event.go              17 EventTypes, FilterEvents()
    hook.go               Hook, MultiPhaseHook, HookFunc
    permission.go         PermissionChecker, AllowAll, DenyList, AllowList, Chain
    provider.go           Provider, Stream, HealthChecker
    ratelimit.go          RateLimiter, TokenBucketLimiter
    subagent.go           SpawnChild, SpawnChildren, Orchestrate
    session.go            SessionStore, Session
    budget.go             TokenBudget, budgetTracker
    result.go             ResultLimiter, TruncateLimiter, HeadTailLimiter
    compactor.go          Compactor interface
    streaming_executor.go Overlapping model gen with tool exec
    errors.go             ProviderError, ToolError, ErrorStrategy
    doc.go                Package documentation

    -- Extension packages --
    compactor/            SlidingWindow, TokenWindow, Summary, Staged
    team/                 Team coordination, Mailbox, SharedMemory
    observability/        Tracer, CostTracker
    trigger/              Scheduled execution
    plan/                 Plan mode, memory extraction
    skill/                Skill registry
    task/                 Task store

    -- Providers --
    provider/openai/      OpenAI (uses internal/sse)
    provider/anthropic/   Anthropic (custom SSE parser)
    provider/gemini/      Google Gemini (custom SSE parser)
    provider/groq/        Groq (uses internal/sse)
    provider/openrouter/  OpenRouter (uses internal/sse)
    provider/ollama/      Ollama/RunPod (JSONL streaming)
    provider/fallback/    Multi-provider failover
    provider/mock/        Deterministic mock

    -- Internal --
    internal/sse/         Shared SSE parser
    internal/jsonschema/  JSON Schema validator

    -- Infrastructure --
    session/filestore/    File-based session store
    session/memstore/     In-memory session store
    middleware/           Logging, metrics, recovery, circuit breaker
    tools/builder.go      Fluent ToolBuilder
    tools/typed.go        Generic TypedTool[I]
    tools/builtin/        12 built-in tools

    -- Examples --
    _examples/basic/      Minimal agent
    _examples/chat/       Interactive chat
    _examples/custom_tools/ Multi-tool with metrics
    _examples/streaming/  HTTP SSE server
```

## License

MIT -- see [LICENSE](LICENSE).
