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
- [Multi-Provider Fallback](#multi-provider-fallback)
- [Multimodal Support](#multimodal-support)
- [Built-in Tools](#built-in-tools)
- [Middleware](#middleware)
- [Examples](#examples)
- [Testing](#testing)
- [License](#license)

## Features

- **Provider-agnostic** -- works with Groq, OpenRouter, OpenAI, Anthropic, or any custom provider
- **Streaming-first** -- real-time event delivery via Go channels
- **Execution modes** -- ModeLocal for full access, ModeRemote for server-safe tools only
- **Sub-agent system** -- spawn child agents, parallel orchestration, delegate_task tool
- **Session persistence** -- save/resume conversations with file-based or in-memory stores
- **Token budget** -- enforce consumption limits with warning thresholds
- **Concurrent tool execution** -- safe tools run in parallel, unsafe tools run in isolation
- **Hook system** -- intercept any lifecycle phase (pre/post tool, pre/post model call)
- **Permission control** -- AllowAll, DenyList, AllowList, ReadOnly, Chain, or custom checkers
- **Result size management** -- automatic truncation of oversized tool outputs
- **Context compaction** -- sliding window, token-aware, and AI-powered summarization
- **Multi-provider fallback** -- automatic failover across providers
- **Multimodal** -- send images (base64 or URL) alongside text
- **11 built-in tools** -- bash, file ops, search, HTTP, web search, and more
- **Panic recovery** -- tool panics are caught and converted to error results
- **Retry logic** -- exponential backoff for transient provider errors
- **Zero external dependencies** -- core uses only the Go standard library
- **Metrics built-in** -- middleware for structured logging and metrics collection

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
    "encoding/json"
    "fmt"

    "github.com/CanArslanDev/agentflow"
    "github.com/CanArslanDev/agentflow/provider/groq"
    "github.com/CanArslanDev/agentflow/tools"
)

func main() {
    provider := groq.New("gsk-...", "llama-3.3-70b-versatile")

    calculator := tools.New("calculator", "Evaluate a math expression").
        WithSchema(map[string]any{
            "type": "object",
            "properties": map[string]any{
                "expression": map[string]any{"type": "string"},
            },
            "required": []string{"expression"},
        }).
        ConcurrencySafe(true).
        ReadOnly(true).
        WithExecute(func(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
            var p struct{ Expression string `json:"expression"` }
            json.Unmarshal(input, &p)
            return &agentflow.ToolResult{Content: "42"}, nil
        }).
        Build()

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
                          <-chan Event
                                |
    +---------------------------+---------------------------+
    |                    AGENTIC LOOP                       |
    |                                                       |
    |   +----------+     +----------+     +----------+      |
    |   | PreModel |---->| Provider |---->| PostModel|      |
    |   |  Hooks   |     | Stream   |     |  Hooks   |      |
    |   +----------+     +----+-----+     +----------+      |
    |                         |                             |
    |                  Tool Calls?                          |
    |                  /          \                         |
    |                Yes          No --> return (complete)  |
    |                /                                      |
    |   +------------+                                      |
    |   |  Executor  |                                      |
    |   |  Pipeline: |                                      |
    |   |  Validate  |                                      |
    |   |  PreHooks  |                                      |
    |   |  Permission|                                      |
    |   |  Execute   |                                      |
    |   |  PostHooks |                                      |
    |   +-----+------+                                      |
    |         |                                             |
    |   Append Results --> continue (next iteration)        |
    +-------------------------------------------------------+
```

The loop follows a think, act, observe, repeat cycle:

1. Call the AI model with conversation history and tool definitions
2. Stream the response and emit text deltas in real time
3. If the model requests tool calls, execute them (parallel when safe)
4. Append tool results to history and loop back to step 1
5. When the model responds without tool calls, the loop completes

## Core Concepts

### Provider

Abstracts the AI model API. Implement the Provider interface to add new backends:

```go
type Provider interface {
    CreateStream(ctx context.Context, req *Request) (Stream, error)
    ModelID() string
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
| Fallback | `provider/fallback` | Cascading failover across any providers | N/A |
| Mock | `provider/mock` | Deterministic testing without API calls | N/A |

Provider examples:

```go
// OpenAI
provider := openai.New("sk-...", "gpt-4o")

// Anthropic Claude
provider := anthropic.New("sk-ant-...", "claude-sonnet-4-20250514")

// Google Gemini
provider := gemini.New("AIza...", "gemini-2.0-flash")

// Groq
provider := groq.New("gsk_...", "llama-3.3-70b-versatile")

// OpenRouter (any model)
provider := openrouter.New("sk-or-...", "anthropic/claude-sonnet-4-20250514")

// Fallback: try OpenAI first, then Groq
provider := fallback.New(
    openai.New("sk-...", "gpt-4o"),
    groq.New("gsk_...", "llama-3.3-70b-versatile"),
)
```

### Tool

A capability the agent can invoke. Use the fluent builder or implement the interface directly:

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

### Permission

Control tool access with composable checkers:

```go
agentflow.AllowAll()                          // permit everything
agentflow.DenyList("write_file", "bash")      // block specific tools
agentflow.AllowList("read_file", "search")    // allow only these
agentflow.ReadOnlyPermission()                // allow only read-only tools
agentflow.ChainPermission(checker1, checker2) // first non-Allow wins
```

### Event Stream

Every action in the loop emits an Event through the channel returned by `Agent.Run()`:

| Event | Description |
|-------|-------------|
| `EventTextDelta` | Streaming text chunk from the model |
| `EventThinkingDelta` | Reasoning/thinking content from the model |
| `EventToolStart` | Tool execution beginning |
| `EventToolProgress` | Incremental progress from a running tool |
| `EventToolEnd` | Tool execution completed (with result and duration) |
| `EventTurnStart` | New agentic loop iteration starting |
| `EventTurnEnd` | Loop iteration or entire agent run finished |
| `EventUsage` | Token usage statistics for the turn |
| `EventBudgetWarning` | Token consumption crossed warning threshold |
| `EventSubAgentStart` | Sub-agent spawned |
| `EventSubAgentEnd` | Sub-agent completed |
| `EventError` | Recoverable error (with retry indicator) |

### Concurrency Model

Tools declare their concurrency safety per invocation. The executor partitions tool calls into batches:

```
Tool calls: [Read, Read, Bash, Read]
                      |
                  partition
                      |
Batch 1 (parallel): [Read, Read]   <-- goroutines + semaphore
Batch 2 (serial):   [Bash]         <-- exclusive execution
Batch 3 (parallel): [Read]         <-- goroutine
```

Configure max parallelism with `WithMaxConcurrency(n)`. Default is 10.

## Execution Modes

agentflow supports two execution modes that control which tools are visible to the model. In ModeRemote, local-only tools are completely filtered from the tool definitions sent to the AI -- the model does not know they exist and cannot attempt to call them.

```go
// Local mode (default) -- all tools available
agent := agentflow.NewAgent(provider,
    agentflow.WithTools(builtin.All()...),
    agentflow.WithExecutionMode(agentflow.ModeLocal),
)

// Remote mode -- only remote-safe tools are sent to the model
agent := agentflow.NewAgent(provider,
    agentflow.WithTools(builtin.All()...),
    agentflow.WithExecutionMode(agentflow.ModeRemote),
)
```

Tool locality is declared via the optional `LocalityAware` interface:

| Locality | ModeLocal | ModeRemote | Examples |
|----------|-----------|------------|----------|
| `ToolLocalOnly` | Allowed | Blocked | bash, read_file, write_file, edit_file, glob, grep |
| `ToolRemoteSafe` | Allowed | Allowed | http_request, web_search |
| `ToolAny` | Allowed | Allowed | sleep, ask_user |

Tools that do not implement `LocalityAware` default to `ToolLocalOnly` (safe default).

Custom tools declare their locality with the builder:

```go
myTool := tools.New("api_call", "Call an external API").
    RemoteSafe().
    WithExecute(fn).
    Build()
```

## Sub-Agent System

Agents can spawn child agents for task delegation and parallel work:

```go
// Single child agent
events := agent.SpawnChild(ctx, agentflow.SubAgentConfig{
    SystemPrompt: "You are a research specialist.",
    MaxTurns:     5,
}, "Research Go concurrency patterns")

for ev := range events {
    // handle child events
}

// Parallel orchestration -- fan-out/fan-in
results := agentflow.Orchestrate(ctx, agent, agentflow.SubAgentConfig{
    SystemPrompt: "Answer concisely.",
    MaxTurns:     3,
}, []string{
    "Research topic A",
    "Research topic B",
    "Research topic C",
})

for _, r := range results {
    fmt.Printf("%s -> %s\n", r.Task, r.Result)
}
```

The `SubAgentTool` lets the model itself decide when to delegate:

```go
agent := agentflow.NewAgent(provider,
    agentflow.WithTools(
        agentflow.SubAgentTool(provider, "You are a researcher.", 5),
    ),
)
```

## Session Persistence

Save and resume conversations across restarts:

```go
import "github.com/CanArslanDev/agentflow/session/filestore"

store, _ := filestore.New("./sessions")
agent := agentflow.NewAgent(provider,
    agentflow.WithSessionStore(store),
)

// First run -- auto-saves after each turn
session := &agentflow.Session{
    Metadata: map[string]any{"user": "alice"},
}
for ev := range agent.RunSession(ctx, session, messages) {
    // handle events
}
// session.ID is now set

// Later -- resume from where you left off
events, _ := agent.Resume(ctx, session.ID, "Continue please")
for ev := range events {
    // handle events
}
```

Built-in stores:

| Store | Package | Use Case |
|-------|---------|----------|
| `filestore` | `session/filestore` | JSON files on disk, single-process agents |
| `memstore` | `session/memstore` | In-memory, testing and short-lived apps |

## Token Budget Management

Enforce token consumption limits to control costs:

```go
agent := agentflow.NewAgent(provider,
    agentflow.WithTokenBudget(agentflow.TokenBudget{
        MaxTokens:        100000,  // total limit across all turns
        WarningThreshold: 0.8,     // emit EventBudgetWarning at 80%
    }),
)
```

When the budget is exhausted, the loop terminates with `TurnEndBudgetExhausted`. The warning event fires exactly once when the threshold is crossed.

## Tool Result Management

Prevent oversized tool outputs from exhausting the context window:

```go
agent := agentflow.NewAgent(provider,
    agentflow.WithMaxResultSize(2000),  // truncate results over 2000 chars
)
```

Built-in limiters:

| Limiter | Behavior |
|---------|----------|
| `TruncateLimiter` (default) | Keeps 80% head + 20% tail, inserts truncation notice |
| `HeadTailLimiter` | Configurable head/tail ratio |
| `NoLimiter` | Pass-through, no size enforcement |

Error results are never truncated -- the model needs full error context.

## Context Compaction

Manage growing conversation history for long-running agents:

```go
import "github.com/CanArslanDev/agentflow/compactor"

// Simple: keep last N messages, discard older ones
agent := agentflow.NewAgent(provider,
    agentflow.WithCompactor(compactor.NewSlidingWindow(20, 0)),
)

// Token-aware: trigger compaction based on estimated token count
agent := agentflow.NewAgent(provider,
    agentflow.WithCompactor(compactor.NewTokenWindow(8000, 20)),
)

// AI-powered: summarize older messages using a provider
agent := agentflow.NewAgent(provider,
    agentflow.WithCompactor(compactor.NewSummary(provider, 20, 0)),
)
```

The first message (initial user prompt) is always preserved. A system notice is inserted to inform the model about compacted context.

## Multi-Provider Fallback

Automatic failover when a provider is unavailable:

```go
import "github.com/CanArslanDev/agentflow/provider/fallback"

provider := fallback.New(
    groq.New(groqKey, "llama-3.3-70b-versatile"),
    openrouter.New(orKey, "anthropic/claude-sonnet-4-20250514"),
)

agent := agentflow.NewAgent(provider)
```

The fallback provider tries each provider in order. Only retryable errors (429, 5xx, network failures) trigger a fallback. Non-retryable errors (401, 400) stop immediately.

## Multimodal Support

Send images alongside text for vision-capable models:

```go
// Base64 image
msg := agentflow.NewImageMessage("What do you see?",
    agentflow.ImageContent{
        MediaType: "image/png",
        Data:      base64EncodedString,
    },
)

// URL image
msg := agentflow.NewImageURLMessage("Describe this photo", "https://example.com/photo.jpg")

// Multiple images
msg := agentflow.NewImageMessage("Compare these",
    agentflow.ImageContent{MediaType: "image/jpeg", Data: img1},
    agentflow.ImageContent{MediaType: "image/png", Data: img2},
)
```

## Built-in Tools

Ready-to-use tools in `tools/builtin`:

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
| `sleep` | Pause execution | Any | Yes | Yes |
| `ask_user` | Prompt user for input (callback) | Any | No | Yes |

Registry presets:

```go
builtin.All()      // all 10 tools (ask_user requires separate registration)
builtin.Local()    // alias for All()
builtin.Remote()   // only remote-safe: http_request, web_search, sleep
builtin.ReadOnly() // only read-only tools
```

## Middleware

```go
import "github.com/CanArslanDev/agentflow/middleware"

// Structured logging with slog
for _, h := range middleware.Logging(slog.Default()) {
    agent = agentflow.NewAgent(provider, agentflow.WithHook(h))
}

// Metrics collection
metrics := middleware.NewMetrics()
for _, h := range metrics.Hooks() {
    agent = agentflow.NewAgent(provider, agentflow.WithHook(h))
}
snap := metrics.Snapshot()
fmt.Printf("Total calls: %d, errors: %d\n", snap.TotalCalls, snap.TotalErrors)

// Turn limit warning
hook := middleware.MaxTurnsGuard(20, logger)
```

## Synchronous Mode

For non-streaming use cases:

```go
messages, err := agent.RunSync(ctx, initialMessages)
if err != nil {
    log.Fatal(err)
}
fmt.Println(messages[len(messages)-1].TextContent())
```

## Examples

See the [`_examples`](_examples/) directory:

| Example | Description |
|---------|-------------|
| [`basic`](_examples/basic/) | Minimal agent with a calculator tool |
| [`custom_tools`](_examples/custom_tools/) | File system agent with metrics and permissions |
| [`streaming`](_examples/streaming/) | HTTP SSE endpoint serving an agent |

## Testing

Run unit tests (no API key required):

```bash
go test ./... -run "Test[^I]"
```

Run integration tests (requires Groq API key):

```bash
GROQ_API_KEY=gsk-... go test ./... -run "TestIntegration_" -timeout 180s
```

Run all tests:

```bash
GROQ_API_KEY=gsk-... go test ./... -timeout 180s
```

## Project Structure

```
agentflow/
    agent.go                 -- Agent, agentic loop, tool executor
    config.go                -- Config, Option functions
    tool.go                  -- Tool interface, ExecutionMode, ToolLocality
    message.go               -- Message, ContentBlock, multimodal support
    event.go                 -- Event types emitted by the loop
    hook.go                  -- Hook interface, HookFunc adapter
    permission.go            -- PermissionChecker, built-in checkers
    provider.go              -- Provider, Stream, StreamEvent interfaces
    subagent.go              -- SpawnChild, SpawnChildren, Orchestrate
    session.go               -- SessionStore interface, Session struct
    budget.go                -- TokenBudget, budget tracking
    result.go                -- ResultLimiter, TruncateLimiter
    compactor.go             -- Compactor interface
    compactor_sliding.go     -- SlidingWindow, TokenWindow compactors
    compactor_summary.go     -- AI-powered SummaryCompactor
    errors.go                -- Sentinel errors, ProviderError, ToolError

    provider/
        openai/              -- OpenAI Chat Completions API
        anthropic/           -- Anthropic Messages API (different format than OpenAI)
        gemini/              -- Google Gemini generateContent API
        groq/                -- Groq API (OpenAI-compatible)
        openrouter/          -- OpenRouter API (OpenAI-compatible, all models)
        fallback/            -- Multi-provider cascading failover
        mock/                -- Deterministic mock for testing

    internal/
        sse/                 -- Shared OpenAI-compatible SSE parser

    session/
        filestore/           -- JSON file-based session store
        memstore/            -- In-memory session store

    middleware/
        logging.go           -- Structured slog logging hooks
        metrics.go           -- Thread-safe metrics collection
        recovery.go          -- Panic recovery, turn limit guard

    tools/
        builder.go           -- Fluent ToolBuilder API
        builtin/             -- 11 ready-to-use tools

    _examples/               -- Usage examples
```

## License

MIT -- see [LICENSE](LICENSE).
