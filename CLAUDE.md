# CLAUDE.md

This file provides guidance to Claude Code when working with the agentflow codebase.

## Project Overview

agentflow is a standalone, open-source Go framework for building agentic AI systems. It provides the core abstractions and orchestration logic for AI agents that autonomously invoke tools, observe results, and continue reasoning in a loop.

The project has zero external dependencies in its core -- it uses only the Go standard library. Provider packages may depend on their respective SDKs.

## Essential Commands

```bash
# Build all packages
go build ./...

# Run all unit tests (no API key required)
go test ./... -run "Test[^I]" -timeout 60s

# Run integration tests (requires GROQ_API_KEY)
GROQ_API_KEY=gsk-... go test ./... -run "TestIntegration_" -timeout 180s

# Run benchmarks
go test -bench=. -benchmem -run="^$" -timeout 60s

# Run with race detector (known pre-existing races in streaming executor)
go test -run "TestRace_" -race -timeout 30s

# Static analysis
go vet ./...
```

## Architecture

### Core Loop (agent.go)

The agentic loop is a `for {}` in `runLoopWithState()`:

```
for {
    1. Check context cancellation
    2. Check turn limit
    3. Check token budget
    4. Run compaction if needed -> emit EventCompaction
    5. Execute pre-model hooks -> emit EventHookBlocked if blocked
    6. Build Request (skip tools if provider rejected them in a previous turn)
    7. Apply rate limiter (Wait)
    8. Call provider.CreateStream() with retry -> emit EventRetry on retries
       - On "tool calling not supported" error: drop tools, retry without them
       - On "context too large" error: auto-compact if compactor set, retry
    9. Consume stream: emit text deltas, start tool execution via streaming executor
       - Tool calls emitted mid-stream when JSON arguments become valid (early emission)
       - Missing tool IDs auto-generated as "tool-call-{index}"
   10. Record usage, enforce budget -> emit EventBudgetWarning
   11. Execute post-model hooks
   12. If no tool calls -> check turn-end hooks -> return (completed)
   13. Execute tools through pipeline:
       a. Tool lookup with fuzzy matching (suffix + substring for shortened names)
       b. Execution mode guard -- blocks local-only tools in remote mode
       c. Pre-tool hooks -> emit EventHookBlocked if blocked
       d. Permission check -> emit EventPermissionDenied if denied
       e. JSON Schema input validation with descriptive repair messages
       f. Execute with timeout + retry; panic recovery marks critical errors
       g. Apply ErrorStrategy if tool returns error
       h. Apply result size limiting
       i. Post-tool hooks
   14. Ensure every tool call has a result (fill missing with error results)
   15. Append tool results to messages
   16. Check for critical errors (tool panics) -> terminate if found
   17. Loop detection: SHA-256 signature of (tool+input+output), terminate if >5 repeats
   18. Continue loop
}
```

The loop runs in a dedicated goroutine launched by `Agent.Run()`. Events are delivered through a buffered channel. When the loop terminates, it signals background goroutines (streaming executor) via a done channel and waits for them to finish before closing the events channel.

### Layered Design

```
Public API (agent.go, config.go)
    |
Core Types (tool.go, message.go, event.go, hook.go, permission.go, provider.go)
    |
Internal (internal/sse/ -- shared SSE parsing, internal/jsonschema/ -- input validation)
    |
Providers (provider/openai, provider/anthropic, provider/gemini, provider/groq, provider/openrouter, provider/ollama, provider/fallback, provider/mock)
    |
Extension Packages (compactor/, team/, observability/, trigger/, plan/, skill/, task/)
    |
Middleware (middleware/ -- logging, metrics, recovery, circuit breaker)
    |
Tools (tools/builder.go, tools/typed.go, tools/builtin/)
```

### Key Interfaces

| Interface | File | Purpose |
|-----------|------|---------|
| `Provider` | provider.go | AI model API abstraction |
| `Stream` | provider.go | Pull-based SSE event stream |
| `HealthChecker` | provider.go | Optional: proactive provider health check |
| `Tool` | tool.go | Agent capability (function calling) |
| `LocalityAware` | tool.go | Optional: declare local/remote safety |
| `TimeoutAware` | tool.go | Optional: tool-specific execution timeout |
| `Hook` | hook.go | Single-phase lifecycle interception |
| `MultiPhaseHook` | hook.go | Multi-phase lifecycle interception |
| `PermissionChecker` | permission.go | Tool access control |
| `Compactor` | compactor.go | Conversation history management |
| `SessionStore` | session.go | Conversation persistence |
| `ResultLimiter` | result.go | Oversized result handling |
| `RateLimiter` | ratelimit.go | Provider API call rate control |
| `ErrorStrategy` | errors.go | Configurable tool error handling |

### Config Options (config.go)

All options are set via functional `Option` values passed to `NewAgent`:

| Option | Purpose |
|--------|---------|
| `WithTool(t)` / `WithTools(t...)` | Register tools |
| `WithHook(h)` | Register lifecycle hook |
| `WithPermission(p)` | Set permission checker |
| `WithMaxTurns(n)` | Limit loop iterations |
| `WithMaxConcurrency(n)` | Parallel tool execution limit |
| `WithSystemPrompt(s)` | System message for every model call |
| `WithTemperature(t)` | Model temperature |
| `WithMaxTokens(n)` | Max response tokens |
| `WithRetryPolicy(p)` | Provider retry with exponential backoff |
| `WithEventBufferSize(n)` | Event channel capacity |
| `WithOnEvent(fn)` | Synchronous event callback |
| `WithCompactor(c)` | Context compaction strategy |
| `WithTokenBudget(b)` | Total token consumption limit |
| `WithExecutionMode(m)` | Local vs remote tool filtering |
| `WithResultLimiter(l)` | Custom result size limiter |
| `WithMaxResultSize(n)` | Max characters per tool result |
| `WithSessionStore(s)` | Enable session persistence |
| `WithErrorStrategy(s)` | Custom tool error handling (abort/transform) |
| `WithToolRetries(n)` | Retry failed tool executions N times |
| `WithToolTimeout(d)` | Cancel tool execution after duration |
| `WithRateLimiter(l)` | Rate limit provider API calls |
| `WithLogger(l)` | Structured logging (slog.Logger) |
| `WithDisableInputValidation()` | Skip JSON Schema validation |
| `WithThinkingPrompt(t,a)` | Agentic thinking for non-native models |

### Event Types (event.go)

17 event types covering the full agent lifecycle:

| Event | When | Key Fields |
|-------|------|------------|
| `EventTextDelta` | Streaming text from model | Text |
| `EventThinkingDelta` | Streaming reasoning content | Text |
| `EventToolStart` | Tool execution begins | ToolCall |
| `EventToolProgress` | Tool progress update | ToolCallID, Message, Data |
| `EventToolEnd` | Tool execution completes | ToolCall, Result, Duration |
| `EventTurnStart` | New loop iteration | TurnNumber |
| `EventTurnEnd` | Loop iteration or loop ends | TurnNumber, Reason, Messages |
| `EventMessage` | Message added to history | Message |
| `EventError` | Recoverable error | Err, Retrying, TurnCount |
| `EventUsage` | Token usage stats | Usage, TurnCount |
| `EventSubAgentStart` | Sub-agent spawned | Index, Task |
| `EventSubAgentEnd` | Sub-agent completed | Index, Task, Result |
| `EventBudgetWarning` | Token budget threshold crossed | ConsumedTokens, MaxTokens, Percentage |
| `EventCompaction` | Context was compacted | BeforeCount, AfterCount, TurnCount |
| `EventRetry` | Provider call being retried | Attempt, Delay, Err, TurnCount |
| `EventPermissionDenied` | Tool blocked by permissions | ToolCall |
| `EventHookBlocked` | Hook blocked execution | Phase, ToolCall, Reason, TurnCount |

Use `FilterEvents(ch, types...)` to consume only specific event types.

## Code Conventions

### General Rules

- Go 1.23+, standard library only in core packages
- All exported types and functions must have godoc comments
- No emojis in code or documentation
- English only for all code, comments, and documentation
- Error messages start with lowercase, no trailing punctuation

### Patterns Used

- **Functional options**: `NewAgent(provider, WithTool(...), WithMaxTurns(10))`
- **Optional interfaces**: `LocalityAware`, `MultiPhaseHook`, `HealthChecker` extend base interfaces without breaking existing implementations
- **Discriminated unions**: `Event.Type` determines which field is set
- **Pull-based streams**: `Stream.Next()` returns `(StreamEvent, error)`, `io.EOF` signals completion
- **Channel-based event delivery**: `Agent.Run()` returns `<-chan Event`
- **Generic typed tools**: `tools.NewTyped[I]()` for type-safe tool implementations

### Tool Implementation Patterns

**Interface-based (full control):**
```go
type myTool struct{}

func (t *myTool) Name() string                                    { return "my_tool" }
func (t *myTool) Description() string                              { return "..." }
func (t *myTool) InputSchema() map[string]any                      { return map[string]any{...} }
func (t *myTool) IsConcurrencySafe(_ json.RawMessage) bool         { return true }
func (t *myTool) IsReadOnly(_ json.RawMessage) bool                { return true }
func (t *myTool) Locality() agentflow.ToolLocality                 { return agentflow.ToolRemoteSafe }
func (t *myTool) Execute(ctx context.Context, input json.RawMessage, progress agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
    // implementation
}
```

**Builder pattern (less boilerplate):**
```go
tool := tools.New("my_tool", "Description").
    WithSchema(map[string]any{...}).
    ConcurrencySafe(true).ReadOnly(true).RemoteSafe().
    WithExecute(func(ctx context.Context, input json.RawMessage, p agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
        // implementation
    }).Build()
```

**Generic typed (type-safe, auto schema):**
```go
type SearchInput struct {
    Query      string `json:"query" description:"Search query"`
    MaxResults int    `json:"max_results,omitempty" description:"Max results"`
}

tool := tools.NewTyped[SearchInput](
    "search", "Search the web",
    []string{"query"}, // required fields
    func(ctx context.Context, input SearchInput, p agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
        // input.Query is already typed -- no json.Unmarshal needed
    },
)
```

### Provider Implementation Patterns

**OpenAI-compatible providers** (OpenAI, Groq, OpenRouter) use the shared `internal/sse` package:

```go
func (p *Provider) CreateStream(ctx context.Context, req *agentflow.Request) (agentflow.Stream, error) {
    body := sse.BuildRequestBody(p.model, req)  // shared request conversion
    // ... HTTP request with Authorization: Bearer header ...
    // Propagate request metadata as HTTP headers:
    for k, v := range req.Metadata {
        httpReq.Header.Set(k, v)
    }
    return sse.NewStream(resp), nil              // shared SSE parser
}
```

**Anthropic** has its own format (tool_use content blocks, x-api-key header, different SSE events):
- Custom stream parser in `provider/anthropic/stream.go`
- Custom message conversion in `provider/anthropic/anthropic.go`
- Auth via `x-api-key` header (not `Authorization: Bearer`)
- Requires `anthropic-version` header
- Supports `thinking` content blocks: `thinking_delta` emitted as `StreamEventThinkingDelta`
- Tool call JSON validated with `json.Valid()` before emission

**Gemini** has its own format (functionCall parts, API key in URL):
- Custom stream parser in `provider/gemini/gemini.go`
- Custom message conversion with `contents` array and `parts`
- Auth via `?key=` URL parameter (not header)
- Uses `generativelanguage.googleapis.com` base URL
- Multi-part handling: pending events queue for chunks with multiple parts
- Tool result name mapping: looks up tool name from message history (Google requires name, not call ID)
- Tool call JSON validated with `json.Valid()` before emission

**Ollama** has its own format (JSONL streaming, no SSE):
- Custom JSONL stream parser in `provider/ollama/stream.go`
- No auth header (RunPod uses pod ID in URL)
- Streaming format: one JSON object per line (not `data: ` prefixed SSE)
- Tool calls arrive in the final message (not streamed as deltas)
- Temperature/max_tokens via `options` object (`num_predict` = max_tokens)
- Images via `images` array (base64 strings, no data URI prefix)
- Documents fallback to text for text MIME types, placeholder for binary
- Implements `HealthChecker` via `/api/tags` endpoint

All providers propagate `Request.Metadata` as HTTP headers for trace context propagation.

### Input Validation (internal/jsonschema)

Tool inputs are validated against `tool.InputSchema()` before execution:
- Supports: `type` (object, string, integer, number, boolean, array), `required`, `enum`, `properties`, `additionalProperties`, `items`
- Validation errors are returned to the model as `ToolResult{IsError: true}` so it can self-correct
- Disable with `WithDisableInputValidation()` if schemas are informational only
- Validation runs after permission check, before tool execution

### Rate Limiting (ratelimit.go)

- `RateLimiter` interface: `Wait(ctx) error`
- `TokenBucketLimiter`: token bucket with burst capacity and steady refill rate
- Applied before every `CreateStream` call (including retries)
- Example: `NewTokenBucketLimiter(10, time.Second)` = 10 req/sec with burst of 10

### Structured Logging

When `WithLogger(slog.Logger)` is set, the agent logs:
- Model call start/completion (turn, message count, duration, tool call count)
- Context compaction (before/after message count)
- Token budget warnings (consumed, max, percentage)
- Provider retry attempts (attempt number, delay, error)
- Tool input validation failures (tool name, error)
- Tool execution retries (tool name, attempt, max attempts)

Logger is nil by default (zero overhead). Does not conflict with middleware logging.

### Trace Context Propagation

Hooks can propagate trace context to provider HTTP requests:
1. Hook writes `hc.Metadata["request:traceparent"] = "00-traceID-spanID-01"`
2. Agent copies `request:*` prefixed metadata into `Request.Metadata`
3. Provider sends `Request.Metadata` entries as HTTP headers
4. The `observability.Tracer` does this automatically via its PreModelCall hook

### Middleware

| Middleware | File | Pattern |
|-----------|------|---------|
| `Logging(logger)` | logging.go | PreToolUse + PostToolUse hooks; logs tool start/end with duration |
| `NewMetrics()` | metrics.go | PreToolUse + PostToolUse + PreModelCall; atomic counters, per-tool stats |
| `Recovery(logger)` | recovery.go | PostToolUse; structured logging of tool panic stack traces |
| `MaxTurnsGuard(n, logger)` | recovery.go | PreModelCall; warns at 80% of turn limit |
| `NewCircuitBreaker(threshold, reset)` | circuitbreaker.go | PreToolUse + PostToolUse; blocks tool after N consecutive failures, half-open recovery |

### Tool Execution Pipeline

Tool execution in `executeSingleTool()` follows this pipeline:

1. **Tool lookup** -- exact match first, then fuzzy (suffix + substring). On failure, error includes available tool names so the model can self-correct
2. **Execution mode guard** -- blocks local-only tools in remote mode
3. **Pre-tool hooks** -- can block or modify input; emits `EventHookBlocked`
4. **Permission check** -- can deny; emits `EventPermissionDenied`
5. **Input validation** -- JSON Schema check; on failure, returns full schema with field descriptions and required fields so the model can fix and retry (tool call repair)
6. **Execute** -- with `context.WithTimeout` if `ToolTimeout` set, with retry loop if `ToolRetries` > 0; panic recovery catches crashes and marks them critical
7. **Error strategy** -- `ErrorStrategy.OnToolError()` can transform or abort
8. **Result limiting** -- `ResultLimiter.Limit()` if result exceeds `MaxResultChars`
9. **Post-tool hooks** -- metrics, logging, observability
10. **Result guarantee** -- after all tools complete, verify every tool call has a result; fill missing with error results to prevent model desynchronization

### Execution Mode Safety

- Tools default to `ToolLocalOnly` if they do not implement `LocalityAware` (safe default)
- In `ModeRemote`, local-only tools are filtered from `toolDefinitions()` -- the model never sees them
- A guard in `executeSingleTool()` blocks local tools even if called by name (defense in depth)
- New built-in tools must declare their `Locality()` explicitly

### Streaming Tool Executor

`streaming_executor.go` overlaps model generation with tool execution:
- Tools are submitted for execution as they arrive from the stream (not waiting for stream end)
- Background goroutines are tracked via `loopState.bgWork` WaitGroup
- When the loop terminates, `state.done` channel is closed, cancelling in-flight tool goroutines
- `bgWork.Wait()` ensures all goroutines finish before the events channel is closed

### Agent Cloning

`Agent.Clone(opts...)` creates a copy of the agent with overridden options:
```go
base := agentflow.NewAgent(provider, agentflow.WithMaxTurns(10))
researcher := base.Clone(agentflow.WithSystemPrompt("You are a researcher."))
writer := base.Clone(agentflow.WithSystemPrompt("You are a writer."))
```

### Budget Tracking

- `budgetTracker` uses atomic operations for thread safety
- Warning fires exactly once via `CompareAndSwap`
- Budget check happens after each `stream.Usage()` in the loop
- When exhausted, the last assistant message is preserved before terminating

## Directory Structure

```
agentflow/
    -- Core package (root): 17 source files --
    agent.go                 # Agent struct, Run(), RunSync(), RunSession(), Resume(), Clone()
    config.go                # Config struct, 22 WithXxx() option functions
    tool.go                  # Tool interface, ExecutionMode, ToolLocality, LocalityAware
    message.go               # Message, ContentBlock (text, tool_call, tool_result, image)
    event.go                 # 17 EventTypes, Event discriminated union, FilterEvents()
    hook.go                  # Hook, MultiPhaseHook, HookFunc adapter, HookContext, HookAction
    permission.go            # PermissionChecker, AllowAll, DenyList, AllowList, Chain
    provider.go              # Provider, Stream, StreamEvent, Usage, HealthChecker, IsHealthy
    ratelimit.go             # RateLimiter interface, TokenBucketLimiter
    subagent.go              # SubAgentConfig, SpawnChild, SpawnChildren, Orchestrate, SubAgentTool
    session.go               # SessionStore interface, Session struct, GenerateSessionID
    budget.go                # TokenBudget, budgetTracker
    result.go                # ResultLimiter, TruncateLimiter, HeadTailLimiter, NoLimiter
    compactor.go             # Compactor interface (17 lines)
    streaming_executor.go    # Streaming tool executor (overlaps model gen with tool exec)
    errors.go                # Sentinel errors, ProviderError (IsRetryable, IsContextTooLarge), ToolError, ErrorStrategy
    doc.go                   # Package-level documentation

    -- Extension packages --
    compactor/               # SlidingWindow, TokenWindow, Summary, Staged, ContextCollapser
    team/                    # Team, Member, Mailbox, SharedMemory, communication tools
    observability/           # Tracer (spans, trace context), CostTracker (token pricing)
    trigger/                 # Trigger, Scheduler (scheduled agent execution)
    plan/                    # Plan(), PlanAndExecute(), ExtractMemories()
    skill/                   # Skill, Registry, Execute(), Parse()
    task/                    # Task, Store, Status (work tracking)

    -- Providers --
    provider/openai/         # OpenAI Chat Completions (uses internal/sse)
    provider/anthropic/      # Anthropic Messages API (custom SSE parser)
    provider/gemini/         # Google Gemini generateContent (custom SSE parser)
    provider/groq/           # Groq API (uses internal/sse, OpenAI-compatible)
    provider/openrouter/     # OpenRouter API (uses internal/sse, OpenAI-compatible)
    provider/ollama/         # Ollama/RunPod API (JSONL streaming, HealthChecker)
    provider/fallback/       # Cascading multi-provider failover
    provider/mock/           # Deterministic mock for testing

    -- Internal --
    internal/sse/            # Shared OpenAI-compatible SSE parser: early tool emission, fuzzy ID recovery, stateless tag cleaning
    internal/jsonschema/     # Lightweight JSON Schema validator (type, required, enum, properties)

    -- Session stores --
    session/filestore/       # JSON file-based persistence
    session/memstore/        # In-memory persistence (testing)

    -- Middleware --
    middleware/logging.go        # slog-based tool execution logging
    middleware/metrics.go        # Thread-safe tool call metrics
    middleware/recovery.go       # Panic logging, turn limit guard
    middleware/circuitbreaker.go # Circuit breaker (open/half-open/closed)

    -- Tools --
    tools/builder.go         # Fluent ToolBuilder API
    tools/typed.go           # Generic TypedTool[I] with auto schema generation
    tools/builtin/           # 14 ready-to-use tools (bash, files, search, http, calculator, datetime, url reader, etc.)

    -- Examples --
    _examples/basic/         # Minimal agent
    _examples/chat/          # Interactive chat loop
    _examples/custom_tools/  # Multi-tool agent with metrics
    _examples/streaming/     # HTTP SSE endpoint
```

## Adding New Features

### Adding a New Provider

1. Create `provider/newprovider/` directory
2. Implement `agentflow.Provider` interface
3. If OpenAI-compatible, use `internal/sse.BuildRequestBody()` and `sse.NewStream()`
4. Propagate `req.Metadata` as HTTP headers for trace context
5. Optionally implement `agentflow.HealthChecker` for proactive health checking
6. Add unit test with `httptest.NewServer` (see `provider/openai/openai_test.go` for pattern)
7. Add integration test with `t.Skip` guard for API key
8. Document in README provider table

### Adding a New Built-in Tool

1. Create file in `tools/builtin/`
2. Implement `agentflow.Tool` interface (or use `tools.NewTyped[I]` for type-safe tools)
3. Implement `agentflow.LocalityAware` with correct locality
4. Add to `registry.go` in the appropriate preset functions (All, Remote, ReadOnly)
5. Add unit test in `builtin_test.go`

### Adding a New Middleware

1. Create file in `middleware/`
2. Return `[]agentflow.Hook` or `agentflow.Hook`
3. Use `agentflow.HookFunc` adapter for simple cases, or implement `agentflow.MultiPhaseHook` for hooks that fire at multiple phases
4. Store timing data in `HookContext.Metadata` (keyed by tool call ID)
5. For trace propagation, write `request:` prefixed keys to `HookContext.Metadata`

### Adding a New Session Store

1. Create `session/newstore/` directory
2. Implement `agentflow.SessionStore` interface
3. Return `agentflow.ErrSessionNotFound` for missing sessions
4. Ensure thread safety for concurrent access

### Adding a New Compactor

1. Create file in `compactor/` directory
2. Implement `agentflow.Compactor` interface (`ShouldCompact`, `Compact`)
3. Use `compactor.NewCompactionNotice()` for system messages about discarded context
4. Consider composability with `compactor.NewStaged()` for multi-stage strategies

## Testing Guidelines

- Every new feature must have unit tests using the mock provider
- Provider tests use `httptest.NewServer` for mock HTTP responses (see `provider/openai/openai_test.go`)
- Integration tests use Groq API (`GROQ_API_KEY`) or OpenRouter API (`OPENROUTER_API_KEY`) and must skip when the key is not set
- Integration tests should use `context.WithTimeout` to prevent hangs
- When running all integration tests together, rate limits may cause failures -- re-run failing tests individually
- Test names: `TestFeature_Scenario` for units, `TestIntegration_Feature` for integration
- Race tests: `TestRace_*` in `race_test.go` -- run with `-race` flag
- Benchmarks: `Benchmark*` in `benchmark_test.go` -- run with `-bench=.`

### Test File Locations

| Test File | Package | What It Tests |
|-----------|---------|---------------|
| `agent_test.go` | root | Core agent loop, RunSync, hooks |
| `budget_test.go` | root | Token budget enforcement |
| `compactor_test.go` | root | Compactor integration with agent |
| `execution_mode_test.go` | root | Local/remote mode tool filtering |
| `integration_test.go` | root | Multi-turn chat, tool use, session, sub-agent |
| `multimodal_test.go` | root | Image/vision support |
| `result_test.go` | root | Result limiting strategies |
| `session_test.go` | root | Session persistence |
| `subagent_test.go` | root | Sub-agent spawning |
| `streaming_executor_test.go` | root | Streaming tool execution |
| `ratelimit_test.go` | root | Token bucket limiter |
| `race_test.go` | root | Concurrent access safety |
| `benchmark_test.go` | root | Performance benchmarks |
| `task/task_test.go` | task | Task create, update, list |
| `skill/skill_test.go` | skill | Skill summarize, translate, list+run |
| `team/team_test.go` | team | Team run all, shared memory |
| `plan/plan_test.go` | plan | Plan, PlanAndExecute, memory extraction |
| `trigger/trigger_test.go` | trigger | Trigger execution |
| `observability/observability_test.go` | observability | Tracer, cost tracker |
| `middleware/circuitbreaker_test.go` | middleware | Circuit breaker, timeout, retry |
| `provider/openai/openai_test.go` | openai | Mock HTTP: text, tool call, error, timeout, metadata |
| `provider/anthropic/anthropic_test.go` | anthropic | Mock HTTP: text, tool call, error |
| `provider/gemini/gemini_test.go` | gemini | Mock HTTP: text, tool call, document, error |
| `provider/ollama/ollama_test.go` | ollama | Mock JSONL: text, tool call, multi-tool, usage, document fallback, health check |
| `provider/fallback/fallback_test.go` | fallback | Cascading failover logic |
| `openrouter_integration_test.go` | root | OpenRouter: simple chat, tool use, multi-tool |
| `internal/jsonschema/validate_test.go` | jsonschema | Schema validation: types, required, enum, nested |

## README Maintenance

When a significant structural change is made to the project, you MUST update README.md to reflect it. Significant changes include:
- New packages or directories added
- New built-in tools added
- New interfaces or config options added
- Provider changes (new providers, API changes)
- New middleware or extension packages
- Architecture changes (new event types, new pipeline steps)
- New examples or deployment patterns

Keep README.md in sync with the actual codebase. Do not let documentation drift from implementation.

## Post-Change Testing Workflow

After every code change, you MUST follow this workflow:

1. **Run `go build ./...` and `go vet ./...`** to verify the code compiles and passes static analysis.

2. **Run existing unit tests** with `go test ./... -run "Test[^I]" -timeout 60s` to catch regressions.

3. **Run or write integration tests for the changed code:**
   - If an integration test already exists for the changed functionality, run it.
   - If no integration test exists, write one following the `TestIntegration_` naming convention with `t.Skip` guard for `GROQ_API_KEY`.
   - Integration tests must use `context.WithTimeout` and the mock provider for unit-level verification where possible.

4. **Ask the user for a Groq API key** to run live integration tests. Do not skip this step. Prompt the user explicitly:
   > "Integration testlerini canli olarak calistirmak icin bir Groq API key'e ihtiyacim var. Paylasir misiniz?"

5. **Run live integration tests** with the provided key:
   ```bash
   GROQ_API_KEY=<key> go test ./... -run "TestIntegration_" -timeout 180s -v
   ```
   - If rate limiting causes failures, re-run failing tests individually.
   - Report which tests passed and which failed with root cause analysis.

6. **Never commit code that fails `go build` or `go vet`.**

## Known Issues

- **Streaming executor race conditions**: The streaming tool executor shares `loopState` with the main loop. While `state.done` channel and `bgWork` WaitGroup prevent send-on-closed-channel panics, the Go race detector may still flag concurrent access to `state.messages` and `state.metadata` between the main loop goroutine and streaming executor goroutines. The message snapshots in tool hooks mitigate this for hook contexts, but a full fix requires adding a mutex to `loopState` or restructuring the streaming executor to not share mutable state.
