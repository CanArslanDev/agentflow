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
go test ./... -run "Test[^I]"

# Run integration tests (requires GROQ_API_KEY)
GROQ_API_KEY=gsk-... go test ./... -run "TestIntegration_" -timeout 180s

# Run everything
GROQ_API_KEY=gsk-... go test ./... -timeout 180s

# Static analysis
go vet ./...
```

## Architecture

### Core Loop (agent.go)

The agentic loop is a `for {}` in `runLoop()`:

```
for {
    1. Check context cancellation
    2. Check turn limit
    3. Check token budget
    4. Run compaction if needed
    5. Execute pre-model hooks
    6. Call provider.CreateStream() -- stream response
    7. Consume stream: emit text deltas, collect tool calls
    8. Execute post-model hooks
    9. If no tool calls -> return (completed)
   10. Execute tools through pipeline (validate -> hooks -> permission -> execute -> hooks)
   11. Apply result size limiting
   12. Append tool results to messages
   13. Continue loop
}
```

The loop runs in a dedicated goroutine launched by `Agent.Run()`. Events are delivered through a buffered channel.

### Layered Design

```
Public API (agent.go, config.go)
    |
Core Types (tool.go, message.go, event.go, hook.go, permission.go, provider.go)
    |
Internal (internal/sse/ -- shared SSE parsing)
    |
Providers (provider/groq, provider/openrouter, provider/fallback, provider/mock)
    |
Extensions (session/, middleware/, tools/builtin/)
```

### Key Interfaces

| Interface | File | Purpose |
|-----------|------|---------|
| `Provider` | provider.go | AI model API abstraction |
| `Stream` | provider.go | Pull-based SSE event stream |
| `Tool` | tool.go | Agent capability (function calling) |
| `LocalityAware` | tool.go | Optional: declare local/remote safety |
| `Hook` | hook.go | Lifecycle interception |
| `PermissionChecker` | permission.go | Tool access control |
| `Compactor` | compactor.go | Conversation history management |
| `SessionStore` | session.go | Conversation persistence |
| `ResultLimiter` | result.go | Oversized result handling |

## Code Conventions

### General Rules

- Go 1.23+, standard library only in core packages
- All exported types and functions must have godoc comments
- No emojis in code or documentation
- English only for all code, comments, and documentation
- Error messages start with lowercase, no trailing punctuation

### Patterns Used

- **Functional options**: `NewAgent(provider, WithTool(...), WithMaxTurns(10))`
- **Optional interfaces**: `LocalityAware` extends `Tool` without breaking existing implementations
- **Discriminated unions**: `Event.Type` determines which field is set
- **Pull-based streams**: `Stream.Next()` returns `(StreamEvent, error)`, `io.EOF` signals completion
- **Channel-based event delivery**: `Agent.Run()` returns `<-chan Event`

### Tool Implementation Pattern

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

### Provider Implementation Pattern

All providers that use OpenAI-compatible APIs should use the shared `internal/sse` package:

```go
func (p *Provider) CreateStream(ctx context.Context, req *agentflow.Request) (agentflow.Stream, error) {
    body := sse.BuildRequestBody(p.model, req)  // shared conversion
    // ... HTTP request ...
    return sse.NewStream(resp), nil              // shared SSE parser
}
```

### Test Naming

- Unit tests: `TestFeatureName` -- no API calls, uses mock provider
- Integration tests: `TestIntegration_FeatureName` -- requires `GROQ_API_KEY`, skips otherwise
- Tests in provider packages: `TestGroqSimpleChat`, `TestFallback_PrimaryFails`, etc.

### Execution Mode Safety

- Tools default to `ToolLocalOnly` if they do not implement `LocalityAware` (safe default)
- In `ModeRemote`, local-only tools are filtered from `toolDefinitions()` -- the model never sees them
- A guard in `executeSingleTool()` blocks local tools even if called by name (defense in depth)
- New built-in tools must declare their `Locality()` explicitly

### Result Limiting

- The `limitResult()` method in agent.go applies after every tool execution
- Error results (`IsError: true`) are never truncated
- Default limiter is `TruncateLimiter` with 80/20 head/tail split
- Default max size is 50,000 characters

### Budget Tracking

- `budgetTracker` uses atomic operations for thread safety
- Warning fires exactly once via `CompareAndSwap`
- Budget check happens after each `stream.Usage()` in the loop
- When exhausted, the last assistant message is preserved before terminating

## Directory Structure

```
agentflow/
    -- Core package (root) --
    agent.go                 # Agent struct, Run(), RunSync(), RunSession(), Resume()
    config.go                # Config struct, all WithXxx() option functions
    tool.go                  # Tool interface, ExecutionMode, ToolLocality, LocalityAware
    message.go               # Message, ContentBlock (text, tool_call, tool_result, image)
    event.go                 # Event discriminated union, all event types
    hook.go                  # Hook interface, HookFunc adapter, HookContext, HookAction
    permission.go            # PermissionChecker, AllowAll, DenyList, AllowList, Chain
    provider.go              # Provider, Stream, StreamEvent, Usage
    subagent.go              # SubAgentConfig, SpawnChild, SpawnChildren, Orchestrate, SubAgentTool
    session.go               # SessionStore interface, Session struct, GenerateSessionID
    budget.go                # TokenBudget, budgetTracker
    result.go                # ResultLimiter, TruncateLimiter, HeadTailLimiter, NoLimiter
    compactor.go             # Compactor interface
    compactor_sliding.go     # SlidingWindowCompactor, TokenWindowCompactor
    compactor_summary.go     # SummaryCompactor (AI-powered)
    errors.go                # Sentinel errors, ProviderError, ToolError
    doc.go                   # Package-level documentation

    -- Providers --
    provider/groq/           # Groq API (uses internal/sse)
    provider/openrouter/     # OpenRouter API (uses internal/sse)
    provider/fallback/       # Cascading multi-provider failover
    provider/mock/           # Deterministic mock for testing

    -- Internal --
    internal/sse/            # Shared OpenAI-compatible SSE parser and request builder

    -- Session stores --
    session/filestore/       # JSON file-based persistence
    session/memstore/        # In-memory persistence (testing)

    -- Middleware --
    middleware/logging.go    # slog-based tool execution logging
    middleware/metrics.go    # Thread-safe tool call metrics
    middleware/recovery.go   # Panic logging, turn limit guard

    -- Tools --
    tools/builder.go         # Fluent ToolBuilder API
    tools/builtin/           # 11 ready-to-use tools (bash, files, search, http, etc.)

    -- Examples --
    _examples/basic/         # Minimal agent
    _examples/custom_tools/  # Multi-tool agent with metrics
    _examples/streaming/     # HTTP SSE endpoint
```

## Adding New Features

### Adding a New Provider

1. Create `provider/newprovider/` directory
2. Implement `agentflow.Provider` interface
3. If OpenAI-compatible, use `internal/sse.BuildRequestBody()` and `sse.NewStream()`
4. Add integration test with `t.Skip` guard for API key
5. Document in README provider table

### Adding a New Built-in Tool

1. Create file in `tools/builtin/`
2. Implement `agentflow.Tool` interface
3. Implement `agentflow.LocalityAware` with correct locality
4. Add to `registry.go` in the appropriate preset functions (All, Remote, ReadOnly)
5. Add unit test in `builtin_test.go`

### Adding a New Middleware

1. Create file in `middleware/`
2. Return `[]agentflow.Hook` or `agentflow.Hook`
3. Use `agentflow.HookFunc` adapter for simple cases
4. Store timing data in `HookContext.Metadata` (keyed by tool call ID)

### Adding a New Session Store

1. Create `session/newstore/` directory
2. Implement `agentflow.SessionStore` interface
3. Return `agentflow.ErrSessionNotFound` for missing sessions
4. Ensure thread safety for concurrent access

## Testing Guidelines

- Every new feature must have unit tests using the mock provider
- Integration tests use Groq API and must skip when `GROQ_API_KEY` is not set
- Integration tests should use `context.WithTimeout` to prevent hangs
- When running all integration tests together, rate limits may cause failures -- run them sequentially with pauses if needed
- Test names: `TestFeature_Scenario` for units, `TestIntegration_Feature` for integration
