package agentflow

import (
	"log/slog"
	"time"
)

// Config controls Agent behavior. Fields are set via functional Option values
// passed to NewAgent. Zero values indicate defaults.
type Config struct {
	// MaxTurns limits the number of agentic loop iterations. Each iteration
	// consists of a model call and optional tool execution. Zero means unlimited.
	MaxTurns int

	// MaxConcurrency limits the number of tools executing in parallel within
	// a single batch. Defaults to DefaultMaxConcurrency.
	MaxConcurrency int

	// SystemPrompt is prepended to the conversation as a system message.
	SystemPrompt string

	// Temperature controls the randomness of the model's output.
	// nil uses the provider's default.
	Temperature *float64

	// MaxTokens limits the maximum tokens in the model's response.
	// Zero uses the provider's default.
	MaxTokens int

	// RetryPolicy configures automatic retries for transient provider errors.
	// nil means no retries.
	RetryPolicy *RetryPolicy

	// EventBufferSize is the capacity of the event channel returned by Agent.Run.
	// Defaults to DefaultEventBufferSize.
	EventBufferSize int

	// OnEvent is an optional synchronous callback invoked for every event
	// before it is sent to the channel. Use for lightweight processing that
	// must see every event. Keep execution fast to avoid blocking the loop.
	OnEvent func(Event)

	// TokenBudget limits total token consumption across all turns.
	// nil means no budget limit.
	TokenBudget *TokenBudget

	// ResultLimiter controls how oversized tool results are handled.
	// Default: TruncateLimiter.
	ResultLimiter ResultLimiter

	// MaxResultChars is the maximum character count for a single tool result.
	// Results exceeding this are passed through the ResultLimiter.
	// Default: DefaultMaxResultChars (50000).
	MaxResultChars int

	// ExecutionMode controls which tools are visible to the model.
	// ModeLocal (default) allows all tools. ModeRemote restricts to remote-safe only.
	ExecutionMode ExecutionMode

	// DisableInputValidation skips JSON Schema validation of tool inputs.
	// Default: false (validation enabled). Set to true if tool schemas are
	// informational only and should not block execution on mismatch.
	DisableInputValidation bool

	// RateLimiter controls the rate of provider API calls. When set, the agent
	// waits for permission before each CreateStream call. nil means no rate limiting.
	RateLimiter RateLimiter

	// ToolRetries is the number of times to retry a failed tool execution before
	// returning the error to the model. Zero means no retries (default). Only
	// retries when the tool returns IsError: true.
	ToolRetries int

	// ToolTimeout sets a maximum duration for individual tool executions.
	// If a tool does not complete within this duration, its context is cancelled
	// and an error result is returned. Zero means no timeout (default).
	ToolTimeout time.Duration

	// ErrorStrategy controls how tool execution errors are handled. When set,
	// it is called after a tool returns IsError: true, allowing error transformation
	// or loop abortion. nil uses default behavior (send error to model as-is).
	ErrorStrategy ErrorStrategy

	// Logger enables structured logging for agent lifecycle events: model calls,
	// retries, compaction, budget warnings, and validation errors. nil disables
	// logging (zero overhead). Use log/slog for structured output.
	Logger *slog.Logger
}

// RetryPolicy configures automatic retries for transient errors.
type RetryPolicy struct {
	// MaxRetries is the maximum number of retry attempts. Zero means no retries.
	MaxRetries int

	// BaseDelay is the initial delay before the first retry.
	// Subsequent retries use exponential backoff: BaseDelay * 2^attempt.
	BaseDelay time.Duration

	// MaxDelay caps the maximum delay between retries.
	MaxDelay time.Duration
}

// Default configuration values.
const (
	DefaultMaxConcurrency  = 10
	DefaultEventBufferSize = 256
)

// Option is a functional option for configuring an Agent.
type Option func(*Agent)

// WithTool registers a single tool with the agent.
func WithTool(t Tool) Option {
	return func(a *Agent) {
		a.tools[t.Name()] = t
	}
}

// WithTools registers multiple tools with the agent.
func WithTools(tools ...Tool) Option {
	return func(a *Agent) {
		for _, t := range tools {
			a.tools[t.Name()] = t
		}
	}
}

// WithHook registers a lifecycle hook with the agent.
func WithHook(h Hook) Option {
	return func(a *Agent) {
		a.hooks = append(a.hooks, h)
	}
}

// WithPermission sets the permission checker for tool invocations.
// Only one permission checker is active; the last one set wins.
// Use ChainPermission to compose multiple checkers.
func WithPermission(p PermissionChecker) Option {
	return func(a *Agent) {
		a.permission = p
	}
}

// WithMaxTurns sets the maximum number of agentic loop iterations.
func WithMaxTurns(n int) Option {
	return func(a *Agent) {
		a.config.MaxTurns = n
	}
}

// WithMaxConcurrency sets the maximum number of parallel tool executions.
func WithMaxConcurrency(n int) Option {
	return func(a *Agent) {
		a.config.MaxConcurrency = n
	}
}

// WithSystemPrompt sets the system prompt prepended to every model call.
func WithSystemPrompt(s string) Option {
	return func(a *Agent) {
		a.config.SystemPrompt = s
	}
}

// WithTemperature sets the model's temperature parameter.
func WithTemperature(t float64) Option {
	return func(a *Agent) {
		a.config.Temperature = &t
	}
}

// WithMaxTokens sets the maximum tokens in the model's response.
func WithMaxTokens(n int) Option {
	return func(a *Agent) {
		a.config.MaxTokens = n
	}
}

// WithRetryPolicy configures automatic retries for transient provider errors.
func WithRetryPolicy(p RetryPolicy) Option {
	return func(a *Agent) {
		a.config.RetryPolicy = &p
	}
}

// WithEventBufferSize sets the capacity of the event channel.
func WithEventBufferSize(n int) Option {
	return func(a *Agent) {
		a.config.EventBufferSize = n
	}
}

// WithOnEvent sets a synchronous event callback. The callback is invoked for
// every event before it is sent to the channel. Keep execution fast.
func WithOnEvent(fn func(Event)) Option {
	return func(a *Agent) {
		a.config.OnEvent = fn
	}
}

// WithCompactor sets the context compaction strategy.
func WithCompactor(c Compactor) Option {
	return func(a *Agent) {
		a.compactor = c
	}
}

// WithTokenBudget sets a token consumption limit for the agent run.
// The loop terminates with TurnEndBudgetExhausted when the budget is exceeded.
//
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithTokenBudget(agentflow.TokenBudget{
//	        MaxTokens:        100000,
//	        WarningThreshold: 0.8,
//	    }),
//	)
func WithTokenBudget(budget TokenBudget) Option {
	return func(a *Agent) {
		a.config.TokenBudget = &budget
	}
}

// WithExecutionMode sets the execution environment mode. ModeRemote restricts
// the agent to only remote-safe tools — local-only tools are completely hidden
// from the model. ModeLocal (default) allows all registered tools.
//
//	// Server deployment — only web search and HTTP tools are available:
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithTools(builtin.All()...),
//	    agentflow.WithExecutionMode(agentflow.ModeRemote),
//	)
func WithExecutionMode(mode ExecutionMode) Option {
	return func(a *Agent) {
		a.config.ExecutionMode = mode
	}
}

// WithResultLimiter sets a custom result limiter for oversized tool outputs.
func WithResultLimiter(limiter ResultLimiter) Option {
	return func(a *Agent) {
		a.config.ResultLimiter = limiter
	}
}

// WithMaxResultSize sets the maximum characters for a single tool result.
// Results exceeding this are passed through the configured ResultLimiter.
func WithMaxResultSize(chars int) Option {
	return func(a *Agent) {
		a.config.MaxResultChars = chars
	}
}

// WithSessionStore enables session persistence. When set, RunSession and Resume
// methods become available for saving and restoring conversation state.
func WithSessionStore(store SessionStore) Option {
	return func(a *Agent) {
		a.sessionStore = store
	}
}

// WithErrorStrategy sets a custom error handling strategy for tool failures.
// The strategy can transform error results or abort the loop entirely.
//
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithErrorStrategy(agentflow.ErrorStrategyFunc(
//	        func(call *agentflow.ToolCall, result *agentflow.ToolResult) (*agentflow.ToolResult, agentflow.ErrorAction) {
//	            if call.Name == "critical_tool" {
//	                return result, agentflow.ErrorActionAbort
//	            }
//	            return result, agentflow.ErrorActionDefault
//	        },
//	    )),
//	)
func WithErrorStrategy(s ErrorStrategy) Option {
	return func(a *Agent) {
		a.config.ErrorStrategy = s
	}
}

// WithToolRetries sets the number of retry attempts for failed tool executions.
// Only retries when a tool returns IsError: true. Zero (default) means no retries.
//
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithToolRetries(2), // retry up to 2 times on error
//	)
func WithToolRetries(n int) Option {
	return func(a *Agent) {
		a.config.ToolRetries = n
	}
}

// WithToolTimeout sets a maximum duration for individual tool executions.
// Tools that exceed this duration have their context cancelled and receive
// an error result. Zero (default) means no timeout.
//
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithToolTimeout(30 * time.Second),
//	)
func WithToolTimeout(d time.Duration) Option {
	return func(a *Agent) {
		a.config.ToolTimeout = d
	}
}

// WithRateLimiter sets a rate limiter for provider API calls. The agent will
// call limiter.Wait() before each model request, blocking if the rate limit
// is exceeded.
//
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithRateLimiter(agentflow.NewTokenBucketLimiter(10, time.Second)),
//	)
func WithRateLimiter(limiter RateLimiter) Option {
	return func(a *Agent) {
		a.config.RateLimiter = limiter
	}
}

// WithLogger enables structured logging for agent lifecycle events.
// The logger receives log entries for model calls, retries, compaction,
// budget warnings, and input validation errors. nil disables logging.
//
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithLogger(slog.Default()),
//	)
func WithLogger(logger *slog.Logger) Option {
	return func(a *Agent) {
		a.config.Logger = logger
	}
}

// WithDisableInputValidation disables JSON Schema validation of tool inputs.
// Use this if tool schemas are informational only or if validation causes
// false positives with a particular model's output format.
func WithDisableInputValidation() Option {
	return func(a *Agent) {
		a.config.DisableInputValidation = true
	}
}
