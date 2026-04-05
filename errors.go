package agentflow

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
)

// Sentinel errors returned by the framework.
var (
	// ErrToolNotFound is returned when a model requests a tool that is not registered.
	ErrToolNotFound = errors.New("agentflow: tool not found")

	// ErrPermissionDenied is returned when a tool invocation is blocked by the permission checker.
	ErrPermissionDenied = errors.New("agentflow: permission denied")

	// ErrMaxTurnsExceeded is returned when the agentic loop exceeds the configured turn limit.
	ErrMaxTurnsExceeded = errors.New("agentflow: max turns exceeded")

	// ErrStreamClosed is returned when attempting to read from a closed stream.
	ErrStreamClosed = errors.New("agentflow: stream closed")

	// ErrInputValidation is returned when tool input fails schema validation.
	ErrInputValidation = errors.New("agentflow: input validation failed")

	// ErrHookBlocked is returned when a hook blocks tool execution or loop continuation.
	ErrHookBlocked = errors.New("agentflow: blocked by hook")

	// ErrProviderUnavailable is returned when the provider cannot be reached.
	ErrProviderUnavailable = errors.New("agentflow: provider unavailable")

	// ErrSessionNotFound is returned when a session ID does not exist in the store.
	ErrSessionNotFound = errors.New("agentflow: session not found")

	// ErrToolLoop is returned when the agent detects a repeated tool calling pattern.
	ErrToolLoop = errors.New("agentflow: tool calling loop detected")
)

// ProviderError wraps an error from the AI provider with status and retry information.
type ProviderError struct {
	StatusCode      int
	Message         string
	Retryable       bool
	Err             error
	ResponseHeaders http.Header
}

func (e *ProviderError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("agentflow: provider error (status %d): %s: %v", e.StatusCode, e.Message, e.Err)
	}
	return fmt.Sprintf("agentflow: provider error (status %d): %s", e.StatusCode, e.Message)
}

func (e *ProviderError) Unwrap() error {
	return e.Err
}

// IsRetryable reports whether the error is transient and the operation can be retried.
// Checks multiple signals following the pattern used by production AI SDKs:
//   - io.ErrUnexpectedEOF (stream ended unexpectedly)
//   - x-should-retry response header
//   - HTTP status codes: 408, 409, 429, 5xx
func (e *ProviderError) IsRetryable() bool {
	if e.Retryable {
		return true
	}
	// Unexpected EOF is always retryable (stream interruption).
	if e.Err != nil && errors.Is(e.Err, io.ErrUnexpectedEOF) {
		return true
	}
	// Check x-should-retry header from provider.
	if e.ResponseHeaders != nil {
		if v := e.ResponseHeaders.Get("x-should-retry"); v != "" {
			if b, err := strconv.ParseBool(v); err == nil {
				return b
			}
		}
	}
	// Standard retryable status codes.
	switch e.StatusCode {
	case 408, 409, 429:
		return true
	}
	return e.StatusCode >= 500
}

// IsContextTooLarge reports whether the error indicates the request context
// exceeds the model's maximum token limit.
func (e *ProviderError) IsContextTooLarge() bool {
	msg := strings.ToLower(e.Message)
	return strings.Contains(msg, "context") && strings.Contains(msg, "too large") ||
		strings.Contains(msg, "context length") && strings.Contains(msg, "exceed") ||
		strings.Contains(msg, "maximum context") ||
		strings.Contains(msg, "token limit") && strings.Contains(msg, "exceed") ||
		strings.Contains(msg, "too many tokens") ||
		strings.Contains(msg, "input too long")
}

// IsContextTooLargeError reports whether err indicates a context size overflow.
func IsContextTooLargeError(err error) bool {
	var pe *ProviderError
	if errors.As(err, &pe) {
		return pe.IsContextTooLarge()
	}
	return false
}

// ToolError wraps an error from tool execution with the tool name and call ID.
type ToolError struct {
	ToolName   string
	ToolCallID string
	Err        error
}

func (e *ToolError) Error() string {
	return fmt.Sprintf("agentflow: tool %q (call %s): %v", e.ToolName, e.ToolCallID, e.Err)
}

func (e *ToolError) Unwrap() error {
	return e.Err
}

// IsRetryableError reports whether err is a retryable provider error.
func IsRetryableError(err error) bool {
	var pe *ProviderError
	if errors.As(err, &pe) {
		return pe.IsRetryable()
	}
	return false
}

// ErrorAction determines how the agent loop handles a tool error.
type ErrorAction int

const (
	// ErrorActionDefault sends the error to the model as-is (default behavior).
	ErrorActionDefault ErrorAction = iota

	// ErrorActionAbort terminates the agent loop immediately.
	ErrorActionAbort
)

// ErrorStrategy controls how the agent handles tool execution errors.
// Implementations can transform error results, suppress them, or abort the loop.
type ErrorStrategy interface {
	// OnToolError is called when a tool returns IsError: true. It receives the
	// tool call and the error result, and returns a (possibly modified) result
	// and an action.
	OnToolError(call *ToolCall, result *ToolResult) (*ToolResult, ErrorAction)
}

// ErrorStrategyFunc adapts a function to the ErrorStrategy interface.
type ErrorStrategyFunc func(call *ToolCall, result *ToolResult) (*ToolResult, ErrorAction)

// OnToolError delegates to the wrapped function.
func (f ErrorStrategyFunc) OnToolError(call *ToolCall, result *ToolResult) (*ToolResult, ErrorAction) {
	return f(call, result)
}
