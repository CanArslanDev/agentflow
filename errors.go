package agentflow

import (
	"errors"
	"fmt"
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
)

// ProviderError wraps an error from the AI provider with status and retry information.
type ProviderError struct {
	StatusCode int
	Message    string
	Retryable  bool
	Err        error
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
func (e *ProviderError) IsRetryable() bool {
	return e.Retryable
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
