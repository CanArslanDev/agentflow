package agentflow

import (
	"context"
	"io"
)

// Provider abstracts an AI model API that supports tool use (function calling).
// Implementations handle authentication, request formatting, and response parsing
// specific to each vendor (Anthropic, OpenAI, OpenRouter, etc.).
//
// The interface operates at the stream level rather than request/response level,
// enabling real-time event delivery to consumers.
type Provider interface {
	// CreateStream initiates a streaming model call and returns a Stream that
	// yields events as they arrive from the API. The caller must consume the
	// stream to completion or cancel via context.
	//
	// The Request contains provider-agnostic fields (messages, tools, parameters).
	// Implementations convert these into vendor-specific API formats.
	CreateStream(ctx context.Context, req *Request) (Stream, error)

	// ModelID returns the identifier of the model this provider targets
	// (e.g., "anthropic/claude-sonnet-4-20250514", "openai/gpt-4o").
	ModelID() string
}

// Request is the provider-agnostic model request built by the agentic loop.
type Request struct {
	// Messages is the conversation history including user messages, assistant
	// responses, and tool results from previous iterations.
	Messages []Message

	// SystemPrompt is an optional system-level instruction prepended to the conversation.
	SystemPrompt string

	// Tools is the list of tool definitions available to the model for this request.
	Tools []ToolDefinition

	// MaxTokens limits the maximum number of tokens in the model's response.
	MaxTokens int

	// Temperature controls the randomness of the model's output.
	// nil means the provider's default is used.
	Temperature *float64

	// StopSequences are optional strings that cause the model to stop generating
	// when encountered. Support varies by provider.
	StopSequences []string

	// Metadata carries key-value pairs that providers may propagate as HTTP
	// headers. Use this for trace context propagation (e.g., "traceparent",
	// "tracestate") or custom request tagging. Providers add these as headers
	// prefixed with nothing — keys map directly to header names.
	Metadata map[string]string

	// ProviderExtras carries provider-specific parameters that are merged into
	// the request body. Examples: OpenRouter "plugins" for file parsing,
	// Anthropic cache control, Gemini safety settings. Each provider picks
	// the keys it understands and ignores the rest.
	ProviderExtras map[string]any
}

// Stream yields events from a model response. The caller reads events sequentially
// via Next() until io.EOF signals completion. The stream must be closed when done,
// even after encountering an error.
//
// Implementations must be safe to consume from a single goroutine. The StreamEvent
// returned by Next is only valid until the next call to Next.
type Stream interface {
	// Next blocks until the next event is available. Returns io.EOF when the
	// stream is complete. Any other error indicates a failure in the stream.
	Next() (StreamEvent, error)

	// Close releases resources associated with the stream. Safe to call
	// multiple times. Must be called even if Next returned an error.
	Close() error

	// Usage returns token usage statistics after the stream completes.
	// Returns nil if the stream has not finished or the provider does not
	// report usage information.
	Usage() *Usage
}

// StreamEventType discriminates the variant within a StreamEvent.
type StreamEventType int

const (
	// StreamEventDelta carries a text content delta from the model.
	StreamEventDelta StreamEventType = iota

	// StreamEventToolCall carries a complete tool invocation from the model.
	StreamEventToolCall

	// StreamEventError carries a non-fatal error from the stream.
	StreamEventError

	// StreamEventDone signals that the stream has completed normally.
	StreamEventDone

	// StreamEventUsage carries token usage information, typically at stream end.
	StreamEventUsage

	// StreamEventThinkingDelta carries a thinking/reasoning content delta.
	StreamEventThinkingDelta
)

// StreamEvent is a discriminated union of events from a model stream. Exactly one
// of the typed fields is set, determined by the Type field.
type StreamEvent struct {
	Type StreamEventType

	// Delta is set when Type == StreamEventDelta.
	Delta *ContentDelta

	// ThinkingDelta is set when Type == StreamEventThinkingDelta.
	ThinkingDelta *ContentDelta

	// ToolCall is set when Type == StreamEventToolCall.
	ToolCall *ToolCall

	// Error is set when Type == StreamEventError.
	Error error

	// Usage is set when Type == StreamEventUsage.
	Usage *Usage
}

// ContentDelta represents an incremental text chunk from the model's response.
type ContentDelta struct {
	Text string
}

// Usage reports token consumption for a model request.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// HealthChecker is an optional interface that providers can implement to
// support proactive health checking. Use with fallback providers to detect
// unhealthy backends before they cause request failures.
type HealthChecker interface {
	// HealthCheck tests whether the provider is reachable and operational.
	// Returns nil if healthy, or an error describing the failure.
	HealthCheck(ctx context.Context) error
}

// IsHealthy checks if a provider is healthy. If the provider does not
// implement HealthChecker, it is assumed healthy.
func IsHealthy(ctx context.Context, p Provider) error {
	if hc, ok := p.(HealthChecker); ok {
		return hc.HealthCheck(ctx)
	}
	return nil
}

// Ensure io.EOF is the canonical stream-end signal.
var _ error = io.EOF
