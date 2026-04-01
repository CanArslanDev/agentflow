// Package mock provides a deterministic mock provider for testing agentflow agents.
//
// The mock provider allows you to program sequences of model responses, including
// text content and tool calls, enabling fully offline and reproducible agent tests.
//
//	p := mock.New(
//	    mock.WithResponse(
//	        mock.TextDelta("Let me search for that."),
//	        mock.ToolCallEvent("tc_1", "web_search", `{"query": "Go patterns"}`),
//	    ),
//	    mock.WithResponse(
//	        mock.TextDelta("Based on the results, here's what I found."),
//	    ),
//	)
package mock

import (
	"context"
	"io"
	"sync"

	"github.com/canarslan/agentflow"
)

// Provider is a deterministic mock that replays pre-programmed responses.
// Each call to CreateStream returns the next response in the sequence.
// Thread-safe for concurrent use.
type Provider struct {
	mu        sync.Mutex
	responses [][]agentflow.StreamEvent
	callIndex int
	model     string
}

// ProviderOption configures the mock provider.
type ProviderOption func(*Provider)

// New creates a mock provider with the given options.
func New(opts ...ProviderOption) *Provider {
	p := &Provider{model: "mock/test-model"}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

// WithModel sets the model ID returned by ModelID().
func WithModel(model string) ProviderOption {
	return func(p *Provider) {
		p.model = model
	}
}

// WithResponse adds a single response to the replay sequence. Each response
// consists of a series of stream events that will be yielded by the stream.
func WithResponse(events ...agentflow.StreamEvent) ProviderOption {
	return func(p *Provider) {
		p.responses = append(p.responses, events)
	}
}

// CreateStream returns a stream that replays the next response in the sequence.
// If the sequence is exhausted, it returns an error.
func (p *Provider) CreateStream(_ context.Context, _ *agentflow.Request) (agentflow.Stream, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.callIndex >= len(p.responses) {
		return nil, &agentflow.ProviderError{
			StatusCode: 500,
			Message:    "mock: no more responses configured",
			Retryable:  false,
		}
	}

	events := p.responses[p.callIndex]
	p.callIndex++

	return &mockStream{events: events}, nil
}

// ModelID returns the configured model identifier.
func (p *Provider) ModelID() string {
	return p.model
}

// CallCount returns the number of CreateStream calls made so far.
func (p *Provider) CallCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.callIndex
}

// Reset resets the call index to replay responses from the beginning.
func (p *Provider) Reset() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.callIndex = 0
}

// mockStream replays a pre-programmed sequence of events.
type mockStream struct {
	events []agentflow.StreamEvent
	index  int
	usage  *agentflow.Usage
}

func (s *mockStream) Next() (agentflow.StreamEvent, error) {
	if s.index >= len(s.events) {
		return agentflow.StreamEvent{}, io.EOF
	}

	ev := s.events[s.index]
	s.index++

	if ev.Type == agentflow.StreamEventUsage && ev.Usage != nil {
		s.usage = ev.Usage
	}

	return ev, nil
}

func (s *mockStream) Close() error {
	return nil
}

func (s *mockStream) Usage() *agentflow.Usage {
	return s.usage
}

// --- Helper functions for building stream events ---

// TextDelta creates a StreamEvent with a text content delta.
func TextDelta(text string) agentflow.StreamEvent {
	return agentflow.StreamEvent{
		Type:  agentflow.StreamEventDelta,
		Delta: &agentflow.ContentDelta{Text: text},
	}
}

// ToolCallEvent creates a StreamEvent with a tool call.
func ToolCallEvent(id, name, inputJSON string) agentflow.StreamEvent {
	return agentflow.StreamEvent{
		Type: agentflow.StreamEventToolCall,
		ToolCall: &agentflow.ToolCall{
			ID:    id,
			Name:  name,
			Input: []byte(inputJSON),
		},
	}
}

// UsageEvent creates a StreamEvent with token usage information.
func UsageEvent(prompt, completion int) agentflow.StreamEvent {
	return agentflow.StreamEvent{
		Type: agentflow.StreamEventUsage,
		Usage: &agentflow.Usage{
			PromptTokens:     prompt,
			CompletionTokens: completion,
			TotalTokens:      prompt + completion,
		},
	}
}

// ThinkingDelta creates a StreamEvent with a thinking content delta.
func ThinkingDelta(text string) agentflow.StreamEvent {
	return agentflow.StreamEvent{
		Type:          agentflow.StreamEventThinkingDelta,
		ThinkingDelta: &agentflow.ContentDelta{Text: text},
	}
}

// Done creates a StreamEvent signaling stream completion.
func Done() agentflow.StreamEvent {
	return agentflow.StreamEvent{Type: agentflow.StreamEventDone}
}
