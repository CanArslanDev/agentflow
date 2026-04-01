package fallback_test

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/fallback"
	"github.com/CanArslanDev/agentflow/provider/groq"
	"github.com/CanArslanDev/agentflow/provider/mock"
	"github.com/CanArslanDev/agentflow/tools"
)

// failingProvider always returns a retryable error.
type failingProvider struct {
	model     string
	retryable bool
}

func (p *failingProvider) CreateStream(_ context.Context, _ *agentflow.Request) (agentflow.Stream, error) {
	return nil, &agentflow.ProviderError{
		StatusCode: 503,
		Message:    "service unavailable",
		Retryable:  p.retryable,
	}
}

func (p *failingProvider) ModelID() string { return p.model }

// TestFallback_PrimarySucceeds — primary başarılı ise fallback'e düşmez.
func TestFallback_PrimarySucceeds(t *testing.T) {
	primary := mock.New(mock.WithResponse(mock.TextDelta("from primary")))
	backup := mock.New(mock.WithResponse(mock.TextDelta("from backup")))

	provider := fallback.New(primary, backup)

	agent := agentflow.NewAgent(provider, agentflow.WithMaxTurns(1))

	var text string
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("hi"),
	}) {
		if ev.Type == agentflow.EventTextDelta {
			text += ev.TextDelta.Text
		}
	}

	if text != "from primary" {
		t.Errorf("expected 'from primary', got %q", text)
	}
	if primary.CallCount() != 1 {
		t.Errorf("expected primary called once, got %d", primary.CallCount())
	}
	if backup.CallCount() != 0 {
		t.Errorf("expected backup not called, got %d", backup.CallCount())
	}
}

// TestFallback_PrimaryFails — primary başarısız, backup devralır.
func TestFallback_PrimaryFails(t *testing.T) {
	primary := &failingProvider{model: "primary", retryable: true}
	backup := mock.New(mock.WithResponse(mock.TextDelta("from backup")))

	provider := fallback.New(primary, backup)
	agent := agentflow.NewAgent(provider, agentflow.WithMaxTurns(1))

	var text string
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("hi"),
	}) {
		if ev.Type == agentflow.EventTextDelta {
			text += ev.TextDelta.Text
		}
	}

	if text != "from backup" {
		t.Errorf("expected 'from backup', got %q", text)
	}
}

// TestFallback_AllFail — tüm provider'lar başarısız.
func TestFallback_AllFail(t *testing.T) {
	p1 := &failingProvider{model: "p1", retryable: true}
	p2 := &failingProvider{model: "p2", retryable: true}
	p3 := &failingProvider{model: "p3", retryable: true}

	provider := fallback.New(p1, p2, p3)
	agent := agentflow.NewAgent(provider, agentflow.WithMaxTurns(1))

	var gotError bool
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("hi"),
	}) {
		if ev.Type == agentflow.EventError {
			gotError = true
			t.Logf("Error: %v", ev.Error.Err)
		}
	}

	if !gotError {
		t.Error("expected error when all providers fail")
	}
}

// TestFallback_NonRetryableStopsChain — auth hatası cascade'i durdurur.
func TestFallback_NonRetryableStopsChain(t *testing.T) {
	primary := &failingProvider{model: "primary", retryable: false} // 403 gibi
	backup := mock.New(mock.WithResponse(mock.TextDelta("should not reach")))

	provider := fallback.New(primary, backup)
	agent := agentflow.NewAgent(provider, agentflow.WithMaxTurns(1))

	var gotError bool
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("hi"),
	}) {
		if ev.Type == agentflow.EventError {
			gotError = true
		}
	}

	if !gotError {
		t.Error("expected error for non-retryable failure")
	}
	if backup.CallCount() != 0 {
		t.Error("backup should not be called for non-retryable error")
	}
}

// TestFallback_ToolUseAcrossProviders — fallback tool use loop ile çalışır.
func TestFallback_ToolUseAcrossProviders(t *testing.T) {
	primary := &failingProvider{model: "broken", retryable: true}
	backup := mock.New(
		mock.WithResponse(mock.ToolCallEvent("tc1", "calc", `{"x":1}`)),
		mock.WithResponse(mock.TextDelta("Result is 42")),
	)

	provider := fallback.New(primary, backup)

	calc := tools.New("calc", "Calculate").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(true).ReadOnly(true).
		WithExecute(func(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{Content: "42"}, nil
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(calc),
		agentflow.WithMaxTurns(5),
	)

	var text string
	var toolCalls int
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("calculate"),
	}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			text += ev.TextDelta.Text
		case agentflow.EventToolStart:
			toolCalls++
		}
	}

	if text != "Result is 42" {
		t.Errorf("expected 'Result is 42', got %q", text)
	}
	if toolCalls != 1 {
		t.Errorf("expected 1 tool call, got %d", toolCalls)
	}
}

// TestFallback_ModelID — primary'nin model ID'sini döner.
func TestFallback_ModelID(t *testing.T) {
	p := fallback.New(
		mock.New(mock.WithModel("primary-model")),
		mock.New(mock.WithModel("backup-model")),
	)
	if p.ModelID() != "primary-model" {
		t.Errorf("expected 'primary-model', got %q", p.ModelID())
	}
}

// --- Integration test ---

// TestIntegration_FallbackWithGroq — fake primary fails, Groq picks up.
func TestIntegration_FallbackWithGroq(t *testing.T) {
	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		t.Skip("GROQ_API_KEY not set")
	}

	broken := &failingProvider{model: "broken-provider", retryable: true}
	groqProvider := groq.New(key, "llama-3.3-70b-versatile")

	provider := fallback.New(broken, groqProvider)

	agent := agentflow.NewAgent(provider,
		agentflow.WithMaxTurns(1),
		agentflow.WithMaxTokens(50),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var text string
	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("Say 'fallback works'"),
	}) {
		if ev.Type == agentflow.EventTextDelta {
			text += ev.TextDelta.Text
		}
	}

	if text == "" {
		t.Error("expected response from Groq fallback")
	}
	t.Logf("Groq fallback response: %q", text)
}

// dummyStream implements agentflow.Stream for inline tests.
type dummyStream struct {
	events []agentflow.StreamEvent
	idx    int
}

func (s *dummyStream) Next() (agentflow.StreamEvent, error) {
	if s.idx >= len(s.events) {
		return agentflow.StreamEvent{}, io.EOF
	}
	ev := s.events[s.idx]
	s.idx++
	return ev, nil
}
func (s *dummyStream) Close() error            { return nil }
func (s *dummyStream) Usage() *agentflow.Usage { return nil }
