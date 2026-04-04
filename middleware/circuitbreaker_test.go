package middleware_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/middleware"
	"github.com/CanArslanDev/agentflow/provider/mock"
	"github.com/CanArslanDev/agentflow/tools"
)

func TestCircuitBreaker_OpensAfterThreshold(t *testing.T) {
	cb := middleware.NewCircuitBreaker(2, time.Second)

	// Simulate 2 failures via PostToolUse hook.
	postHook := cb.Hooks()[1]
	for i := 0; i < 2; i++ {
		postHook.Execute(context.Background(), &agentflow.HookContext{
			Phase:    agentflow.HookPostToolUse,
			ToolCall: &agentflow.ToolCall{Name: "flaky_tool", ID: "tc1"},
			ToolResult: &agentflow.ToolResult{
				Content: "error",
				IsError: true,
			},
			Metadata: make(map[string]any),
		})
	}

	if cb.State("flaky_tool") != middleware.CircuitOpen {
		t.Errorf("expected CircuitOpen, got %d", cb.State("flaky_tool"))
	}

	// PreToolUse should block.
	preHook := cb.Hooks()[0]
	action, err := preHook.Execute(context.Background(), &agentflow.HookContext{
		Phase:    agentflow.HookPreToolUse,
		ToolCall: &agentflow.ToolCall{Name: "flaky_tool", ID: "tc2"},
		Metadata: make(map[string]any),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if action == nil || !action.Block {
		t.Error("expected circuit breaker to block the tool call")
	}
	t.Logf("Block reason: %s", action.BlockReason)
}

func TestCircuitBreaker_ClosesOnSuccess(t *testing.T) {
	cb := middleware.NewCircuitBreaker(1, time.Millisecond)
	postHook := cb.Hooks()[1]

	// Trip the circuit.
	postHook.Execute(context.Background(), &agentflow.HookContext{
		Phase:      agentflow.HookPostToolUse,
		ToolCall:   &agentflow.ToolCall{Name: "tool", ID: "tc1"},
		ToolResult: &agentflow.ToolResult{Content: "fail", IsError: true},
		Metadata:   make(map[string]any),
	})

	if cb.State("tool") != middleware.CircuitOpen {
		t.Fatal("expected open")
	}

	// Wait for reset.
	time.Sleep(5 * time.Millisecond)

	// PreToolUse should allow (half-open).
	preHook := cb.Hooks()[0]
	action, _ := preHook.Execute(context.Background(), &agentflow.HookContext{
		Phase:    agentflow.HookPreToolUse,
		ToolCall: &agentflow.ToolCall{Name: "tool", ID: "tc2"},
		Metadata: make(map[string]any),
	})
	if action != nil && action.Block {
		t.Error("expected half-open to allow probe request")
	}

	// Success should close the circuit.
	postHook.Execute(context.Background(), &agentflow.HookContext{
		Phase:      agentflow.HookPostToolUse,
		ToolCall:   &agentflow.ToolCall{Name: "tool", ID: "tc2"},
		ToolResult: &agentflow.ToolResult{Content: "ok"},
		Metadata:   make(map[string]any),
	})

	if cb.State("tool") != middleware.CircuitClosed {
		t.Errorf("expected closed after success, got %d", cb.State("tool"))
	}
}

func TestCircuitBreaker_Integration(t *testing.T) {
	// Mock provider: first turn calls flaky_tool (errors), second turn no tools.
	provider := mock.New(
		mock.WithResponse(mock.ToolCallEvent("tc1", "flaky", `{}`)),
		mock.WithResponse(mock.ToolCallEvent("tc2", "flaky", `{}`)),
		mock.WithResponse(mock.TextDelta("gave up")),
	)

	failCount := 0
	flaky := tools.New("flaky", "A flaky tool").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(true).ReadOnly(true).
		WithExecute(func(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			failCount++
			return &agentflow.ToolResult{Content: "service unavailable", IsError: true}, nil
		}).Build()

	cb := middleware.NewCircuitBreaker(2, time.Second)

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(flaky),
		agentflow.WithMaxTurns(5),
		agentflow.WithHook(cb.Hooks()[0]),
		agentflow.WithHook(cb.Hooks()[1]),
	)

	var lastText string
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("test"),
	}) {
		if ev.Type == agentflow.EventTextDelta && ev.TextDelta != nil {
			lastText += ev.TextDelta.Text
		}
	}

	t.Logf("Fail count: %d, Final text: %s, Circuit state: %d", failCount, lastText, cb.State("flaky"))

	if cb.State("flaky") != middleware.CircuitOpen {
		t.Error("expected circuit to be open after failures")
	}
}

func TestToolTimeout(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(mock.ToolCallEvent("tc1", "slow", `{}`)),
		mock.WithResponse(mock.TextDelta("done")),
	)

	slow := tools.New("slow", "A slow tool").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(true).ReadOnly(true).
		WithExecute(func(ctx context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			select {
			case <-time.After(5 * time.Second):
				return &agentflow.ToolResult{Content: "done"}, nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(slow),
		agentflow.WithMaxTurns(3),
		agentflow.WithToolTimeout(100*time.Millisecond),
	)

	start := time.Now()
	var toolErr bool
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("test"),
	}) {
		if ev.Type == agentflow.EventToolEnd && ev.ToolEnd != nil && ev.ToolEnd.Result.IsError {
			toolErr = true
			t.Logf("Tool error: %s (duration: %v)", ev.ToolEnd.Result.Content, ev.ToolEnd.Duration)
		}
	}
	elapsed := time.Since(start)

	if !toolErr {
		t.Error("expected tool to timeout with error")
	}
	if elapsed > 2*time.Second {
		t.Errorf("expected fast timeout, but took %v", elapsed)
	}
}

func TestToolRetry(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(mock.ToolCallEvent("tc1", "flaky", `{}`)),
		mock.WithResponse(mock.TextDelta("done")),
	)

	attempts := 0
	flaky := tools.New("flaky", "Fails then succeeds").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(true).ReadOnly(true).
		WithExecute(func(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			attempts++
			if attempts < 3 {
				return &agentflow.ToolResult{Content: "temporary error", IsError: true}, nil
			}
			return &agentflow.ToolResult{Content: "success"}, nil
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(flaky),
		agentflow.WithMaxTurns(3),
		agentflow.WithToolRetries(3),
	)

	var lastToolResult string
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("test"),
	}) {
		if ev.Type == agentflow.EventToolEnd && ev.ToolEnd != nil {
			lastToolResult = ev.ToolEnd.Result.Content
		}
	}

	t.Logf("Attempts: %d, Last result: %s", attempts, lastToolResult)
	if attempts != 3 {
		t.Errorf("expected 3 attempts (2 retries + 1 success), got %d", attempts)
	}
	if lastToolResult != "success" {
		t.Errorf("expected success after retry, got %q", lastToolResult)
	}
}
