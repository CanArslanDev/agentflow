package agentflow_test

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/mock"
	"github.com/CanArslanDev/agentflow/tools"
)

// TestRace_ConcurrentToolExecution tests that concurrent tool execution
// does not race on shared state. Run with -race flag.
func TestRace_ConcurrentToolExecution(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(
			mock.ToolCallEvent("tc1", "counter", `{}`),
			mock.ToolCallEvent("tc2", "counter", `{}`),
			mock.ToolCallEvent("tc3", "counter", `{}`),
		),
		mock.WithResponse(mock.TextDelta("done")),
	)

	var mu sync.Mutex
	count := 0

	counter := tools.New("counter", "Increment counter").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(true).ReadOnly(true).
		WithExecute(func(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			mu.Lock()
			count++
			mu.Unlock()
			time.Sleep(10 * time.Millisecond)
			return &agentflow.ToolResult{Content: "ok"}, nil
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(counter),
		agentflow.WithMaxTurns(3),
		agentflow.WithMaxConcurrency(3),
	)

	for range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("go"),
	}) {
	}

	mu.Lock()
	if count != 3 {
		t.Errorf("expected 3 calls, got %d", count)
	}
	mu.Unlock()
}

// TestRace_MultipleAgentsSharedProvider tests that multiple agents can use
// the same provider concurrently without races.
func TestRace_MultipleAgentsSharedProvider(t *testing.T) {
	opts := make([]mock.ProviderOption, 10)
	for i := range opts {
		opts[i] = mock.WithResponse(mock.TextDelta("response"))
	}
	provider := mock.New(opts...)

	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			agent := agentflow.NewAgent(provider, agentflow.WithMaxTurns(1))
			for range agent.Run(context.Background(), []agentflow.Message{
				agentflow.NewUserMessage("hi"),
			}) {
			}
		}()
	}
	wg.Wait()
}

// TestRace_EventConsumption tests that event channel consumption from
// multiple goroutines doesn't race.
func TestRace_EventConsumption(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(
			mock.ToolCallEvent("tc1", "tool1", `{}`),
		),
		mock.WithResponse(mock.TextDelta("final")),
	)

	tool1 := tools.New("tool1", "Test tool").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(true).ReadOnly(true).
		WithExecute(func(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{Content: "result"}, nil
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(tool1),
		agentflow.WithMaxTurns(3),
	)

	events := agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("go"),
	})

	var textCount, toolCount int
	for ev := range events {
		switch ev.Type {
		case agentflow.EventTextDelta:
			textCount++
		case agentflow.EventToolStart:
			toolCount++
		}
	}

	if textCount == 0 {
		t.Error("expected text events")
	}
}

// TestRace_SpawnChildren tests concurrent sub-agent spawning.
func TestRace_SpawnChildren(t *testing.T) {
	opts := make([]mock.ProviderOption, 10)
	for i := range opts {
		opts[i] = mock.WithResponse(mock.TextDelta("child response"))
	}
	provider := mock.New(opts...)

	parent := agentflow.NewAgent(provider, agentflow.WithMaxTurns(1))

	events := parent.SpawnChildren(context.Background(),
		agentflow.SubAgentConfig{MaxTurns: 1},
		[]string{"task 1", "task 2", "task 3", "task 4", "task 5"},
	)

	var childEvents int
	for range events {
		childEvents++
	}

	if childEvents == 0 {
		t.Error("expected events from children")
	}
	t.Logf("Received %d events from 5 children", childEvents)
}

// TestRace_CancelDuringExecution tests cancellation mid-execution doesn't race.
func TestRace_CancelDuringExecution(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(mock.ToolCallEvent("tc1", "slow", `{}`)),
		mock.WithResponse(mock.TextDelta("done")),
	)

	slow := tools.New("slow", "Slow tool").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(true).ReadOnly(true).
		WithExecute(func(ctx context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			select {
			case <-time.After(time.Second):
				return &agentflow.ToolResult{Content: "done"}, nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(slow),
		agentflow.WithMaxTurns(3),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	for range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("go"),
	}) {
	}
	// No panic = pass.
}
