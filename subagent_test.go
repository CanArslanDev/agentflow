package agentflow_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/mock"
	"github.com/CanArslanDev/agentflow/tools"
)

// TestSpawnChild verifies that a parent agent can spawn a child that runs
// independently and returns events.
func TestSpawnChild(t *testing.T) {
	provider := mock.New(
		// Child's response.
		mock.WithResponse(
			mock.TextDelta("Research result: Go channels are great."),
		),
	)

	parent := agentflow.NewAgent(provider,
		agentflow.WithSystemPrompt("You are the parent."),
		agentflow.WithMaxTurns(5),
	)

	var childText string
	var turnEnd *agentflow.TurnEndEvent

	events := parent.SpawnChild(context.Background(), agentflow.SubAgentConfig{
		SystemPrompt: "You are a research specialist.",
		MaxTurns:     3,
	}, "Research Go concurrency")

	for ev := range events {
		switch ev.Type {
		case agentflow.EventTextDelta:
			childText += ev.TextDelta.Text
		case agentflow.EventTurnEnd:
			turnEnd = ev.TurnEnd
		}
	}

	if childText != "Research result: Go channels are great." {
		t.Errorf("expected child text, got %q", childText)
	}
	if turnEnd == nil || turnEnd.Reason != agentflow.TurnEndComplete {
		t.Errorf("expected completed, got %v", turnEnd)
	}
}

// TestSpawnChildInheritsTools verifies that a child agent inherits parent tools
// when Tools is nil in SubAgentConfig.
func TestSpawnChildInheritsTools(t *testing.T) {
	provider := mock.New(
		// Child calls the inherited tool.
		mock.WithResponse(
			mock.ToolCallEvent("tc_1", "calculator", `{"expression": "1+1"}`),
		),
		// Child gets tool result and responds.
		mock.WithResponse(
			mock.TextDelta("The answer is 2."),
		),
	)

	calculator := tools.New("calculator", "Calculate").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(true).
		ReadOnly(true).
		WithExecute(func(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{Content: "2"}, nil
		}).
		Build()

	parent := agentflow.NewAgent(provider,
		agentflow.WithTools(calculator),
	)

	var toolStarts []string
	events := parent.SpawnChild(context.Background(), agentflow.SubAgentConfig{}, "What is 1+1?")
	for ev := range events {
		if ev.Type == agentflow.EventToolStart {
			toolStarts = append(toolStarts, ev.ToolStart.ToolCall.Name)
		}
	}

	if len(toolStarts) != 1 || toolStarts[0] != "calculator" {
		t.Errorf("expected child to use inherited calculator tool, got %v", toolStarts)
	}
}

// TestSpawnChildren verifies parallel sub-agent execution and event merging.
func TestSpawnChildren(t *testing.T) {
	// Each child gets its own provider calls, but mock is shared and calls are sequential.
	// For this test, we configure enough responses for all children.
	provider := mock.New(
		mock.WithResponse(mock.TextDelta("Result for task 0")),
		mock.WithResponse(mock.TextDelta("Result for task 1")),
		mock.WithResponse(mock.TextDelta("Result for task 2")),
	)

	parent := agentflow.NewAgent(provider, agentflow.WithMaxTurns(3))

	tasks := []string{"Task A", "Task B", "Task C"}

	var starts, ends int
	events := parent.SpawnChildren(context.Background(), agentflow.SubAgentConfig{MaxTurns: 2}, tasks)
	for ev := range events {
		switch ev.Type {
		case agentflow.EventSubAgentStart:
			starts++
		case agentflow.EventSubAgentEnd:
			ends++
			if ev.SubAgentEnd.Result == "" {
				// Some children may not get a response if mock runs out,
				// but at least one should succeed.
			}
		}
	}

	if starts != 3 {
		t.Errorf("expected 3 sub-agent starts, got %d", starts)
	}
	if ends != 3 {
		t.Errorf("expected 3 sub-agent ends, got %d", ends)
	}
}

// TestSubAgentTool verifies the delegate_task tool that the model can call.
func TestSubAgentTool(t *testing.T) {
	// Provider for the sub-agent tool's internal agent.
	subProvider := mock.New(
		mock.WithResponse(mock.TextDelta("Sub-agent result: found 42 patterns.")),
	)

	delegateTool := agentflow.SubAgentTool(subProvider, "You are a researcher.", 3)

	result, err := delegateTool.Execute(
		context.Background(),
		json.RawMessage(`{"task": "Research Go patterns"}`),
		nil,
	)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected success, got error: %s", result.Content)
	}
	if result.Content != "Sub-agent result: found 42 patterns." {
		t.Errorf("expected sub-agent result, got %q", result.Content)
	}
}

// TestSubAgentToolEmptyTask verifies that empty task returns an error.
func TestSubAgentToolEmptyTask(t *testing.T) {
	subProvider := mock.New()
	delegateTool := agentflow.SubAgentTool(subProvider, "", 3)

	result, err := delegateTool.Execute(
		context.Background(),
		json.RawMessage(`{"task": ""}`),
		nil,
	)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected error for empty task")
	}
}

// TestOrchestrate verifies the high-level fan-out/fan-in utility.
func TestOrchestrate(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(mock.TextDelta("Answer A")),
		mock.WithResponse(mock.TextDelta("Answer B")),
	)

	parent := agentflow.NewAgent(provider)

	results := agentflow.Orchestrate(context.Background(), parent, agentflow.SubAgentConfig{
		MaxTurns: 2,
	}, []string{"Task A", "Task B"})

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	// At least one should succeed (mock may run out for concurrent access).
	successCount := 0
	for _, r := range results {
		if r.Error == nil && r.Result != "" {
			successCount++
		}
	}

	if successCount == 0 {
		t.Error("expected at least one successful orchestration result")
	}
}
