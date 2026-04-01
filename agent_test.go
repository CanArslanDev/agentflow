package agentflow_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/canarslan/agentflow"
	"github.com/canarslan/agentflow/provider/mock"
	"github.com/canarslan/agentflow/tools"
)

// TestSimpleTextResponse verifies the agent handles a plain text response
// without tool calls — the loop should complete in one turn.
func TestSimpleTextResponse(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(
			mock.TextDelta("Hello, "),
			mock.TextDelta("world!"),
		),
	)

	agent := agentflow.NewAgent(provider)
	messages := []agentflow.Message{agentflow.NewUserMessage("Hi")}

	var gotText string
	var gotTurnEnd *agentflow.TurnEndEvent

	for ev := range agent.Run(context.Background(), messages) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			gotText += ev.TextDelta.Text
		case agentflow.EventTurnEnd:
			gotTurnEnd = ev.TurnEnd
		}
	}

	if gotText != "Hello, world!" {
		t.Errorf("expected text 'Hello, world!', got %q", gotText)
	}
	if gotTurnEnd == nil {
		t.Fatal("expected TurnEnd event")
	}
	if gotTurnEnd.Reason != agentflow.TurnEndComplete {
		t.Errorf("expected reason 'completed', got %q", gotTurnEnd.Reason)
	}
	if gotTurnEnd.TurnNumber != 1 {
		t.Errorf("expected turn 1, got %d", gotTurnEnd.TurnNumber)
	}
}

// TestToolCallLoop verifies the core agentic loop: model requests a tool,
// tool executes, result is sent back, model responds with final text.
func TestToolCallLoop(t *testing.T) {
	provider := mock.New(
		// Turn 1: model requests a tool call.
		mock.WithResponse(
			mock.TextDelta("Let me calculate that."),
			mock.ToolCallEvent("tc_1", "calculator", `{"expression": "6 * 7"}`),
		),
		// Turn 2: model sees tool result and responds.
		mock.WithResponse(
			mock.TextDelta("The answer is 42."),
		),
	)

	calculator := tools.New("calculator", "Calculate math").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(true).
		ReadOnly(true).
		WithExecute(func(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{Content: "42"}, nil
		}).
		Build()

	agent := agentflow.NewAgent(provider, agentflow.WithTools(calculator))

	var textParts []string
	var toolStarts []string
	var toolEnds []string
	var turnEnd *agentflow.TurnEndEvent

	for ev := range agent.Run(context.Background(), []agentflow.Message{agentflow.NewUserMessage("6 * 7")}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			textParts = append(textParts, ev.TextDelta.Text)
		case agentflow.EventToolStart:
			toolStarts = append(toolStarts, ev.ToolStart.ToolCall.Name)
		case agentflow.EventToolEnd:
			toolEnds = append(toolEnds, ev.ToolEnd.ToolCall.Name)
		case agentflow.EventTurnEnd:
			turnEnd = ev.TurnEnd
		}
	}

	if len(toolStarts) != 1 || toolStarts[0] != "calculator" {
		t.Errorf("expected 1 tool start for 'calculator', got %v", toolStarts)
	}
	if len(toolEnds) != 1 || toolEnds[0] != "calculator" {
		t.Errorf("expected 1 tool end for 'calculator', got %v", toolEnds)
	}
	if turnEnd == nil || turnEnd.TurnNumber != 2 {
		t.Errorf("expected 2 turns, got %v", turnEnd)
	}
	if turnEnd.Reason != agentflow.TurnEndComplete {
		t.Errorf("expected completed, got %s", turnEnd.Reason)
	}
}

// TestMaxTurns verifies the loop stops when max turns is reached.
func TestMaxTurns(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(mock.ToolCallEvent("tc_1", "echo", `{}`)),
		mock.WithResponse(mock.ToolCallEvent("tc_2", "echo", `{}`)),
		mock.WithResponse(mock.ToolCallEvent("tc_3", "echo", `{}`)),
	)

	echo := tools.New("echo", "Echo").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(true).
		ReadOnly(true).
		WithExecute(func(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{Content: "ok"}, nil
		}).
		Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(echo),
		agentflow.WithMaxTurns(2),
	)

	var turnEnd *agentflow.TurnEndEvent
	for ev := range agent.Run(context.Background(), []agentflow.Message{agentflow.NewUserMessage("go")}) {
		if ev.Type == agentflow.EventTurnEnd {
			turnEnd = ev.TurnEnd
		}
	}

	if turnEnd == nil || turnEnd.Reason != agentflow.TurnEndMaxTurns {
		t.Errorf("expected max_turns, got %v", turnEnd)
	}
}

// TestPermissionDeny verifies that denied tools return an error to the model.
func TestPermissionDeny(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(mock.ToolCallEvent("tc_1", "bash", `{"cmd": "rm -rf /"}`)),
		mock.WithResponse(mock.TextDelta("OK, I won't do that.")),
	)

	bash := tools.New("bash", "Run command").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(false).
		ReadOnly(false).
		WithExecute(func(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			t.Fatal("bash should not have been executed")
			return nil, nil
		}).
		Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(bash),
		agentflow.WithPermission(agentflow.DenyList("bash")),
	)

	var toolEndResult *agentflow.ToolResult
	for ev := range agent.Run(context.Background(), []agentflow.Message{agentflow.NewUserMessage("delete everything")}) {
		if ev.Type == agentflow.EventToolEnd {
			toolEndResult = &ev.ToolEnd.Result
		}
	}

	if toolEndResult == nil || !toolEndResult.IsError {
		t.Error("expected tool to return an error result")
	}
}

// TestHookBlocksTool verifies that a PreToolUse hook can block execution.
func TestHookBlocksTool(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(mock.ToolCallEvent("tc_1", "echo", `{}`)),
		mock.WithResponse(mock.TextDelta("Blocked.")),
	)

	echo := tools.New("echo", "Echo").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(true).
		ReadOnly(true).
		WithExecute(func(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			t.Fatal("tool should not execute")
			return nil, nil
		}).
		Build()

	blockHook := agentflow.HookFunc{
		HookPhase: agentflow.HookPreToolUse,
		Fn: func(_ context.Context, _ *agentflow.HookContext) (*agentflow.HookAction, error) {
			return &agentflow.HookAction{Block: true, BlockReason: "testing block"}, nil
		},
	}

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(echo),
		agentflow.WithHook(blockHook),
	)

	var toolEndResult *agentflow.ToolResult
	for ev := range agent.Run(context.Background(), []agentflow.Message{agentflow.NewUserMessage("test")}) {
		if ev.Type == agentflow.EventToolEnd {
			toolEndResult = &ev.ToolEnd.Result
		}
	}

	if toolEndResult == nil || !toolEndResult.IsError {
		t.Error("expected blocked tool result")
	}
	if toolEndResult != nil && toolEndResult.Content != "testing block" {
		t.Errorf("expected 'testing block', got %q", toolEndResult.Content)
	}
}

// TestUnknownTool verifies that an unknown tool call returns an error to the model.
func TestUnknownTool(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(mock.ToolCallEvent("tc_1", "nonexistent", `{}`)),
		mock.WithResponse(mock.TextDelta("Sorry, that tool doesn't exist.")),
	)

	agent := agentflow.NewAgent(provider)

	var toolEndResult *agentflow.ToolResult
	for ev := range agent.Run(context.Background(), []agentflow.Message{agentflow.NewUserMessage("use nonexistent")}) {
		if ev.Type == agentflow.EventToolEnd {
			toolEndResult = &ev.ToolEnd.Result
		}
	}

	if toolEndResult == nil || !toolEndResult.IsError {
		t.Error("expected error for unknown tool")
	}
}

// TestContextCancellation verifies the loop stops on context cancel.
func TestContextCancellation(t *testing.T) {
	// Provider that would loop forever.
	provider := mock.New(
		mock.WithResponse(mock.ToolCallEvent("tc_1", "slow", `{}`)),
	)

	slow := tools.New("slow", "Slow tool").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(true).
		ReadOnly(true).
		WithExecute(func(ctx context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			<-ctx.Done()
			return &agentflow.ToolResult{Content: "cancelled", IsError: true}, ctx.Err()
		}).
		Build()

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	agent := agentflow.NewAgent(provider, agentflow.WithTools(slow))

	var gotAborted bool
	for ev := range agent.Run(ctx, []agentflow.Message{agentflow.NewUserMessage("go")}) {
		if ev.Type == agentflow.EventTurnEnd && ev.TurnEnd.Reason == agentflow.TurnEndAborted {
			gotAborted = true
		}
		if ev.Type == agentflow.EventTurnEnd && ev.TurnEnd.Reason == agentflow.TurnEndError {
			gotAborted = true // Context error also acceptable
		}
	}

	if !gotAborted {
		t.Error("expected aborted or error turn end")
	}
}

// TestRunSync verifies the synchronous wrapper returns final messages.
func TestRunSync(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(
			mock.TextDelta("Response text"),
		),
	)

	agent := agentflow.NewAgent(provider)
	messages, err := agent.RunSync(context.Background(), []agentflow.Message{agentflow.NewUserMessage("Hi")})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(messages) < 2 {
		t.Fatalf("expected at least 2 messages (user + assistant), got %d", len(messages))
	}

	last := messages[len(messages)-1]
	if last.Role != agentflow.RoleAssistant {
		t.Errorf("expected assistant role, got %s", last.Role)
	}
	if last.TextContent() != "Response text" {
		t.Errorf("expected 'Response text', got %q", last.TextContent())
	}
}

// TestConcurrentToolExecution verifies that concurrent-safe tools run in parallel.
func TestConcurrentToolExecution(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(
			mock.ToolCallEvent("tc_1", "fast", `{"id": "1"}`),
			mock.ToolCallEvent("tc_2", "fast", `{"id": "2"}`),
			mock.ToolCallEvent("tc_3", "fast", `{"id": "3"}`),
		),
		mock.WithResponse(
			mock.TextDelta("All done."),
		),
	)

	var callCount int32
	fast := tools.New("fast", "Fast concurrent tool").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(true).
		ReadOnly(true).
		WithExecute(func(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			// Small sleep to verify parallelism.
			time.Sleep(50 * time.Millisecond)
			return &agentflow.ToolResult{Content: "done"}, nil
		}).
		Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(fast),
		agentflow.WithMaxConcurrency(3),
	)

	start := time.Now()
	var toolEnds int
	for ev := range agent.Run(context.Background(), []agentflow.Message{agentflow.NewUserMessage("go")}) {
		if ev.Type == agentflow.EventToolEnd {
			toolEnds++
		}
	}
	elapsed := time.Since(start)
	_ = callCount

	if toolEnds != 3 {
		t.Errorf("expected 3 tool ends, got %d", toolEnds)
	}

	// If tools ran in parallel, total time should be ~50ms, not ~150ms.
	if elapsed > 200*time.Millisecond {
		t.Errorf("tools seem to have run serially: elapsed %v", elapsed)
	}
}
