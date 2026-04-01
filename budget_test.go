package agentflow_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/canarslan/agentflow"
	"github.com/canarslan/agentflow/provider/mock"
	"github.com/canarslan/agentflow/tools"
)

// TestBudgetExhausted verifies the loop stops when token budget is consumed.
func TestBudgetExhausted(t *testing.T) {
	provider := mock.New(
		// Turn 1: tool call + 500 tokens.
		mock.WithResponse(
			mock.ToolCallEvent("tc_1", "echo", `{}`),
			mock.UsageEvent(300, 200), // 500 total
		),
		// Turn 2: would exceed budget of 600.
		mock.WithResponse(
			mock.TextDelta("Response"),
			mock.UsageEvent(200, 200), // 400 total → cumulative 900 > 600
		),
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
		agentflow.WithTokenBudget(agentflow.TokenBudget{
			MaxTokens:        600,
			WarningThreshold: 0.7,
		}),
		agentflow.WithMaxTurns(10),
	)

	var turnEnd *agentflow.TurnEndEvent
	var budgetWarnings int

	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("go"),
	}) {
		switch ev.Type {
		case agentflow.EventTurnEnd:
			turnEnd = ev.TurnEnd
		case agentflow.EventBudgetWarning:
			budgetWarnings++
			t.Logf("Budget warning: %d/%d tokens (%.0f%%)",
				ev.BudgetWarning.ConsumedTokens,
				ev.BudgetWarning.MaxTokens,
				ev.BudgetWarning.Percentage*100)
		case agentflow.EventUsage:
			t.Logf("Usage: total=%d tokens", ev.Usage.Usage.TotalTokens)
		}
	}

	if turnEnd == nil {
		t.Fatal("expected TurnEnd event")
	}
	if turnEnd.Reason != agentflow.TurnEndBudgetExhausted {
		t.Errorf("expected budget_exhausted, got %s", turnEnd.Reason)
	}
	t.Logf("Loop ended at turn %d with reason: %s", turnEnd.TurnNumber, turnEnd.Reason)
}

// TestBudgetWarningFires verifies the warning event fires at the threshold.
func TestBudgetWarningFires(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(
			mock.TextDelta("Hello"),
			mock.UsageEvent(400, 100), // 500 total → 50% of 1000
		),
		// This won't be called since first turn completes without tool calls.
	)

	agent := agentflow.NewAgent(provider,
		agentflow.WithTokenBudget(agentflow.TokenBudget{
			MaxTokens:        1000,
			WarningThreshold: 0.4, // Warn at 40% → threshold = 400 tokens
		}),
	)

	var warnings int
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("hi"),
	}) {
		if ev.Type == agentflow.EventBudgetWarning {
			warnings++
			if ev.BudgetWarning.ConsumedTokens != 500 {
				t.Errorf("expected 500 consumed, got %d", ev.BudgetWarning.ConsumedTokens)
			}
			if ev.BudgetWarning.MaxTokens != 1000 {
				t.Errorf("expected max 1000, got %d", ev.BudgetWarning.MaxTokens)
			}
		}
	}

	if warnings != 1 {
		t.Errorf("expected exactly 1 warning, got %d", warnings)
	}
}

// TestBudgetWarningFiresOnlyOnce verifies the warning fires exactly once.
func TestBudgetWarningFiresOnlyOnce(t *testing.T) {
	provider := mock.New(
		// Turn 1: 500 tokens → crosses 40% threshold.
		mock.WithResponse(
			mock.ToolCallEvent("tc_1", "echo", `{}`),
			mock.UsageEvent(300, 200),
		),
		// Turn 2: 300 more → still above threshold, should NOT warn again.
		mock.WithResponse(
			mock.TextDelta("Done"),
			mock.UsageEvent(200, 100),
		),
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
		agentflow.WithTokenBudget(agentflow.TokenBudget{
			MaxTokens:        2000,
			WarningThreshold: 0.2, // 400 tokens → warn after turn 1 (500)
		}),
	)

	var warnings int
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("go"),
	}) {
		if ev.Type == agentflow.EventBudgetWarning {
			warnings++
		}
	}

	if warnings != 1 {
		t.Errorf("expected exactly 1 warning (fires once), got %d", warnings)
	}
}

// TestNoBudgetNoRestriction verifies that without a budget, the loop runs freely.
func TestNoBudgetNoRestriction(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(
			mock.TextDelta("Free running"),
			mock.UsageEvent(50000, 50000), // 100K tokens, no limit
		),
	)

	agent := agentflow.NewAgent(provider) // No budget set.

	var turnEnd *agentflow.TurnEndEvent
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("hi"),
	}) {
		if ev.Type == agentflow.EventTurnEnd {
			turnEnd = ev.TurnEnd
		}
	}

	if turnEnd == nil || turnEnd.Reason != agentflow.TurnEndComplete {
		t.Errorf("expected completed without budget, got %v", turnEnd)
	}
}
