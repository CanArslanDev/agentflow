package observability_test

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/observability"
	"github.com/CanArslanDev/agentflow/provider/groq"
	"github.com/CanArslanDev/agentflow/provider/mock"
	"github.com/CanArslanDev/agentflow/tools"
)

func groqProvider(t *testing.T) *groq.Provider {
	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		t.Skip("GROQ_API_KEY not set")
	}
	return groq.New(key, "llama-3.3-70b-versatile")
}

func TestIntegration_Tracer(t *testing.T) {
	provider := groqProvider(t)

	tracer := observability.NewTracer()

	calc := tools.New("calc", "Calculate math").
		WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"expr": map[string]any{"type": "string"},
			},
			"required": []string{"expr"},
		}).
		ConcurrencySafe(true).ReadOnly(true).RemoteSafe().
		WithExecute(func(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			time.Sleep(50 * time.Millisecond)
			return &agentflow.ToolResult{Content: "42"}, nil
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(calc),
		agentflow.WithSystemPrompt("Use calc tool for math."),
		agentflow.WithMaxTurns(3),
		agentflow.WithMaxTokens(300),
		agentflow.WithHook(tracer.Hooks()[0]),
		agentflow.WithHook(tracer.Hooks()[1]),
		agentflow.WithHook(tracer.Hooks()[2]),
		agentflow.WithHook(tracer.Hooks()[3]),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("What is 6 * 7?"),
	}) {
		if ev.Type == agentflow.EventTurnEnd {
			t.Logf("Turn %d: %s", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		}
	}

	trace := tracer.Finish()
	t.Logf("Trace ID: %s", trace.ID)
	t.Logf("Duration: %v", trace.Duration)
	t.Logf("Spans: %d", len(trace.Spans))

	for _, span := range trace.Spans {
		t.Logf("  [%s] %s — %v (%s)", span.Kind, span.Name, span.Duration.Round(time.Millisecond), span.Status)
	}

	if len(trace.Spans) == 0 {
		t.Error("expected at least 1 span")
	}
}

func TestIntegration_CostTracker(t *testing.T) {
	provider := groqProvider(t)

	tracker := observability.NewCostTracker()

	agent := agentflow.NewAgent(provider,
		agentflow.WithSystemPrompt("Reply in one sentence."),
		agentflow.WithMaxTurns(1),
		agentflow.WithMaxTokens(100),
		agentflow.WithOnEvent(tracker.OnEvent),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("What is Go?"),
	}) {
		if ev.Type == agentflow.EventUsage {
			t.Logf("Usage: prompt=%d, completion=%d",
				ev.Usage.Usage.PromptTokens, ev.Usage.Usage.CompletionTokens)
		}
	}

	prompt, completion := tracker.TotalTokens()
	cost := tracker.TotalCost()

	t.Logf("Total tokens: prompt=%d, completion=%d", prompt, completion)
	t.Logf("Estimated cost: $%.6f", cost)

	if prompt == 0 {
		t.Error("expected non-zero prompt tokens")
	}
	if cost <= 0 {
		t.Error("expected positive cost estimate")
	}

	records := tracker.Records()
	t.Logf("Usage records: %d", len(records))
	if len(records) == 0 {
		t.Error("expected at least 1 usage record")
	}
}

func TestIntegration_CostTrackerWithModel(t *testing.T) {
	tracker := observability.NewCostTracker()

	tracker.TrackUsage("llama-3.3-70b-versatile", 1000, 500)
	tracker.TrackUsage("llama-3.3-70b-versatile", 2000, 1000)

	cost := tracker.TotalCost()
	prompt, completion := tracker.TotalTokens()

	t.Logf("Simulated: prompt=%d, completion=%d, cost=$%.6f", prompt, completion, cost)

	if cost < 0.002 || cost > 0.004 {
		t.Errorf("expected cost ~$0.003, got $%.6f", cost)
	}
}

func TestIntegration_TracerWithToolCalls(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(
			mock.ToolCallEvent("tc1", "test_tool", `{}`),
			mock.UsageEvent(100, 50),
		),
		mock.WithResponse(
			mock.TextDelta("Done."),
			mock.UsageEvent(150, 30),
		),
	)

	tracer := observability.NewTracer()

	testTool := tools.New("test_tool", "Test").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(true).ReadOnly(true).
		WithExecute(func(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			time.Sleep(25 * time.Millisecond)
			return &agentflow.ToolResult{Content: "ok"}, nil
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(testTool),
		agentflow.WithMaxTurns(5),
		agentflow.WithHook(tracer.Hooks()[0]),
		agentflow.WithHook(tracer.Hooks()[1]),
		agentflow.WithHook(tracer.Hooks()[2]),
		agentflow.WithHook(tracer.Hooks()[3]),
	)

	for range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("test"),
	}) {
	}

	trace := tracer.Finish()

	if len(trace.Spans) < 2 {
		t.Errorf("expected at least 2 spans (model + tool), got %d", len(trace.Spans))
	}

	var hasModelSpan, hasToolSpan bool
	for _, span := range trace.Spans {
		t.Logf("Span: %s (%s) — %v", span.Name, span.Kind, span.Duration)
		if span.Kind == observability.SpanKindModelCall {
			hasModelSpan = true
		}
		if span.Kind == observability.SpanKindToolExecution {
			hasToolSpan = true
			if span.Duration < 20*time.Millisecond {
				t.Errorf("tool span too short: %v (expected ~25ms)", span.Duration)
			}
		}
	}

	if !hasModelSpan {
		t.Error("expected model_call span")
	}
	if !hasToolSpan {
		t.Error("expected tool_execution span")
	}
}

func TestIntegration_CostAndTracer(t *testing.T) {
	provider := groqProvider(t)

	tracer := observability.NewTracer()
	tracker := observability.NewCostTracker()

	agent := agentflow.NewAgent(provider,
		agentflow.WithSystemPrompt("Reply briefly."),
		agentflow.WithMaxTurns(1),
		agentflow.WithMaxTokens(50),
		agentflow.WithHook(tracer.Hooks()[0]),
		agentflow.WithHook(tracer.Hooks()[1]),
		agentflow.WithHook(tracer.Hooks()[2]),
		agentflow.WithHook(tracer.Hooks()[3]),
		agentflow.WithOnEvent(tracker.OnEvent),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	for range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("Say hello"),
	}) {
	}

	trace := tracer.Finish()
	cost := tracker.TotalCost()
	prompt, completion := tracker.TotalTokens()

	t.Logf("Trace: %d spans, %v duration", len(trace.Spans), trace.Duration)
	t.Logf("Cost: $%.6f (prompt=%d, completion=%d)", cost, prompt, completion)

	if len(trace.Spans) == 0 {
		t.Error("expected spans from tracer")
	}
	if cost <= 0 {
		t.Error("expected positive cost from tracker")
	}
}
