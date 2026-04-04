package agentflow_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/compactor"
	"github.com/CanArslanDev/agentflow/provider/mock"
	"github.com/CanArslanDev/agentflow/tools"
)

func benchProvider(n int, responses ...[]agentflow.StreamEvent) *mock.Provider {
	opts := make([]mock.ProviderOption, 0, n*len(responses))
	for i := 0; i < n; i++ {
		for _, r := range responses {
			opts = append(opts, mock.WithResponse(r...))
		}
	}
	return mock.New(opts...)
}

func BenchmarkToolExecution_SingleTool(b *testing.B) {
	provider := benchProvider(b.N,
		[]agentflow.StreamEvent{mock.ToolCallEvent("tc1", "noop", `{}`)},
		[]agentflow.StreamEvent{mock.TextDelta("done")},
	)

	noop := tools.New("noop", "No-op").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(true).ReadOnly(true).
		WithExecute(func(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{Content: "ok"}, nil
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(noop),
		agentflow.WithMaxTurns(3),
	)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for range agent.Run(context.Background(), []agentflow.Message{
			agentflow.NewUserMessage("go"),
		}) {
		}
	}
}

func BenchmarkEventDelivery(b *testing.B) {
	provider := benchProvider(b.N,
		[]agentflow.StreamEvent{mock.TextDelta("hello world")},
	)

	agent := agentflow.NewAgent(provider, agentflow.WithMaxTurns(1))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for range agent.Run(context.Background(), []agentflow.Message{
			agentflow.NewUserMessage("go"),
		}) {
		}
	}
}

func BenchmarkToolExecution_MixedConcurrency(b *testing.B) {
	safeTool := tools.New("safe", "Safe").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(true).ReadOnly(true).
		WithExecute(func(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{Content: "ok"}, nil
		}).Build()

	unsafeTool := tools.New("unsafe", "Unsafe").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(false).ReadOnly(false).
		WithExecute(func(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{Content: "ok"}, nil
		}).Build()

	provider := benchProvider(b.N,
		[]agentflow.StreamEvent{
			mock.ToolCallEvent("tc1", "safe", `{}`),
			mock.ToolCallEvent("tc2", "safe", `{}`),
			mock.ToolCallEvent("tc3", "unsafe", `{}`),
		},
		[]agentflow.StreamEvent{mock.TextDelta("done")},
	)

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(safeTool, unsafeTool),
		agentflow.WithMaxTurns(3),
		agentflow.WithMaxConcurrency(5),
	)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for range agent.Run(context.Background(), []agentflow.Message{
			agentflow.NewUserMessage("go"),
		}) {
		}
	}
}

func BenchmarkCompaction_SlidingWindow(b *testing.B) {
	c := compactor.NewSlidingWindow(10, 15)

	messages := make([]agentflow.Message, 20)
	for i := range messages {
		messages[i] = agentflow.NewUserMessage("message content for benchmarking")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Compact(context.Background(), messages)
	}
}

func BenchmarkInputValidation(b *testing.B) {
	provider := benchProvider(b.N,
		[]agentflow.StreamEvent{mock.ToolCallEvent("tc1", "validated", `{"query":"test","max_results":10}`)},
		[]agentflow.StreamEvent{mock.TextDelta("done")},
	)

	validated := tools.New("validated", "Test").
		WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query":       map[string]any{"type": "string"},
				"max_results": map[string]any{"type": "integer"},
			},
			"required": []string{"query"},
		}).
		ConcurrencySafe(true).ReadOnly(true).
		WithExecute(func(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{Content: "ok"}, nil
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(validated),
		agentflow.WithMaxTurns(3),
	)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for range agent.Run(context.Background(), []agentflow.Message{
			agentflow.NewUserMessage("go"),
		}) {
		}
	}
}
