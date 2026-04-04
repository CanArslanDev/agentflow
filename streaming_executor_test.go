package agentflow_test

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/groq"
	"github.com/CanArslanDev/agentflow/tools"
)

func groqProvider(t *testing.T) *groq.Provider {
	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		t.Skip("GROQ_API_KEY not set")
	}
	return groq.New(key, "llama-3.3-70b-versatile")
}

// ==========================================================================
// T1.1: STREAMING TOOL EXECUTION
// ==========================================================================

// TestIntegration_Tier1_StreamingExec_Parallel — model 2 tool call yapar,
// streaming executor paralel calistirir.
func TestIntegration_Tier1_StreamingExec_Parallel(t *testing.T) {
	provider := groqProvider(t)

	weather := tools.New("get_weather", "Get weather for a city. Returns temperature.").
		WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string"},
			},
			"required": []string{"city"},
		}).
		ConcurrencySafe(true).ReadOnly(true).RemoteSafe().
		WithExecute(func(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			var p struct{ City string `json:"city"` }
			json.Unmarshal(input, &p)
			time.Sleep(100 * time.Millisecond) // Simulate latency.
			return &agentflow.ToolResult{Content: p.City + ": 22C sunny"}, nil
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(weather),
		agentflow.WithSystemPrompt("You are a weather assistant. When asked about multiple cities, call get_weather for each city."),
		agentflow.WithMaxTurns(5),
		agentflow.WithMaxTokens(500),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var text string
	var toolCalls int
	start := time.Now()

	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("What is the weather in Istanbul and Tokyo?"),
	}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			text += ev.TextDelta.Text
		case agentflow.EventToolStart:
			toolCalls++
			t.Logf("[%v] Tool start: %s", time.Since(start).Round(time.Millisecond), ev.ToolStart.ToolCall.Name)
		case agentflow.EventToolEnd:
			t.Logf("[%v] Tool end: %s (%v)", time.Since(start).Round(time.Millisecond), ev.ToolEnd.ToolCall.Name, ev.ToolEnd.Duration)
		case agentflow.EventTurnEnd:
			t.Logf("[%v] Turn %d: %s", time.Since(start).Round(time.Millisecond), ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		}
	}

	t.Logf("Response: %s", text)
	if toolCalls < 2 {
		t.Errorf("expected at least 2 tool calls, got %d", toolCalls)
	}
}

// TestIntegration_Tier1_StreamingExec_SingleTool — tek tool call da dogru calisir.
func TestIntegration_Tier1_StreamingExec_SingleTool(t *testing.T) {
	provider := groqProvider(t)

	calc := tools.New("calculator", "Calculate math").
		WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"a": map[string]any{"type": "number"}, "b": map[string]any{"type": "number"},
				"op": map[string]any{"type": "string"},
			},
			"required": []string{"a", "b", "op"},
		}).
		ConcurrencySafe(true).ReadOnly(true).RemoteSafe().
		WithExecute(func(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			var p struct{ A, B float64; Op string }
			json.Unmarshal(input, &p)
			return &agentflow.ToolResult{Content: "420"}, nil
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(calc),
		agentflow.WithSystemPrompt("Use the calculator tool for math."),
		agentflow.WithMaxTurns(3),
		agentflow.WithMaxTokens(300),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var text string
	var toolCalls int
	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("What is 15 * 28?"),
	}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			text += ev.TextDelta.Text
		case agentflow.EventToolStart:
			toolCalls++
		case agentflow.EventTurnEnd:
			t.Logf("Turn %d: %s", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		}
	}

	t.Logf("Response: %s", text)
	if toolCalls == 0 {
		t.Error("expected tool call")
	}
	if !strings.Contains(text, "420") {
		t.Errorf("expected '420' in response, got: %s", text)
	}
}
