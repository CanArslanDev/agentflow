package agentflow_test

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/openrouter"
	"github.com/CanArslanDev/agentflow/tools"
)

func openrouterKey(t *testing.T) string {
	key := os.Getenv("OPENROUTER_API_KEY")
	if key == "" {
		t.Skip("OPENROUTER_API_KEY not set")
	}
	return key
}

func TestIntegration_OpenRouter_SimpleChat(t *testing.T) {
	provider := openrouter.New(openrouterKey(t), "google/gemini-2.0-flash-001")
	agent := agentflow.NewAgent(provider, agentflow.WithMaxTurns(1), agentflow.WithMaxTokens(50))

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var text string
	for ev := range agent.Run(ctx, []agentflow.Message{agentflow.NewUserMessage("Say 'pong'")}) {
		if ev.Type == agentflow.EventTextDelta {
			text += ev.TextDelta.Text
		}
		if ev.Type == agentflow.EventError {
			t.Logf("Error: %v", ev.Error.Err)
		}
	}

	if text == "" {
		t.Fatal("empty response")
	}
	t.Logf("Response: %q", text)
}

func TestIntegration_OpenRouter_ToolUse(t *testing.T) {
	provider := openrouter.New(openrouterKey(t), "google/gemini-2.0-flash-001")

	weather := tools.New("get_weather", "Get current weather for a city").
		WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string", "description": "City name"},
			},
			"required": []string{"city"},
		}).
		ConcurrencySafe(true).ReadOnly(true).RemoteSafe().
		WithExecute(func(ctx context.Context, input json.RawMessage, p agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{Content: `{"city":"Istanbul","temp_c":22,"condition":"sunny"}`}, nil
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTool(weather),
		agentflow.WithMaxTurns(5),
		agentflow.WithMaxTokens(200),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var text string
	var toolCalls []string
	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("What is the weather in Istanbul?"),
	}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			text += ev.TextDelta.Text
		case agentflow.EventToolStart:
			toolCalls = append(toolCalls, ev.ToolStart.ToolCall.Name)
			t.Logf("Tool call: %s(%s)", ev.ToolStart.ToolCall.Name, string(ev.ToolStart.ToolCall.Input))
		case agentflow.EventError:
			t.Logf("Error: %v (retrying: %v)", ev.Error.Err, ev.Error.Retrying)
		}
	}

	if len(toolCalls) == 0 {
		t.Fatal("model did not call any tools")
	}
	t.Logf("Tool calls: %v", toolCalls)
	t.Logf("Response: %q", text)
}

func TestIntegration_OpenRouter_MultiTool(t *testing.T) {
	provider := openrouter.New(openrouterKey(t), "google/gemini-2.0-flash-001")

	weather := tools.New("get_weather", "Get current weather for a city").
		WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string", "description": "City name"},
			},
			"required": []string{"city"},
		}).
		ConcurrencySafe(true).ReadOnly(true).RemoteSafe().
		WithExecute(func(ctx context.Context, input json.RawMessage, p agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			var params struct{ City string `json:"city"` }
			json.Unmarshal(input, &params)
			return &agentflow.ToolResult{Content: params.City + ": 20C, cloudy"}, nil
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTool(weather),
		agentflow.WithMaxTurns(5),
		agentflow.WithMaxTokens(300),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var text string
	var toolCalls []string
	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("Compare weather in Istanbul and Tokyo"),
	}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			text += ev.TextDelta.Text
		case agentflow.EventToolStart:
			toolCalls = append(toolCalls, ev.ToolStart.ToolCall.Name)
			t.Logf("Tool call: %s(%s)", ev.ToolStart.ToolCall.Name, string(ev.ToolStart.ToolCall.Input))
		case agentflow.EventError:
			t.Logf("Error: %v", ev.Error.Err)
		}
	}

	if len(toolCalls) < 2 {
		t.Fatalf("expected at least 2 tool calls, got %d", len(toolCalls))
	}
	t.Logf("Tool calls: %v", toolCalls)
	t.Logf("Response: %q", text)
}
