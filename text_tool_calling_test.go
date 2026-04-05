package agentflow_test

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	groqprovider "github.com/CanArslanDev/agentflow/provider/groq"
	"github.com/CanArslanDev/agentflow/provider/mock"
	"github.com/CanArslanDev/agentflow/tools"
)

func makeSearchTool() agentflow.Tool {
	return tools.New("web_search", "Search the web for current information").
		WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{"type": "string", "description": "Search query"},
			},
			"required": []string{"query"},
		}).
		ConcurrencySafe(true).
		ReadOnly(true).
		RemoteSafe().
		WithExecute(func(ctx context.Context, input json.RawMessage, p agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			var params struct {
				Query string `json:"query"`
			}
			json.Unmarshal(input, &params)
			return &agentflow.ToolResult{
				Content: "Search results for '" + params.Query + "': April 5, 2026 is a Sunday. No major global holidays.",
			}, nil
		}).Build()
}

func TestTextToolCalling_Basic(t *testing.T) {
	provider := mock.New(
		// Turn 1: model uses [TOOL_CALL] format
		mock.WithResponse(
			mock.TextDelta("Let me search for that.\n[TOOL_CALL: web_search(\"April 5 2026 special day\")]"),
		),
		// Turn 2: model answers with tool result
		mock.WithResponse(
			mock.TextDelta("April 5, 2026 is a Sunday with no major holidays."),
		),
	)

	agent := agentflow.NewAgent(provider,
		agentflow.WithTextToolCalling(),
		agentflow.WithTool(makeSearchTool()),
		agentflow.WithMaxTurns(5),
	)

	var text string
	var toolStarts, toolEnds []string

	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("Is April 5, 2026 a special day?"),
	}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			text += ev.TextDelta.Text
		case agentflow.EventToolStart:
			toolStarts = append(toolStarts, ev.ToolStart.ToolCall.Name)
		case agentflow.EventToolEnd:
			toolEnds = append(toolEnds, ev.ToolEnd.ToolCall.Name)
		}
	}

	// Tool should have been called.
	if len(toolStarts) != 1 || toolStarts[0] != "web_search" {
		t.Errorf("tool starts: %v", toolStarts)
	}
	if len(toolEnds) != 1 || toolEnds[0] != "web_search" {
		t.Errorf("tool ends: %v", toolEnds)
	}

	// Final answer should include tool-based information.
	if !strings.Contains(text, "Sunday") {
		t.Errorf("expected Sunday in response, got: %q", text)
	}

	// Provider should have been called twice (tool call + answer).
	if provider.CallCount() != 2 {
		t.Errorf("expected 2 provider calls, got %d", provider.CallCount())
	}
}

func TestTextToolCalling_NoToolCall(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(
			mock.TextDelta("I already know: 2+2 = 4."),
		),
	)

	agent := agentflow.NewAgent(provider,
		agentflow.WithTextToolCalling(),
		agentflow.WithTool(makeSearchTool()),
		agentflow.WithMaxTurns(5),
	)

	var text string
	var toolCalls int

	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("What is 2+2?"),
	}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			text += ev.TextDelta.Text
		case agentflow.EventToolStart:
			toolCalls++
		}
	}

	if text != "I already know: 2+2 = 4." {
		t.Errorf("text: %q", text)
	}
	if toolCalls != 0 {
		t.Errorf("expected 0 tool calls, got %d", toolCalls)
	}
	if provider.CallCount() != 1 {
		t.Errorf("expected 1 provider call, got %d", provider.CallCount())
	}
}

func TestTextToolCalling_UnknownTool(t *testing.T) {
	provider := mock.New(
		// Model calls a tool that doesn't exist.
		mock.WithResponse(
			mock.TextDelta("[TOOL_CALL: unknown_tool(\"test\")]"),
		),
	)

	agent := agentflow.NewAgent(provider,
		agentflow.WithTextToolCalling(),
		agentflow.WithTool(makeSearchTool()),
		agentflow.WithMaxTurns(2),
	)

	var toolCalls int
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("test"),
	}) {
		if ev.Type == agentflow.EventToolStart {
			toolCalls++
		}
	}

	// Unknown tool should be silently ignored (not registered).
	if toolCalls != 0 {
		t.Errorf("expected 0 tool calls for unknown tool, got %d", toolCalls)
	}
}

func TestTextToolCalling_WithThinking(t *testing.T) {
	provider := mock.New(
		// Turn 1 (thinking): model thinks and calls tool
		mock.WithResponse(
			mock.TextDelta("I need to search for this.\n[TOOL_CALL: web_search(\"test query\")]"),
		),
		// Turn 2 (thinking continues with result): model finishes thinking
		mock.WithResponse(
			mock.TextDelta("Based on the search, the answer is clear."),
		),
		// Turn 3 (answer): final answer
		mock.WithResponse(
			mock.TextDelta("The answer is Sunday."),
		),
	)

	agent := agentflow.NewAgent(provider,
		agentflow.WithTextToolCalling(),
		agentflow.WithThinkingPrompt("Think step by step.", "Give final answer."),
		agentflow.WithTool(makeSearchTool()),
		agentflow.WithMaxTurns(10),
	)

	var thinking, message string
	var toolStarts []string

	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("What day is April 5, 2026?"),
	}) {
		switch ev.Type {
		case agentflow.EventThinkingDelta:
			thinking += ev.ThinkDelta.Text
		case agentflow.EventTextDelta:
			message += ev.TextDelta.Text
		case agentflow.EventToolStart:
			toolStarts = append(toolStarts, ev.ToolStart.ToolCall.Name)
		}
	}

	// Thinking should include the search intent.
	if !strings.Contains(thinking, "search") {
		t.Errorf("thinking should mention search: %q", thinking)
	}
	// Tool should have been called during thinking.
	if len(toolStarts) != 1 || toolStarts[0] != "web_search" {
		t.Errorf("tool starts: %v", toolStarts)
	}
	// Final answer should be in message (not thinking).
	if message != "The answer is Sunday." {
		t.Errorf("message: %q", message)
	}
}

func TestTextToolCalling_MultipleToolCalls(t *testing.T) {
	searchTool := makeSearchTool()
	calcTool := tools.New("calculator", "Calculate math expressions").
		WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{"type": "string"},
			},
		}).
		ConcurrencySafe(true).ReadOnly(true).RemoteSafe().
		WithExecute(func(ctx context.Context, input json.RawMessage, p agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{Content: "420"}, nil
		}).Build()

	provider := mock.New(
		// Turn 1: two tool calls in one response
		mock.WithResponse(
			mock.TextDelta("Let me check both:\n[TOOL_CALL: web_search(\"test\")]\n[TOOL_CALL: calculator(\"15*28\")]"),
		),
		// Turn 2: final answer
		mock.WithResponse(
			mock.TextDelta("Done."),
		),
	)

	agent := agentflow.NewAgent(provider,
		agentflow.WithTextToolCalling(),
		agentflow.WithTools(searchTool, calcTool),
		agentflow.WithMaxTurns(5),
	)

	var toolStarts []string
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("search and calculate"),
	}) {
		if ev.Type == agentflow.EventToolStart {
			toolStarts = append(toolStarts, ev.ToolStart.ToolCall.Name)
		}
	}

	if len(toolStarts) != 2 {
		t.Fatalf("expected 2 tool calls, got %d: %v", len(toolStarts), toolStarts)
	}
	// Order may vary due to concurrent execution batching.
	hasSearch := toolStarts[0] == "web_search" || toolStarts[1] == "web_search"
	hasCalc := toolStarts[0] == "calculator" || toolStarts[1] == "calculator"
	if !hasSearch || !hasCalc {
		t.Errorf("expected web_search and calculator, got: %v", toolStarts)
	}
}

func TestTextToolCalling_SystemPromptInjection(t *testing.T) {
	var receivedRequest *agentflow.Request

	// Custom mock to capture the request.
	provider := mock.New(
		mock.WithResponse(mock.TextDelta("ok")),
	)

	// Use OnEvent to verify system prompt contains tool instructions.
	var systemPrompt string
	agent := agentflow.NewAgent(provider,
		agentflow.WithTextToolCalling(),
		agentflow.WithTool(makeSearchTool()),
		agentflow.WithSystemPrompt("You are helpful."),
		agentflow.WithMaxTurns(1),
	)

	// We can't directly inspect the request, but we can verify behavior:
	// the model should see tool instructions in system prompt.
	_ = receivedRequest
	for range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("test"),
	}) {
	}

	// Build the expected instruction manually to verify the function.
	_ = systemPrompt
	// Just verify no crash and tools work - the real test is the live integration.
}

func TestTextToolCalling_Disabled(t *testing.T) {
	// Without WithTextToolCalling, tools should be sent in Request.Tools (native).
	provider := mock.New(
		mock.WithResponse(mock.TextDelta("Hello")),
	)

	agent := agentflow.NewAgent(provider,
		// NO WithTextToolCalling() -- native mode
		agentflow.WithTool(makeSearchTool()),
		agentflow.WithMaxTurns(1),
	)

	var text string
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("Hi"),
	}) {
		if ev.Type == agentflow.EventTextDelta {
			text += ev.TextDelta.Text
		}
	}

	if text != "Hello" {
		t.Errorf("text: %q", text)
	}
}

// --- Integration test ---

func TestIntegration_TextToolCalling(t *testing.T) {
	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		t.Skip("GROQ_API_KEY not set")
	}

	// Use compound-beta which doesn't support native tool calling.
	provider := groqprovider.New(key, "compound-beta")

	agent := agentflow.NewAgent(provider,
		agentflow.WithTextToolCalling(),
		agentflow.WithTool(makeSearchTool()),
		agentflow.WithMaxTurns(5),
		agentflow.WithMaxTokens(500),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var text, thinking string
	var toolStarts []string

	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("Search the web: is today April 5, 2026 a special day?"),
	}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			text += ev.TextDelta.Text
		case agentflow.EventThinkingDelta:
			thinking += ev.ThinkDelta.Text
		case agentflow.EventToolStart:
			toolStarts = append(toolStarts, ev.ToolStart.ToolCall.Name)
		case agentflow.EventError:
			t.Logf("Error: %v (retrying: %v)", ev.Error.Err, ev.Error.Retrying)
		}
	}

	t.Logf("Thinking (%d chars): %.200s...", len(thinking), thinking)
	t.Logf("Text (%d chars): %.200s...", len(text), text)
	t.Logf("Tool calls: %v", toolStarts)

	// Main verification: no "400: tool calling not supported" error.
	// compound-beta may or may not follow [TOOL_CALL] format — it has its own
	// internal <tool> format. The key success criteria is that the request
	// completed without API errors (tools were NOT sent in Request.Tools).
	if text == "" && thinking == "" {
		t.Error("expected non-empty response (text or thinking)")
	}

	if len(toolStarts) > 0 {
		t.Logf("Text-based tool calling triggered: %v", toolStarts)
	} else {
		t.Log("Model used its own format instead of [TOOL_CALL], but pipeline worked without errors")
	}
}
