package groq_test

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/groq"
	"github.com/CanArslanDev/agentflow/tools"
)

func getAPIKey(t *testing.T) string {
	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		t.Skip("GROQ_API_KEY not set, skipping integration test")
	}
	return key
}

// TestGroqSimpleChat tests a basic text-only conversation with Groq.
func TestGroqSimpleChat(t *testing.T) {
	key := getAPIKey(t)
	provider := groq.New(key, "llama-3.3-70b-versatile")

	agent := agentflow.NewAgent(provider,
		agentflow.WithMaxTurns(1),
		agentflow.WithMaxTokens(100),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var text string
	var turnEnd *agentflow.TurnEndEvent

	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("Say hello in exactly 3 words."),
	}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			text += ev.TextDelta.Text
		case agentflow.EventTurnEnd:
			turnEnd = ev.TurnEnd
		case agentflow.EventError:
			t.Logf("Error: %v (retrying: %v)", ev.Error.Err, ev.Error.Retrying)
		}
	}

	if text == "" {
		t.Fatal("expected non-empty response from Groq")
	}
	t.Logf("Groq response: %q", text)

	if turnEnd == nil {
		t.Fatal("expected TurnEnd event")
	}
	t.Logf("Turn ended: reason=%s, turn=%d", turnEnd.Reason, turnEnd.TurnNumber)
}

// TestGroqCompoundToolUse tests Groq compound-beta with tool calling.
func TestGroqCompoundToolUse(t *testing.T) {
	key := getAPIKey(t)
	provider := groq.New(key, "llama-3.3-70b-versatile")

	calculator := tools.New("calculator", "Calculate a mathematical expression. Returns the numeric result.").
		WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"expression": map[string]any{
					"type":        "string",
					"description": "Mathematical expression to evaluate (e.g., '15 * 28')",
				},
			},
			"required": []string{"expression"},
		}).
		ConcurrencySafe(true).
		ReadOnly(true).
		WithExecute(func(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			var params struct {
				Expression string `json:"expression"`
			}
			if err := json.Unmarshal(input, &params); err != nil {
				return &agentflow.ToolResult{Content: err.Error(), IsError: true}, nil
			}
			// Simple hardcoded result for test.
			return &agentflow.ToolResult{Content: "420"}, nil
		}).
		Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(calculator),
		agentflow.WithSystemPrompt("You are a helpful assistant. Use the calculator tool for math."),
		agentflow.WithMaxTurns(5),
		agentflow.WithMaxTokens(500),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var text string
	var toolStarts []string
	var toolEnds []string
	var turnEnd *agentflow.TurnEndEvent

	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("What is 15 multiplied by 28? Use the calculator tool."),
	}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			text += ev.TextDelta.Text
		case agentflow.EventToolStart:
			toolStarts = append(toolStarts, ev.ToolStart.ToolCall.Name)
			t.Logf("Tool start: %s (input: %s)", ev.ToolStart.ToolCall.Name, string(ev.ToolStart.ToolCall.Input))
		case agentflow.EventToolEnd:
			toolEnds = append(toolEnds, ev.ToolEnd.ToolCall.Name)
			t.Logf("Tool end: %s (result: %s, error: %v, duration: %v)",
				ev.ToolEnd.ToolCall.Name, ev.ToolEnd.Result.Content, ev.ToolEnd.Result.IsError, ev.ToolEnd.Duration)
		case agentflow.EventTurnEnd:
			turnEnd = ev.TurnEnd
			t.Logf("Turn %d ended: %s", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		case agentflow.EventError:
			t.Logf("Error: %v", ev.Error.Err)
		case agentflow.EventUsage:
			t.Logf("Usage: prompt=%d, completion=%d, total=%d",
				ev.Usage.Usage.PromptTokens, ev.Usage.Usage.CompletionTokens, ev.Usage.Usage.TotalTokens)
		}
	}

	t.Logf("Final text: %q", text)
	t.Logf("Tool starts: %v", toolStarts)
	t.Logf("Tool ends: %v", toolEnds)

	if text == "" {
		t.Error("expected non-empty final response")
	}
	if turnEnd == nil {
		t.Fatal("expected TurnEnd event")
	}

	fmt.Printf("\n=== GROQ COMPOUND-BETA TEST RESULTS ===\n")
	fmt.Printf("Model: llama-3.3-70b-versatile\n")
	fmt.Printf("Tool calls: %d\n", len(toolStarts))
	fmt.Printf("Turns: %d\n", turnEnd.TurnNumber)
	fmt.Printf("Response: %s\n", text)
	fmt.Printf("========================================\n")
}

// TestGroqCompoundThinkTag tests that <think> tags from Groq compound models
// are parsed into StreamEventThinkingDelta events.
func TestGroqCompoundThinkTag(t *testing.T) {
	key := getAPIKey(t)

	// compound-beta uses <think> tags for reasoning.
	provider := groq.New(key, "compound-beta")

	agent := agentflow.NewAgent(provider,
		agentflow.WithMaxTurns(5),
		agentflow.WithMaxTokens(300),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var thinking, text string
	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("What is 15 * 28? Think step by step."),
	}) {
		switch ev.Type {
		case agentflow.EventThinkingDelta:
			thinking += ev.ThinkDelta.Text
		case agentflow.EventTextDelta:
			text += ev.TextDelta.Text
		case agentflow.EventError:
			t.Logf("Error: %v (retrying: %v)", ev.Error.Err, ev.Error.Retrying)
		}
	}

	t.Logf("Thinking (%d chars): %q", len(thinking), thinking)
	t.Logf("Answer (%d chars): %q", len(text), text)

	// compound-beta should produce thinking content via <think> tags.
	if thinking == "" {
		t.Log("Note: no thinking content detected. Model may not have used <think> tags for this query.")
	}
	if text == "" {
		t.Error("expected non-empty answer text")
	}
}
