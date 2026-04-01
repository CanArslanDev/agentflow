// Basic example: a minimal agent that uses a single tool.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/canarslan/agentflow"
	"github.com/canarslan/agentflow/provider/openrouter"
	"github.com/canarslan/agentflow/tools"
)

func main() {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "OPENROUTER_API_KEY environment variable required")
		os.Exit(1)
	}

	provider := openrouter.New(apiKey, "anthropic/claude-sonnet-4-20250514",
		openrouter.WithReferer("https://github.com/canarslan/agentflow"),
		openrouter.WithTitle("agentflow-example"),
	)

	calculator := tools.New("calculator", "Evaluate a mathematical expression and return the result.").
		WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"expression": map[string]any{
					"type":        "string",
					"description": "The mathematical expression to evaluate (e.g., '2 + 2', '15 * 3')",
				},
			},
			"required": []string{"expression"},
		}).
		ReadOnly(true).
		ConcurrencySafe(true).
		WithExecute(func(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			var params struct {
				Expression string `json:"expression"`
			}
			if err := json.Unmarshal(input, &params); err != nil {
				return &agentflow.ToolResult{Content: err.Error(), IsError: true}, nil
			}
			// Simple demo — in production you'd use a proper expression evaluator.
			return &agentflow.ToolResult{
				Content: fmt.Sprintf("Result of '%s' = 42 (demo)", params.Expression),
			}, nil
		}).
		Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(calculator),
		agentflow.WithSystemPrompt("You are a helpful assistant. Use the calculator tool when asked about math."),
		agentflow.WithMaxTurns(5),
	)

	messages := []agentflow.Message{
		agentflow.NewUserMessage("What is 15 multiplied by 28?"),
	}

	fmt.Println("Agent running...")
	fmt.Println()

	for ev := range agent.Run(context.Background(), messages) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			fmt.Print(ev.TextDelta.Text)
		case agentflow.EventToolStart:
			fmt.Printf("\n⚡ [%s] executing...\n", ev.ToolStart.ToolCall.Name)
		case agentflow.EventToolEnd:
			fmt.Printf("✓ [%s] completed in %v\n\n", ev.ToolEnd.ToolCall.Name, ev.ToolEnd.Duration)
		case agentflow.EventTurnEnd:
			fmt.Printf("\n\n--- Agent finished (reason: %s, turns: %d) ---\n", ev.TurnEnd.Reason, ev.TurnEnd.TurnNumber)
		case agentflow.EventError:
			fmt.Fprintf(os.Stderr, "\nError: %v\n", ev.Error.Err)
		}
	}
}
