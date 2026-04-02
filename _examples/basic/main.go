// Basic example: a minimal agent with a working calculator tool.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/groq"
	"github.com/CanArslanDev/agentflow/tools"
)

func main() {
	apiKey := os.Getenv("GROQ_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "GROQ_API_KEY environment variable required")
		os.Exit(1)
	}

	provider := groq.New(apiKey, "llama-3.3-70b-versatile")

	calculator := tools.New("calculator", "Evaluate a mathematical expression with two numbers. Supports add, subtract, multiply, divide.").
		WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"a":        map[string]any{"type": "number", "description": "First number"},
				"b":        map[string]any{"type": "number", "description": "Second number"},
				"operator": map[string]any{"type": "string", "description": "Operator: add, subtract, multiply, divide"},
			},
			"required": []string{"a", "b", "operator"},
		}).
		ConcurrencySafe(true).
		ReadOnly(true).
		RemoteSafe().
		WithExecute(func(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			var p struct {
				A        float64 `json:"a"`
				B        float64 `json:"b"`
				Operator string  `json:"operator"`
			}
			if err := json.Unmarshal(input, &p); err != nil {
				return &agentflow.ToolResult{Content: "invalid input: " + err.Error(), IsError: true}, nil
			}

			var result float64
			switch strings.ToLower(p.Operator) {
			case "add", "+":
				result = p.A + p.B
			case "subtract", "-":
				result = p.A - p.B
			case "multiply", "*":
				result = p.A * p.B
			case "divide", "/":
				if p.B == 0 {
					return &agentflow.ToolResult{Content: "division by zero", IsError: true}, nil
				}
				result = p.A / p.B
			default:
				return &agentflow.ToolResult{Content: "unknown operator: " + p.Operator, IsError: true}, nil
			}

			return &agentflow.ToolResult{
				Content: strconv.FormatFloat(result, 'f', -1, 64),
			}, nil
		}).
		Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(calculator),
		agentflow.WithSystemPrompt("You are a helpful assistant. Use the calculator tool for math operations."),
		agentflow.WithMaxTurns(5),
	)

	messages := []agentflow.Message{
		agentflow.NewUserMessage("What is 15 multiplied by 28?"),
	}

	for ev := range agent.Run(context.Background(), messages) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			fmt.Print(ev.TextDelta.Text)
		case agentflow.EventToolStart:
			fmt.Printf("\n[tool: %s]\n", ev.ToolStart.ToolCall.Name)
		case agentflow.EventToolEnd:
			fmt.Printf("[result: %s]\n\n", ev.ToolEnd.Result.Content)
		case agentflow.EventTurnEnd:
			fmt.Printf("\n--- turn %d: %s ---\n", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		case agentflow.EventError:
			fmt.Fprintf(os.Stderr, "Error: %v\n", ev.Error.Err)
		}
	}
}
