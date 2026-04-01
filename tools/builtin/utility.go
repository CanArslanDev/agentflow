package builtin

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/canarslan/agentflow"
)

// --- Sleep ---

// Sleep returns a tool that pauses execution for a specified duration.
// Useful for rate limiting, polling intervals, or timed workflows.
func Sleep() agentflow.Tool { return &sleepTool{} }

type sleepTool struct{}

func (t *sleepTool) Name() string { return "sleep" }
func (t *sleepTool) Description() string {
	return "Pause execution for a specified number of seconds. Useful for rate limiting or waiting between operations."
}
func (t *sleepTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"seconds": map[string]any{
				"type":        "number",
				"description": "Number of seconds to sleep (max 60)",
			},
		},
		"required": []string{"seconds"},
	}
}

func (t *sleepTool) Execute(ctx context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var params struct {
		Seconds float64 `json:"seconds"`
	}
	if err := json.Unmarshal(input, &params); err != nil {
		return &agentflow.ToolResult{Content: "invalid input: " + err.Error(), IsError: true}, nil
	}

	if params.Seconds <= 0 {
		return &agentflow.ToolResult{Content: "seconds must be positive", IsError: true}, nil
	}
	if params.Seconds > 60 {
		params.Seconds = 60
	}

	duration := time.Duration(params.Seconds * float64(time.Second))
	select {
	case <-time.After(duration):
		return &agentflow.ToolResult{
			Content: fmt.Sprintf("Slept for %.1f seconds", params.Seconds),
		}, nil
	case <-ctx.Done():
		return &agentflow.ToolResult{Content: "sleep cancelled", IsError: true}, nil
	}
}

func (t *sleepTool) IsConcurrencySafe(_ json.RawMessage) bool { return true }
func (t *sleepTool) IsReadOnly(_ json.RawMessage) bool        { return true }
func (t *sleepTool) Locality() agentflow.ToolLocality          { return agentflow.ToolAny }

// --- AskUser ---

// AskUser returns a tool that prompts the user for input via a callback.
// The askFn is called with the question and should return the user's response.
//
//	askTool := builtin.AskUser(func(ctx context.Context, question string) (string, error) {
//	    fmt.Print(question + " > ")
//	    var answer string
//	    fmt.Scanln(&answer)
//	    return answer, nil
//	})
func AskUser(askFn func(ctx context.Context, question string) (string, error)) agentflow.Tool {
	return &askUserTool{askFn: askFn}
}

type askUserTool struct {
	askFn func(ctx context.Context, question string) (string, error)
}

func (t *askUserTool) Name() string { return "ask_user" }
func (t *askUserTool) Description() string {
	return "Ask the user a question and wait for their response. Use when you need clarification, confirmation, or additional information from the user."
}
func (t *askUserTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"question": map[string]any{
				"type":        "string",
				"description": "The question to ask the user",
			},
		},
		"required": []string{"question"},
	}
}

func (t *askUserTool) Execute(ctx context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var params struct {
		Question string `json:"question"`
	}
	if err := json.Unmarshal(input, &params); err != nil {
		return &agentflow.ToolResult{Content: "invalid input: " + err.Error(), IsError: true}, nil
	}

	if t.askFn == nil {
		return &agentflow.ToolResult{Content: "ask_user: no callback configured", IsError: true}, nil
	}

	answer, err := t.askFn(ctx, params.Question)
	if err != nil {
		return &agentflow.ToolResult{Content: "user interaction failed: " + err.Error(), IsError: true}, nil
	}

	return &agentflow.ToolResult{Content: answer}, nil
}

func (t *askUserTool) IsConcurrencySafe(_ json.RawMessage) bool { return false }
func (t *askUserTool) IsReadOnly(_ json.RawMessage) bool        { return true }
func (t *askUserTool) Locality() agentflow.ToolLocality          { return agentflow.ToolAny }
