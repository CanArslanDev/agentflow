package builtin

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"time"

	"github.com/CanArslanDev/agentflow"
)

const defaultBashTimeout = 120 * time.Second

type bashInput struct {
	Command string `json:"command"`
	Timeout int    `json:"timeout,omitempty"` // milliseconds
}

// Bash returns a tool that executes shell commands via /bin/sh.
// Non-concurrent-safe and non-read-only — mutations are expected.
func Bash() agentflow.Tool {
	return &bashTool{}
}

type bashTool struct{}

func (t *bashTool) Name() string        { return "bash" }
func (t *bashTool) Description() string {
	return "Execute a shell command and return its stdout and stderr. Use for running scripts, installing packages, building code, or any system operation."
}

func (t *bashTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"command": map[string]any{
				"type":        "string",
				"description": "The shell command to execute",
			},
			"timeout": map[string]any{
				"type":        "integer",
				"description": "Optional timeout in milliseconds (default: 120000)",
			},
		},
		"required": []string{"command"},
	}
}

func (t *bashTool) Execute(ctx context.Context, input json.RawMessage, progress agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var params bashInput
	if err := json.Unmarshal(input, &params); err != nil {
		return &agentflow.ToolResult{Content: "invalid input: " + err.Error(), IsError: true}, nil
	}

	timeout := defaultBashTimeout
	if params.Timeout > 0 {
		timeout = time.Duration(params.Timeout) * time.Millisecond
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	if progress != nil {
		progress(agentflow.ProgressEvent{Message: "Executing: " + params.Command})
	}

	cmd := exec.CommandContext(ctx, "/bin/sh", "-c", params.Command)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()

	var result string
	if stdout.Len() > 0 {
		result = stdout.String()
	}
	if stderr.Len() > 0 {
		if result != "" {
			result += "\n"
		}
		result += "STDERR:\n" + stderr.String()
	}

	if err != nil {
		if result == "" {
			result = err.Error()
		} else {
			result = fmt.Sprintf("Exit code: %v\n%s", err, result)
		}
		return &agentflow.ToolResult{Content: result, IsError: true}, nil
	}

	if result == "" {
		result = "(no output)"
	}

	return &agentflow.ToolResult{Content: result}, nil
}

func (t *bashTool) IsConcurrencySafe(_ json.RawMessage) bool { return false }
func (t *bashTool) IsReadOnly(_ json.RawMessage) bool        { return false }
func (t *bashTool) Locality() agentflow.ToolLocality          { return agentflow.ToolLocalOnly }
