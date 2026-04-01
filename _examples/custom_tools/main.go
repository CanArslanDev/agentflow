// Custom tools example: an agent with multiple tools including web search and file reading.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/canarslan/agentflow"
	"github.com/canarslan/agentflow/middleware"
	"github.com/canarslan/agentflow/provider/openrouter"
	"github.com/canarslan/agentflow/tools"
)

func main() {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "OPENROUTER_API_KEY environment variable required")
		os.Exit(1)
	}

	provider := openrouter.New(apiKey, "anthropic/claude-sonnet-4-20250514")

	// Tool 1: Read a file from disk.
	readFile := tools.New("read_file", "Read the contents of a file from the filesystem.").
		WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{
					"type":        "string",
					"description": "Absolute or relative path to the file",
				},
			},
			"required": []string{"path"},
		}).
		ReadOnly(true).
		ConcurrencySafe(true).
		WithExecute(func(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			var params struct {
				Path string `json:"path"`
			}
			if err := json.Unmarshal(input, &params); err != nil {
				return &agentflow.ToolResult{Content: err.Error(), IsError: true}, nil
			}

			data, err := os.ReadFile(params.Path)
			if err != nil {
				return &agentflow.ToolResult{Content: err.Error(), IsError: true}, nil
			}

			return &agentflow.ToolResult{Content: string(data)}, nil
		}).
		Build()

	// Tool 2: List files in a directory.
	listDir := tools.New("list_directory", "List files and directories in the given path.").
		WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{
					"type":        "string",
					"description": "Directory path to list",
				},
			},
			"required": []string{"path"},
		}).
		ReadOnly(true).
		ConcurrencySafe(true).
		WithExecute(func(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			var params struct {
				Path string `json:"path"`
			}
			if err := json.Unmarshal(input, &params); err != nil {
				return &agentflow.ToolResult{Content: err.Error(), IsError: true}, nil
			}

			entries, err := os.ReadDir(params.Path)
			if err != nil {
				return &agentflow.ToolResult{Content: err.Error(), IsError: true}, nil
			}

			var lines []string
			for _, entry := range entries {
				prefix := "📄"
				if entry.IsDir() {
					prefix = "📁"
				}
				lines = append(lines, fmt.Sprintf("%s %s", prefix, entry.Name()))
			}

			return &agentflow.ToolResult{Content: strings.Join(lines, "\n")}, nil
		}).
		Build()

	// Tool 3: Write to a file (non-read-only, non-concurrent-safe).
	writeFile := tools.New("write_file", "Write content to a file. Creates the file if it doesn't exist.").
		WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{
					"type":        "string",
					"description": "Path to the file to write",
				},
				"content": map[string]any{
					"type":        "string",
					"description": "Content to write to the file",
				},
			},
			"required": []string{"path", "content"},
		}).
		ReadOnly(false).
		ConcurrencySafe(false).
		WithExecute(func(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			var params struct {
				Path    string `json:"path"`
				Content string `json:"content"`
			}
			if err := json.Unmarshal(input, &params); err != nil {
				return &agentflow.ToolResult{Content: err.Error(), IsError: true}, nil
			}

			if err := os.WriteFile(params.Path, []byte(params.Content), 0644); err != nil {
				return &agentflow.ToolResult{Content: err.Error(), IsError: true}, nil
			}

			return &agentflow.ToolResult{
				Content: fmt.Sprintf("Successfully wrote %d bytes to %s", len(params.Content), params.Path),
			}, nil
		}).
		Build()

	// Metrics collection.
	metrics := middleware.NewMetrics()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(readFile, listDir, writeFile),
		agentflow.WithSystemPrompt("You are a file system assistant. Help users explore and manage files."),
		agentflow.WithMaxTurns(10),
		agentflow.WithMaxConcurrency(3),
		// Register metrics hooks.
		agentflow.WithHook(metrics.Hooks()[0]),
		agentflow.WithHook(metrics.Hooks()[1]),
		agentflow.WithHook(metrics.Hooks()[2]),
		// Only allow read operations.
		agentflow.WithPermission(agentflow.ReadOnlyPermission()),
	)

	messages := []agentflow.Message{
		agentflow.NewUserMessage("List the files in the current directory, then read the go.mod file if it exists."),
	}

	for ev := range agent.Run(context.Background(), messages) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			fmt.Print(ev.TextDelta.Text)
		case agentflow.EventToolStart:
			fmt.Printf("\n⚡ [%s]\n", ev.ToolStart.ToolCall.Name)
		case agentflow.EventToolEnd:
			status := "✓"
			if ev.ToolEnd.Result.IsError {
				status = "✗"
			}
			fmt.Printf("%s [%s] %v\n\n", status, ev.ToolEnd.ToolCall.Name, ev.ToolEnd.Duration)
		case agentflow.EventTurnEnd:
			fmt.Printf("\n--- %s (turn %d) ---\n", ev.TurnEnd.Reason, ev.TurnEnd.TurnNumber)
		}
	}

	// Print metrics.
	snap := metrics.Snapshot()
	fmt.Printf("\nMetrics: %d calls, %d errors, %d turns\n", snap.TotalCalls, snap.TotalErrors, snap.TotalTurns)
	for name, tm := range snap.ByTool {
		fmt.Printf("  %s: %d calls, avg %v\n", name, tm.Calls, tm.AvgLatency)
	}
}
