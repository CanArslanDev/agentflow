package builtin

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/CanArslanDev/agentflow"
)

// --- Glob ---

// Glob returns a tool that finds files matching a glob pattern.
func Glob() agentflow.Tool { return &globTool{} }

type globTool struct{}

func (t *globTool) Name() string { return "glob" }
func (t *globTool) Description() string {
	return "Find files matching a glob pattern. Supports ** for recursive matching. Returns matching file paths."
}
func (t *globTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"pattern": map[string]any{
				"type":        "string",
				"description": "Glob pattern (e.g., '**/*.go', 'src/**/*.ts', '*.json')",
			},
			"path": map[string]any{
				"type":        "string",
				"description": "Base directory to search in (default: current directory)",
			},
		},
		"required": []string{"pattern"},
	}
}

func (t *globTool) Execute(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var params struct {
		Pattern string `json:"pattern"`
		Path    string `json:"path"`
	}
	if err := json.Unmarshal(input, &params); err != nil {
		return &agentflow.ToolResult{Content: "invalid input: " + err.Error(), IsError: true}, nil
	}

	base := params.Path
	if base == "" {
		base = "."
	}

	var matches []string

	// Handle ** recursive patterns by walking the directory tree.
	if strings.Contains(params.Pattern, "**") {
		parts := strings.SplitN(params.Pattern, "**", 2)
		prefix := parts[0]
		suffix := ""
		if len(parts) > 1 {
			suffix = strings.TrimPrefix(parts[1], "/")
		}

		root := filepath.Join(base, prefix)
		filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return nil
			}
			if info.IsDir() {
				return nil
			}
			if suffix == "" {
				matches = append(matches, path)
				return nil
			}
			matched, _ := filepath.Match(suffix, filepath.Base(path))
			if matched {
				matches = append(matches, path)
			}
			return nil
		})
	} else {
		pattern := filepath.Join(base, params.Pattern)
		var err error
		matches, err = filepath.Glob(pattern)
		if err != nil {
			return &agentflow.ToolResult{Content: err.Error(), IsError: true}, nil
		}
	}

	if len(matches) == 0 {
		return &agentflow.ToolResult{Content: "no files matched"}, nil
	}

	return &agentflow.ToolResult{
		Content:  strings.Join(matches, "\n"),
		Metadata: map[string]any{"count": len(matches)},
	}, nil
}

func (t *globTool) IsConcurrencySafe(_ json.RawMessage) bool { return true }
func (t *globTool) IsReadOnly(_ json.RawMessage) bool        { return true }
func (t *globTool) Locality() agentflow.ToolLocality          { return agentflow.ToolLocalOnly }

// --- Grep ---

// Grep returns a tool that searches file contents for a pattern.
func Grep() agentflow.Tool { return &grepTool{} }

type grepTool struct{}

func (t *grepTool) Name() string { return "grep" }
func (t *grepTool) Description() string {
	return "Search file contents for a text pattern. Returns matching lines with file paths and line numbers."
}
func (t *grepTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"pattern": map[string]any{
				"type":        "string",
				"description": "Text pattern to search for (substring match)",
			},
			"path": map[string]any{
				"type":        "string",
				"description": "File or directory to search in (default: current directory)",
			},
			"glob": map[string]any{
				"type":        "string",
				"description": "File pattern filter (e.g., '*.go', '*.ts')",
			},
		},
		"required": []string{"pattern"},
	}
}

func (t *grepTool) Execute(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var params struct {
		Pattern string `json:"pattern"`
		Path    string `json:"path"`
		Glob    string `json:"glob"`
	}
	if err := json.Unmarshal(input, &params); err != nil {
		return &agentflow.ToolResult{Content: "invalid input: " + err.Error(), IsError: true}, nil
	}

	base := params.Path
	if base == "" {
		base = "."
	}

	var results []string
	maxResults := 100

	filepath.Walk(base, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || len(results) >= maxResults {
			return nil
		}

		// Apply glob filter.
		if params.Glob != "" {
			matched, _ := filepath.Match(params.Glob, filepath.Base(path))
			if !matched {
				return nil
			}
		}

		data, err := os.ReadFile(path)
		if err != nil {
			return nil
		}

		lines := strings.Split(string(data), "\n")
		for i, line := range lines {
			if strings.Contains(line, params.Pattern) {
				results = append(results, fmt.Sprintf("%s:%d: %s", path, i+1, strings.TrimSpace(line)))
				if len(results) >= maxResults {
					return filepath.SkipAll
				}
			}
		}
		return nil
	})

	if len(results) == 0 {
		return &agentflow.ToolResult{Content: "no matches found"}, nil
	}

	content := strings.Join(results, "\n")
	if len(results) >= maxResults {
		content += fmt.Sprintf("\n... (limited to %d results)", maxResults)
	}

	return &agentflow.ToolResult{
		Content:  content,
		Metadata: map[string]any{"count": len(results)},
	}, nil
}

func (t *grepTool) IsConcurrencySafe(_ json.RawMessage) bool { return true }
func (t *grepTool) IsReadOnly(_ json.RawMessage) bool        { return true }
func (t *grepTool) Locality() agentflow.ToolLocality          { return agentflow.ToolLocalOnly }
