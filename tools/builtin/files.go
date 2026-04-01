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

// --- ReadFile ---

// ReadFile returns a tool that reads file contents from the filesystem.
func ReadFile() agentflow.Tool { return &readFileTool{} }

type readFileTool struct{}

func (t *readFileTool) Name() string { return "read_file" }
func (t *readFileTool) Description() string {
	return "Read the contents of a file from the filesystem. Returns the file text content."
}
func (t *readFileTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{
				"type":        "string",
				"description": "Absolute or relative path to the file",
			},
			"offset": map[string]any{
				"type":        "integer",
				"description": "Line number to start reading from (0-based, optional)",
			},
			"limit": map[string]any{
				"type":        "integer",
				"description": "Maximum number of lines to read (optional, default: all)",
			},
		},
		"required": []string{"path"},
	}
}

func (t *readFileTool) Execute(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var params struct {
		Path   string `json:"path"`
		Offset int    `json:"offset"`
		Limit  int    `json:"limit"`
	}
	if err := json.Unmarshal(input, &params); err != nil {
		return &agentflow.ToolResult{Content: "invalid input: " + err.Error(), IsError: true}, nil
	}

	data, err := os.ReadFile(params.Path)
	if err != nil {
		return &agentflow.ToolResult{Content: err.Error(), IsError: true}, nil
	}

	content := string(data)

	// Apply offset and limit if specified.
	if params.Offset > 0 || params.Limit > 0 {
		lines := strings.Split(content, "\n")
		start := params.Offset
		if start > len(lines) {
			start = len(lines)
		}
		end := len(lines)
		if params.Limit > 0 && start+params.Limit < end {
			end = start + params.Limit
		}
		content = strings.Join(lines[start:end], "\n")
	}

	return &agentflow.ToolResult{
		Content:  content,
		Metadata: map[string]any{"path": params.Path, "size": len(data)},
	}, nil
}

func (t *readFileTool) IsConcurrencySafe(_ json.RawMessage) bool { return true }
func (t *readFileTool) IsReadOnly(_ json.RawMessage) bool        { return true }
func (t *readFileTool) Locality() agentflow.ToolLocality          { return agentflow.ToolLocalOnly }

// --- WriteFile ---

// WriteFile returns a tool that creates or overwrites files.
func WriteFile() agentflow.Tool { return &writeFileTool{} }

type writeFileTool struct{}

func (t *writeFileTool) Name() string { return "write_file" }
func (t *writeFileTool) Description() string {
	return "Write content to a file. Creates the file and parent directories if they don't exist. Overwrites existing content."
}
func (t *writeFileTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path":    map[string]any{"type": "string", "description": "Path to the file to write"},
			"content": map[string]any{"type": "string", "description": "Content to write to the file"},
		},
		"required": []string{"path", "content"},
	}
}

func (t *writeFileTool) Execute(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var params struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal(input, &params); err != nil {
		return &agentflow.ToolResult{Content: "invalid input: " + err.Error(), IsError: true}, nil
	}

	dir := filepath.Dir(params.Path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return &agentflow.ToolResult{Content: err.Error(), IsError: true}, nil
	}

	if err := os.WriteFile(params.Path, []byte(params.Content), 0644); err != nil {
		return &agentflow.ToolResult{Content: err.Error(), IsError: true}, nil
	}

	return &agentflow.ToolResult{
		Content: fmt.Sprintf("Wrote %d bytes to %s", len(params.Content), params.Path),
	}, nil
}

func (t *writeFileTool) IsConcurrencySafe(_ json.RawMessage) bool { return false }
func (t *writeFileTool) IsReadOnly(_ json.RawMessage) bool        { return false }
func (t *writeFileTool) Locality() agentflow.ToolLocality          { return agentflow.ToolLocalOnly }

// --- EditFile ---

// EditFile returns a tool that performs string replacement in files.
func EditFile() agentflow.Tool { return &editFileTool{} }

type editFileTool struct{}

func (t *editFileTool) Name() string { return "edit_file" }
func (t *editFileTool) Description() string {
	return "Edit a file by replacing an exact string match with new content. The old_string must appear exactly once in the file."
}
func (t *editFileTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path":       map[string]any{"type": "string", "description": "Path to the file to edit"},
			"old_string": map[string]any{"type": "string", "description": "Exact string to find and replace (must be unique in the file)"},
			"new_string": map[string]any{"type": "string", "description": "Replacement string"},
		},
		"required": []string{"path", "old_string", "new_string"},
	}
}

func (t *editFileTool) Execute(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var params struct {
		Path      string `json:"path"`
		OldString string `json:"old_string"`
		NewString string `json:"new_string"`
	}
	if err := json.Unmarshal(input, &params); err != nil {
		return &agentflow.ToolResult{Content: "invalid input: " + err.Error(), IsError: true}, nil
	}

	data, err := os.ReadFile(params.Path)
	if err != nil {
		return &agentflow.ToolResult{Content: err.Error(), IsError: true}, nil
	}

	content := string(data)
	count := strings.Count(content, params.OldString)
	if count == 0 {
		return &agentflow.ToolResult{Content: "old_string not found in file", IsError: true}, nil
	}
	if count > 1 {
		return &agentflow.ToolResult{
			Content: fmt.Sprintf("old_string found %d times — must be unique. Provide more context.", count),
			IsError: true,
		}, nil
	}

	newContent := strings.Replace(content, params.OldString, params.NewString, 1)
	if err := os.WriteFile(params.Path, []byte(newContent), 0644); err != nil {
		return &agentflow.ToolResult{Content: err.Error(), IsError: true}, nil
	}

	return &agentflow.ToolResult{Content: "Edit applied successfully"}, nil
}

func (t *editFileTool) IsConcurrencySafe(_ json.RawMessage) bool { return false }
func (t *editFileTool) IsReadOnly(_ json.RawMessage) bool        { return false }
func (t *editFileTool) Locality() agentflow.ToolLocality          { return agentflow.ToolLocalOnly }

// --- ListDir ---

// ListDir returns a tool that lists files and directories.
func ListDir() agentflow.Tool { return &listDirTool{} }

type listDirTool struct{}

func (t *listDirTool) Name() string { return "list_dir" }
func (t *listDirTool) Description() string {
	return "List files and directories in the given path. Returns names with type indicators."
}
func (t *listDirTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{"type": "string", "description": "Directory path to list"},
		},
		"required": []string{"path"},
	}
}

func (t *listDirTool) Execute(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var params struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(input, &params); err != nil {
		return &agentflow.ToolResult{Content: "invalid input: " + err.Error(), IsError: true}, nil
	}

	entries, err := os.ReadDir(params.Path)
	if err != nil {
		return &agentflow.ToolResult{Content: err.Error(), IsError: true}, nil
	}

	var lines []string
	for _, entry := range entries {
		info, _ := entry.Info()
		size := ""
		if info != nil && !entry.IsDir() {
			size = fmt.Sprintf(" (%d bytes)", info.Size())
		}
		prefix := "[file]"
		if entry.IsDir() {
			prefix = "[dir] "
		}
		lines = append(lines, prefix+" "+entry.Name()+size)
	}

	if len(lines) == 0 {
		return &agentflow.ToolResult{Content: "(empty directory)"}, nil
	}

	return &agentflow.ToolResult{Content: strings.Join(lines, "\n")}, nil
}

func (t *listDirTool) IsConcurrencySafe(_ json.RawMessage) bool { return true }
func (t *listDirTool) IsReadOnly(_ json.RawMessage) bool        { return true }
func (t *listDirTool) Locality() agentflow.ToolLocality          { return agentflow.ToolLocalOnly }
