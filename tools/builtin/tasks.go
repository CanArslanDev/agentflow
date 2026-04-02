package builtin

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/CanArslanDev/agentflow"
)

// TaskTools returns the set of task management tools that share the given store.
// Register all of them together so the agent can create, update, list, and get tasks.
//
//	store := agentflow.NewTaskStore()
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithTools(builtin.TaskTools(store)...),
//	)
func TaskTools(store *agentflow.TaskStore) []agentflow.Tool {
	return []agentflow.Tool{
		&taskCreateTool{store: store},
		&taskUpdateTool{store: store},
		&taskListTool{store: store},
		&taskGetTool{store: store},
	}
}

// --- task_create ---

type taskCreateTool struct{ store *agentflow.TaskStore }

func (t *taskCreateTool) Name() string { return "task_create" }
func (t *taskCreateTool) Description() string {
	return "Create a new task to track work. Use this to break down complex goals into manageable steps. Returns the created task with its assigned ID."
}
func (t *taskCreateTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"title":       map[string]any{"type": "string", "description": "Short title for the task"},
			"description": map[string]any{"type": "string", "description": "Optional detailed description"},
		},
		"required": []string{"title"},
	}
}
func (t *taskCreateTool) Execute(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var p struct {
		Title       string `json:"title"`
		Description string `json:"description"`
	}
	if err := json.Unmarshal(input, &p); err != nil {
		return &agentflow.ToolResult{Content: "invalid input: " + err.Error(), IsError: true}, nil
	}
	if strings.TrimSpace(p.Title) == "" {
		return &agentflow.ToolResult{Content: "title is required", IsError: true}, nil
	}

	task := t.store.Create(p.Title, p.Description)
	data, _ := json.Marshal(task)
	return &agentflow.ToolResult{Content: string(data)}, nil
}
func (t *taskCreateTool) IsConcurrencySafe(_ json.RawMessage) bool { return false }
func (t *taskCreateTool) IsReadOnly(_ json.RawMessage) bool        { return false }
func (t *taskCreateTool) Locality() agentflow.ToolLocality          { return agentflow.ToolAny }

// --- task_update ---

type taskUpdateTool struct{ store *agentflow.TaskStore }

func (t *taskUpdateTool) Name() string { return "task_update" }
func (t *taskUpdateTool) Description() string {
	return "Update a task's status. Use 'in_progress' when starting work, 'completed' when done, 'failed' if it cannot be finished."
}
func (t *taskUpdateTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"id":          map[string]any{"type": "integer", "description": "Task ID to update"},
			"status":      map[string]any{"type": "string", "enum": []string{"pending", "in_progress", "completed", "failed"}, "description": "New status"},
			"description": map[string]any{"type": "string", "description": "Optional updated description"},
		},
		"required": []string{"id", "status"},
	}
}
func (t *taskUpdateTool) Execute(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var p struct {
		ID          int    `json:"id"`
		Status      string `json:"status"`
		Description string `json:"description"`
	}
	if err := json.Unmarshal(input, &p); err != nil {
		return &agentflow.ToolResult{Content: "invalid input: " + err.Error(), IsError: true}, nil
	}

	status := agentflow.TaskStatus(p.Status)
	switch status {
	case agentflow.TaskPending, agentflow.TaskInProgress, agentflow.TaskCompleted, agentflow.TaskFailed:
	default:
		return &agentflow.ToolResult{Content: "invalid status: " + p.Status, IsError: true}, nil
	}

	if err := t.store.Update(p.ID, status, p.Description); err != nil {
		return &agentflow.ToolResult{Content: err.Error(), IsError: true}, nil
	}

	task := t.store.Get(p.ID)
	data, _ := json.Marshal(task)
	return &agentflow.ToolResult{Content: string(data)}, nil
}
func (t *taskUpdateTool) IsConcurrencySafe(_ json.RawMessage) bool { return false }
func (t *taskUpdateTool) IsReadOnly(_ json.RawMessage) bool        { return false }
func (t *taskUpdateTool) Locality() agentflow.ToolLocality          { return agentflow.ToolAny }

// --- task_list ---

type taskListTool struct{ store *agentflow.TaskStore }

func (t *taskListTool) Name() string { return "task_list" }
func (t *taskListTool) Description() string {
	return "List all tasks with their current status. Use to review progress and decide what to work on next."
}
func (t *taskListTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"status": map[string]any{"type": "string", "description": "Optional: filter by status (pending, in_progress, completed, failed)"},
		},
	}
}
func (t *taskListTool) Execute(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var p struct {
		Status string `json:"status"`
	}
	json.Unmarshal(input, &p)

	tasks := t.store.List()

	if p.Status != "" {
		filtered := make([]*agentflow.Task, 0)
		for _, task := range tasks {
			if string(task.Status) == p.Status {
				filtered = append(filtered, task)
			}
		}
		tasks = filtered
	}

	if len(tasks) == 0 {
		return &agentflow.ToolResult{Content: "No tasks found."}, nil
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Tasks (%d total):\n\n", len(tasks)))
	for _, task := range tasks {
		marker := "[ ]"
		switch task.Status {
		case agentflow.TaskInProgress:
			marker = "[~]"
		case agentflow.TaskCompleted:
			marker = "[x]"
		case agentflow.TaskFailed:
			marker = "[!]"
		}
		sb.WriteString(fmt.Sprintf("#%d %s %s", task.ID, marker, task.Title))
		if task.Description != "" {
			sb.WriteString(" — " + task.Description)
		}
		sb.WriteString("\n")
	}

	return &agentflow.ToolResult{Content: sb.String()}, nil
}
func (t *taskListTool) IsConcurrencySafe(_ json.RawMessage) bool { return true }
func (t *taskListTool) IsReadOnly(_ json.RawMessage) bool        { return true }
func (t *taskListTool) Locality() agentflow.ToolLocality          { return agentflow.ToolAny }

// --- task_get ---

type taskGetTool struct{ store *agentflow.TaskStore }

func (t *taskGetTool) Name() string { return "task_get" }
func (t *taskGetTool) Description() string {
	return "Get details of a specific task by its ID."
}
func (t *taskGetTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"id": map[string]any{"type": "integer", "description": "Task ID"},
		},
		"required": []string{"id"},
	}
}
func (t *taskGetTool) Execute(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var p struct {
		ID int `json:"id"`
	}
	if err := json.Unmarshal(input, &p); err != nil {
		return &agentflow.ToolResult{Content: "invalid input: " + err.Error(), IsError: true}, nil
	}

	task := t.store.Get(p.ID)
	if task == nil {
		return &agentflow.ToolResult{Content: fmt.Sprintf("task #%d not found", p.ID), IsError: true}, nil
	}

	data, _ := json.Marshal(task)
	return &agentflow.ToolResult{Content: string(data)}, nil
}
func (t *taskGetTool) IsConcurrencySafe(_ json.RawMessage) bool { return true }
func (t *taskGetTool) IsReadOnly(_ json.RawMessage) bool        { return true }
func (t *taskGetTool) Locality() agentflow.ToolLocality          { return agentflow.ToolAny }
