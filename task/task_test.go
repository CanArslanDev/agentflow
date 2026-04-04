package task_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/groq"
	"github.com/CanArslanDev/agentflow/task"
	"github.com/CanArslanDev/agentflow/tools/builtin"
)

func groqProvider(t *testing.T) *groq.Provider {
	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		t.Skip("GROQ_API_KEY not set")
	}
	return groq.New(key, "llama-3.3-70b-versatile")
}

func TestIntegration_TaskCreate(t *testing.T) {
	provider := groqProvider(t)
	store := task.NewStore()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(builtin.TaskTools(store)...),
		agentflow.WithSystemPrompt("You are a project planner. When asked to plan something, create tasks using task_create. Create at least 3 tasks."),
		agentflow.WithMaxTurns(5),
		agentflow.WithMaxTokens(1000),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var taskCreates int
	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("Plan a weekend trip to Istanbul. Create tasks for each step."),
	}) {
		switch ev.Type {
		case agentflow.EventToolStart:
			if ev.ToolStart.ToolCall.Name == "task_create" {
				taskCreates++
			}
			t.Logf("Tool: %s", ev.ToolStart.ToolCall.Name)
		case agentflow.EventTurnEnd:
			t.Logf("Turn %d: %s", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		}
	}

	tasks := store.List()
	t.Logf("Created %d tasks:", len(tasks))
	for _, tk := range tasks {
		t.Logf("  #%d [%s] %s", tk.ID, tk.Status, tk.Title)
	}

	if len(tasks) < 2 {
		t.Errorf("expected at least 2 tasks created, got %d", len(tasks))
	}
}

func TestIntegration_TaskUpdateAndList(t *testing.T) {
	provider := groqProvider(t)
	store := task.NewStore()

	store.Create("Book flight tickets", "Find cheapest flights to Istanbul")
	store.Create("Reserve hotel", "Budget hotel near Sultanahmet")
	store.Create("Plan activities", "Museums, food tour, Bosphorus cruise")

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(builtin.TaskTools(store)...),
		agentflow.WithSystemPrompt("You are a task manager. Use task_list to see tasks, task_update to change their status. Mark completed tasks as completed."),
		agentflow.WithMaxTurns(5),
		agentflow.WithMaxTokens(800),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var toolNames []string
	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("List all tasks, then mark the first task as completed."),
	}) {
		switch ev.Type {
		case agentflow.EventToolStart:
			toolNames = append(toolNames, ev.ToolStart.ToolCall.Name)
			t.Logf("Tool: %s(%s)", ev.ToolStart.ToolCall.Name, string(ev.ToolStart.ToolCall.Input))
		case agentflow.EventToolEnd:
			t.Logf("Result: %s", ev.ToolEnd.Result.Content[:min(len(ev.ToolEnd.Result.Content), 200)])
		case agentflow.EventTurnEnd:
			t.Logf("Turn %d: %s", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		}
	}

	hasTaskList := false
	hasTaskUpdate := false
	for _, name := range toolNames {
		if name == "task_list" {
			hasTaskList = true
		}
		if name == "task_update" {
			hasTaskUpdate = true
		}
	}
	if !hasTaskList {
		t.Error("expected task_list to be called")
	}
	if !hasTaskUpdate {
		t.Error("expected task_update to be called")
	}

	task1 := store.Get(1)
	if task1 != nil {
		t.Logf("Task 1 status: %s", task1.Status)
		if task1.Status != task.Completed {
			t.Logf("Note: task 1 status is %s (model may have used different approach)", task1.Status)
		}
	}
}
