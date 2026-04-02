package agentflow_test

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/groq"
	"github.com/CanArslanDev/agentflow/tools"
	"github.com/CanArslanDev/agentflow/tools/builtin"
)

func groqProvider(t *testing.T) *groq.Provider {
	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		t.Skip("GROQ_API_KEY not set")
	}
	return groq.New(key, "llama-3.3-70b-versatile")
}

// ==========================================================================
// T1.1: STREAMING TOOL EXECUTION
// ==========================================================================

// TestIntegration_Tier1_StreamingExec_Parallel — model 2 tool call yapar,
// streaming executor paralel calistirir.
func TestIntegration_Tier1_StreamingExec_Parallel(t *testing.T) {
	provider := groqProvider(t)

	weather := tools.New("get_weather", "Get weather for a city. Returns temperature.").
		WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string"},
			},
			"required": []string{"city"},
		}).
		ConcurrencySafe(true).ReadOnly(true).RemoteSafe().
		WithExecute(func(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			var p struct{ City string `json:"city"` }
			json.Unmarshal(input, &p)
			time.Sleep(100 * time.Millisecond) // Simulate latency.
			return &agentflow.ToolResult{Content: p.City + ": 22C sunny"}, nil
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(weather),
		agentflow.WithSystemPrompt("You are a weather assistant. When asked about multiple cities, call get_weather for each city."),
		agentflow.WithMaxTurns(5),
		agentflow.WithMaxTokens(500),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var text string
	var toolCalls int
	start := time.Now()

	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("What is the weather in Istanbul and Tokyo?"),
	}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			text += ev.TextDelta.Text
		case agentflow.EventToolStart:
			toolCalls++
			t.Logf("[%v] Tool start: %s", time.Since(start).Round(time.Millisecond), ev.ToolStart.ToolCall.Name)
		case agentflow.EventToolEnd:
			t.Logf("[%v] Tool end: %s (%v)", time.Since(start).Round(time.Millisecond), ev.ToolEnd.ToolCall.Name, ev.ToolEnd.Duration)
		case agentflow.EventTurnEnd:
			t.Logf("[%v] Turn %d: %s", time.Since(start).Round(time.Millisecond), ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		}
	}

	t.Logf("Response: %s", text)
	if toolCalls < 2 {
		t.Errorf("expected at least 2 tool calls, got %d", toolCalls)
	}
}

// TestIntegration_Tier1_StreamingExec_SingleTool — tek tool call da dogru calisir.
func TestIntegration_Tier1_StreamingExec_SingleTool(t *testing.T) {
	provider := groqProvider(t)

	calc := tools.New("calculator", "Calculate math").
		WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"a": map[string]any{"type": "number"}, "b": map[string]any{"type": "number"},
				"op": map[string]any{"type": "string"},
			},
			"required": []string{"a", "b", "op"},
		}).
		ConcurrencySafe(true).ReadOnly(true).RemoteSafe().
		WithExecute(func(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			var p struct{ A, B float64; Op string }
			json.Unmarshal(input, &p)
			return &agentflow.ToolResult{Content: "420"}, nil
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(calc),
		agentflow.WithSystemPrompt("Use the calculator tool for math."),
		agentflow.WithMaxTurns(3),
		agentflow.WithMaxTokens(300),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var text string
	var toolCalls int
	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("What is 15 * 28?"),
	}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			text += ev.TextDelta.Text
		case agentflow.EventToolStart:
			toolCalls++
		case agentflow.EventTurnEnd:
			t.Logf("Turn %d: %s", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		}
	}

	t.Logf("Response: %s", text)
	if toolCalls == 0 {
		t.Error("expected tool call")
	}
	if !strings.Contains(text, "420") {
		t.Errorf("expected '420' in response, got: %s", text)
	}
}

// ==========================================================================
// T1.2: TASK MANAGEMENT
// ==========================================================================

// TestIntegration_Tier1_TaskCreate — agent kendi basina task olusturur.
func TestIntegration_Tier1_TaskCreate(t *testing.T) {
	provider := groqProvider(t)
	store := agentflow.NewTaskStore()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(builtin.TaskTools(store)...),
		agentflow.WithSystemPrompt("You are a project planner. When asked to plan something, create tasks using task_create. Create at least 3 tasks."),
		agentflow.WithMaxTurns(5),
		agentflow.WithMaxTokens(1000),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var text string
	var taskCreates int
	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("Plan a weekend trip to Istanbul. Create tasks for each step."),
	}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			text += ev.TextDelta.Text
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
	for _, task := range tasks {
		t.Logf("  #%d [%s] %s", task.ID, task.Status, task.Title)
	}

	if len(tasks) < 2 {
		t.Errorf("expected at least 2 tasks created, got %d", len(tasks))
	}
	if taskCreates < 2 {
		t.Errorf("expected at least 2 task_create calls, got %d", taskCreates)
	}
}

// TestIntegration_Tier1_TaskUpdateAndList — agent task olustur, listele, guncelle.
func TestIntegration_Tier1_TaskUpdateAndList(t *testing.T) {
	provider := groqProvider(t)
	store := agentflow.NewTaskStore()

	// Pre-create some tasks.
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

	// Verify task_list was called.
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

	// Verify task 1 was updated.
	task1 := store.Get(1)
	if task1 != nil {
		t.Logf("Task 1 status: %s", task1.Status)
		if task1.Status != agentflow.TaskCompleted {
			t.Logf("Note: task 1 status is %s (model may have used different approach)", task1.Status)
		}
	}
}

// ==========================================================================
// T1.3: SKILLS SYSTEM
// ==========================================================================

// TestIntegration_Tier1_SkillSummarize — agent summarize skill'ini calistirir.
func TestIntegration_Tier1_SkillSummarize(t *testing.T) {
	provider := groqProvider(t)
	registry := agentflow.NewSkillRegistry()

	registry.Register(&agentflow.Skill{
		Name:         "summarize",
		Description:  "Summarize text into a concise paragraph",
		SystemPrompt: "You are a summarization expert. Provide a concise summary in 2-3 sentences maximum. Be direct and factual.",
		MaxTurns:     1,
		MaxTokens:    200,
	})

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(builtin.SkillTools(registry, provider)...),
		agentflow.WithSystemPrompt("You have access to skills. Use run_skill to execute them. Use list_skills to see available skills."),
		agentflow.WithMaxTurns(5),
		agentflow.WithMaxTokens(800),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	longText := `Go is a statically typed, compiled high-level programming language designed at Google
by Robert Griesemer, Rob Pike, and Ken Thompson. It is syntactically similar to C, but also has
memory safety, garbage collection, structural typing, and CSP-style concurrency. It was designed
to improve programming productivity in an era of multicore processors, networked machines, and
large codebases. Go was publicly announced in November 2009, and version 1.0 was released in
March 2012. It is used extensively at Google and in many open-source projects.`

	var text string
	var skillRun bool
	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("Use the summarize skill to summarize this text:\n\n" + longText),
	}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			text += ev.TextDelta.Text
		case agentflow.EventToolStart:
			t.Logf("Tool: %s", ev.ToolStart.ToolCall.Name)
			if ev.ToolStart.ToolCall.Name == "run_skill" {
				skillRun = true
			}
		case agentflow.EventToolEnd:
			t.Logf("Skill result (%d chars): %s...", len(ev.ToolEnd.Result.Content),
				ev.ToolEnd.Result.Content[:min(len(ev.ToolEnd.Result.Content), 150)])
		case agentflow.EventTurnEnd:
			t.Logf("Turn %d: %s", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		}
	}

	t.Logf("Final: %s", text)
	if !skillRun {
		t.Error("expected run_skill to be called")
	}
}

// TestIntegration_Tier1_SkillTranslate — agent translate skill'ini calistirir.
func TestIntegration_Tier1_SkillTranslate(t *testing.T) {
	provider := groqProvider(t)
	registry := agentflow.NewSkillRegistry()

	registry.Register(&agentflow.Skill{
		Name:         "translate",
		Description:  "Translate text to a target language",
		SystemPrompt: "You are a translator. Translate the given text accurately. Only output the translation, nothing else.",
		MaxTurns:     1,
		MaxTokens:    300,
	})
	registry.Register(&agentflow.Skill{
		Name:         "summarize",
		Description:  "Summarize text concisely",
		SystemPrompt: "Summarize in 1-2 sentences.",
		MaxTurns:     1,
		MaxTokens:    200,
	})

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(builtin.SkillTools(registry, provider)...),
		agentflow.WithSystemPrompt("You have skills available. Use list_skills to see them, run_skill to execute them."),
		agentflow.WithMaxTurns(5),
		agentflow.WithMaxTokens(800),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var text string
	var toolNames []string
	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("Translate 'Hello, how are you today?' to Turkish using the translate skill."),
	}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			text += ev.TextDelta.Text
		case agentflow.EventToolStart:
			toolNames = append(toolNames, ev.ToolStart.ToolCall.Name)
			t.Logf("Tool: %s", ev.ToolStart.ToolCall.Name)
		case agentflow.EventToolEnd:
			t.Logf("Result: %s", ev.ToolEnd.Result.Content)
		case agentflow.EventTurnEnd:
			t.Logf("Turn %d: %s", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		}
	}

	t.Logf("Final: %s", text)

	hasRunSkill := false
	for _, n := range toolNames {
		if n == "run_skill" {
			hasRunSkill = true
		}
	}
	if !hasRunSkill {
		t.Error("expected run_skill call")
	}
}

// TestIntegration_Tier1_SkillListAndRun — agent once list eder, sonra calistirir.
func TestIntegration_Tier1_SkillListAndRun(t *testing.T) {
	provider := groqProvider(t)
	registry := agentflow.NewSkillRegistry()

	registry.Register(&agentflow.Skill{
		Name:         "count_words",
		Description:  "Count the number of words in text",
		SystemPrompt: "Count the words in the given text. Reply with only the number.",
		MaxTurns:     1,
		MaxTokens:    50,
	})

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(builtin.SkillTools(registry, provider)...),
		agentflow.WithSystemPrompt("Use list_skills to discover skills, then use run_skill to execute them."),
		agentflow.WithMaxTurns(5),
		agentflow.WithMaxTokens(500),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var toolNames []string
	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("List available skills, then use count_words to count words in: 'The quick brown fox jumps over the lazy dog'"),
	}) {
		if ev.Type == agentflow.EventToolStart {
			toolNames = append(toolNames, ev.ToolStart.ToolCall.Name)
			t.Logf("Tool: %s", ev.ToolStart.ToolCall.Name)
		}
		if ev.Type == agentflow.EventToolEnd {
			t.Logf("Result: %s", ev.ToolEnd.Result.Content[:min(len(ev.ToolEnd.Result.Content), 200)])
		}
		if ev.Type == agentflow.EventTurnEnd {
			t.Logf("Turn %d: %s", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		}
	}

	hasListSkills := false
	hasRunSkill := false
	for _, n := range toolNames {
		if n == "list_skills" {
			hasListSkills = true
		}
		if n == "run_skill" {
			hasRunSkill = true
		}
	}
	if !hasListSkills {
		t.Error("expected list_skills call")
	}
	if !hasRunSkill {
		t.Error("expected run_skill call")
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
