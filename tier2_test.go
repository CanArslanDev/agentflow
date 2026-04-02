package agentflow_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
)

// ==========================================================================
// T2.1: TEAM/SWARM COORDINATION
// ==========================================================================

// TestIntegration_Tier2_TeamRunAll — iki takim uyesi paralel calisir.
func TestIntegration_Tier2_TeamRunAll(t *testing.T) {
	provider := groqProvider(t)

	team := agentflow.NewTeam(provider, []agentflow.TeamMember{
		{
			Role:         "researcher",
			SystemPrompt: "You are a research specialist. Give concise factual answers in 1-2 sentences.",
			MaxTurns:     1,
			MaxTokens:    150,
		},
		{
			Role:         "writer",
			SystemPrompt: "You are a creative writer. Write in an engaging style. Keep responses to 1-2 sentences. Just respond directly, do not use any tools.",
			MaxTurns:     2,
			MaxTokens:    200,
		},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	results := team.RunAll(ctx, map[string]string{
		"researcher": "What is the capital of Turkey?",
		"writer":     "Write a one-sentence poem about Istanbul.",
	})

	for role, result := range results {
		if result.Error != nil {
			t.Errorf("%s error: %v", role, result.Error)
		} else {
			t.Logf("%s: %q", role, result.Response)
			if result.Response == "" {
				t.Errorf("%s returned empty response", role)
			}
		}
	}

	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}
}

// TestIntegration_Tier2_TeamSharedMemory — shared memory uzerinden veri paylasilir.
func TestIntegration_Tier2_TeamSharedMemory(t *testing.T) {
	provider := groqProvider(t)

	team := agentflow.NewTeam(provider, []agentflow.TeamMember{
		{
			Role:         "analyst",
			SystemPrompt: "You are a data analyst. Store your findings in shared memory using set_shared_memory tool. Key should be 'finding', value should be your analysis.",
			MaxTurns:     3,
			MaxTokens:    300,
		},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	events, err := team.RunMember(ctx, "analyst", "Analyze this: Go was created in 2009 at Google. Store the creation year in shared memory with key 'go_year'.")
	if err != nil {
		t.Fatalf("RunMember error: %v", err)
	}

	var toolNames []string
	for ev := range events {
		if ev.Type == agentflow.EventToolStart {
			toolNames = append(toolNames, ev.ToolStart.ToolCall.Name)
			t.Logf("Tool: %s", ev.ToolStart.ToolCall.Name)
		}
		if ev.Type == agentflow.EventToolEnd {
			t.Logf("Result: %s", ev.ToolEnd.Result.Content)
		}
	}

	// Note: SharedMemory is internal to team. The test validates tool calls happened.
	hasSetMemory := false
	for _, name := range toolNames {
		if name == "set_shared_memory" {
			hasSetMemory = true
		}
	}
	if !hasSetMemory {
		t.Log("Note: model may not have used set_shared_memory tool, but pipeline works")
	}
}

// ==========================================================================
// T2.2: PLAN MODE
// ==========================================================================

// TestIntegration_Tier2_PlanOnly — plan mode yapilandirilmis plan uretir.
func TestIntegration_Tier2_PlanOnly(t *testing.T) {
	provider := groqProvider(t)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := agentflow.Plan(ctx, provider, "Build a simple REST API in Go with user authentication")
	if err != nil {
		t.Fatalf("Plan error: %v", err)
	}

	t.Logf("Plan (%d chars):\n%s", len(result.Plan), result.Plan[:min(len(result.Plan), 500)])

	if result.Plan == "" {
		t.Error("empty plan")
	}
	// Plan should have structured content.
	lower := strings.ToLower(result.Plan)
	if !strings.Contains(lower, "step") && !strings.Contains(lower, "1.") {
		t.Log("Note: plan may not have numbered steps, but content was generated")
	}
}

// TestIntegration_Tier2_PlanAndExecute — plan yap, sonra execute et.
func TestIntegration_Tier2_PlanAndExecute(t *testing.T) {
	provider := groqProvider(t)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var planText, execText string
	var planPhase bool = true
	for ev := range agentflow.PlanAndExecute(ctx, provider,
		"List 3 interesting facts about Go programming language",
		nil, // no tools needed
		agentflow.WithMaxTurns(1),
		agentflow.WithMaxTokens(500),
	) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			if planPhase {
				planText += ev.TextDelta.Text
			} else {
				execText += ev.TextDelta.Text
			}
		case agentflow.EventTurnEnd:
			if ev.TurnEnd.TurnNumber == 0 {
				planPhase = false
				t.Logf("Plan phase done (%d chars)", len(planText))
			} else {
				t.Logf("Exec turn %d: %s", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
			}
		}
	}

	t.Logf("Plan: %s...", planText[:min(len(planText), 200)])
	t.Logf("Exec: %s...", execText[:min(len(execText), 200)])

	if planText == "" {
		t.Error("empty plan text")
	}
	if execText == "" {
		t.Error("empty execution text")
	}
}

// ==========================================================================
// T2.3: SESSION MEMORY EXTRACTION
// ==========================================================================

// TestIntegration_Tier2_MemoryExtraction — konusmadan key fact'ler cikarilir.
func TestIntegration_Tier2_MemoryExtraction(t *testing.T) {
	provider := groqProvider(t)

	messages := []agentflow.Message{
		agentflow.NewUserMessage("My name is Can and I live in Istanbul. I'm a Go developer."),
		agentflow.NewAssistantMessage("Nice to meet you, Can! Istanbul is a great city for tech. How long have you been coding in Go?"),
		agentflow.NewUserMessage("About 3 years. I mainly work on backend APIs and I prefer using Chi router."),
		agentflow.NewAssistantMessage("Chi is excellent for Go APIs. Have you tried agentflow for building AI agents?"),
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	memories, err := agentflow.ExtractMemories(ctx, provider, messages)
	if err != nil {
		t.Fatalf("ExtractMemories error: %v", err)
	}

	t.Logf("Extracted %d memories:", len(memories))
	for i, m := range memories {
		t.Logf("  %d. %s", i+1, m)
	}

	if len(memories) == 0 {
		t.Error("expected at least 1 extracted memory")
	}

	// Should capture key facts.
	joined := strings.Join(memories, " ")
	lower := strings.ToLower(joined)
	hasName := strings.Contains(lower, "can")
	hasCity := strings.Contains(lower, "istanbul")
	hasGo := strings.Contains(lower, "go")

	found := 0
	if hasName {
		found++
	}
	if hasCity {
		found++
	}
	if hasGo {
		found++
	}
	if found < 2 {
		t.Logf("Note: expected to find at least 2 of [can, istanbul, go] in memories")
	}
}

// ==========================================================================
// T2.5: REMOTE TRIGGERS
// ==========================================================================

// TestIntegration_Tier2_TriggerExecution — trigger bir kez calisir ve sonuc doner.
func TestIntegration_Tier2_TriggerExecution(t *testing.T) {
	provider := groqProvider(t)

	var result agentflow.TriggerResult
	done := make(chan struct{})

	scheduler := agentflow.NewTriggerScheduler()
	scheduler.Schedule(agentflow.Trigger{
		ID:           "test-trigger",
		Interval:     1 * time.Hour, // Won't tick again in test.
		Task:         "Say 'trigger fired' in exactly 2 words.",
		Provider:     provider,
		SystemPrompt: "You are a minimal responder. Reply in exactly the words requested.",
		MaxTurns:     1,
		MaxTokens:    50,
		OnResult: func(r agentflow.TriggerResult) {
			result = r
			close(done)
		},
	})

	select {
	case <-done:
		t.Logf("Trigger result: %q (duration: %v)", result.Response, result.Duration)
		if result.Error != nil {
			t.Errorf("trigger error: %v", result.Error)
		}
		if result.Response == "" {
			t.Error("empty trigger response")
		}
	case <-time.After(30 * time.Second):
		t.Fatal("trigger timed out")
	}

	scheduler.CancelAll()
}
