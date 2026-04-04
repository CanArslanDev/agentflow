package plan_test

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/plan"
	"github.com/CanArslanDev/agentflow/provider/groq"
)

func groqProvider(t *testing.T) *groq.Provider {
	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		t.Skip("GROQ_API_KEY not set")
	}
	return groq.New(key, "llama-3.3-70b-versatile")
}

func TestIntegration_PlanOnly(t *testing.T) {
	provider := groqProvider(t)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := plan.Plan(ctx, provider, "Build a simple REST API in Go with user authentication")
	if err != nil {
		t.Fatalf("Plan error: %v", err)
	}

	t.Logf("Plan (%d chars):\n%s", len(result.Plan), result.Plan[:min(len(result.Plan), 500)])

	if result.Plan == "" {
		t.Error("empty plan")
	}
	lower := strings.ToLower(result.Plan)
	if !strings.Contains(lower, "step") && !strings.Contains(lower, "1.") {
		t.Log("Note: plan may not have numbered steps, but content was generated")
	}
}

func TestIntegration_PlanAndExecute(t *testing.T) {
	provider := groqProvider(t)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var planText, execText string
	var planPhase bool = true
	for ev := range plan.PlanAndExecute(ctx, provider,
		"List 3 interesting facts about Go programming language",
		nil,
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

func TestIntegration_MemoryExtraction(t *testing.T) {
	provider := groqProvider(t)

	messages := []agentflow.Message{
		agentflow.NewUserMessage("My name is Can and I live in Istanbul. I'm a Go developer."),
		agentflow.NewAssistantMessage("Nice to meet you, Can! Istanbul is a great city for tech. How long have you been coding in Go?"),
		agentflow.NewUserMessage("About 3 years. I mainly work on backend APIs and I prefer using Chi router."),
		agentflow.NewAssistantMessage("Chi is excellent for Go APIs. Have you tried agentflow for building AI agents?"),
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	memories, err := plan.ExtractMemories(ctx, provider, messages)
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
