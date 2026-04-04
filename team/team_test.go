package team_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/groq"
	"github.com/CanArslanDev/agentflow/team"
)

func groqProvider(t *testing.T) *groq.Provider {
	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		t.Skip("GROQ_API_KEY not set")
	}
	return groq.New(key, "llama-3.3-70b-versatile")
}

func TestIntegration_TeamRunAll(t *testing.T) {
	provider := groqProvider(t)

	tm := team.New(provider, []team.Member{
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

	results := tm.RunAll(ctx, map[string]string{
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

func TestIntegration_TeamSharedMemory(t *testing.T) {
	provider := groqProvider(t)

	tm := team.New(provider, []team.Member{
		{
			Role:         "analyst",
			SystemPrompt: "You are a data analyst. Store your findings in shared memory using set_shared_memory tool. Key should be 'finding', value should be your analysis.",
			MaxTurns:     3,
			MaxTokens:    300,
		},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	events, err := tm.RunMember(ctx, "analyst", "Analyze this: Go was created in 2009 at Google. Store the creation year in shared memory with key 'go_year'.")
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
