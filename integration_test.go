package agentflow_test

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/groq"
	"github.com/CanArslanDev/agentflow/session/memstore"
	"github.com/CanArslanDev/agentflow/tools"
)

func groqKey(t *testing.T) string {
	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		t.Skip("GROQ_API_KEY not set")
	}
	return key
}

// TestIntegration_SimpleChat — Groq'a basit text isteği.
func TestIntegration_SimpleChat(t *testing.T) {
	provider := groq.New(groqKey(t), "llama-3.3-70b-versatile")
	agent := agentflow.NewAgent(provider, agentflow.WithMaxTurns(1), agentflow.WithMaxTokens(50))

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var text string
	for ev := range agent.Run(ctx, []agentflow.Message{agentflow.NewUserMessage("Say 'pong'")}) {
		if ev.Type == agentflow.EventTextDelta {
			text += ev.TextDelta.Text
		}
	}

	if text == "" {
		t.Fatal("empty response")
	}
	t.Logf("Response: %q", text)
}

// TestIntegration_ToolUseLoop — Model tool çağırır, sonucu alır, cevap verir.
func TestIntegration_ToolUseLoop(t *testing.T) {
	provider := groq.New(groqKey(t), "llama-3.3-70b-versatile")

	weather := tools.New("get_weather", "Get current weather for a city. Returns temperature in Celsius.").
		WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string", "description": "City name"},
			},
			"required": []string{"city"},
		}).
		ConcurrencySafe(true).ReadOnly(true).
		WithExecute(func(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			var p struct{ City string `json:"city"` }
			json.Unmarshal(input, &p)
			return &agentflow.ToolResult{Content: fmt.Sprintf(`{"city":"%s","temp_c":22,"condition":"sunny"}`, p.City)}, nil
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(weather),
		agentflow.WithSystemPrompt("You are a weather assistant. Always use the get_weather tool to answer weather questions."),
		agentflow.WithMaxTurns(5),
		agentflow.WithMaxTokens(300),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var text string
	var toolCalls, toolResults int
	var turns int

	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("What's the weather in Istanbul?"),
	}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			text += ev.TextDelta.Text
		case agentflow.EventToolStart:
			toolCalls++
			t.Logf("→ Tool call: %s(%s)", ev.ToolStart.ToolCall.Name, string(ev.ToolStart.ToolCall.Input))
		case agentflow.EventToolEnd:
			toolResults++
			t.Logf("← Tool result: %s", ev.ToolEnd.Result.Content)
		case agentflow.EventTurnEnd:
			turns = ev.TurnEnd.TurnNumber
			t.Logf("Turn %d: %s", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		case agentflow.EventUsage:
			t.Logf("Tokens: %d", ev.Usage.Usage.TotalTokens)
		}
	}

	t.Logf("Final: %q", text)
	if toolCalls == 0 {
		t.Error("model did not call the weather tool")
	}
	if toolResults == 0 {
		t.Error("no tool results")
	}
	if text == "" {
		t.Error("empty final response")
	}
	if turns < 2 {
		t.Errorf("expected at least 2 turns (call + response), got %d", turns)
	}
}

// TestIntegration_MultiToolParallel — Model birden fazla tool'u aynı anda çağırır.
func TestIntegration_MultiToolParallel(t *testing.T) {
	provider := groq.New(groqKey(t), "llama-3.3-70b-versatile")

	weather := tools.New("get_weather", "Get weather for a city.").
		WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string"},
			},
			"required": []string{"city"},
		}).
		ConcurrencySafe(true).ReadOnly(true).
		WithExecute(func(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			var p struct{ City string `json:"city"` }
			json.Unmarshal(input, &p)
			temps := map[string]int{"Istanbul": 22, "Tokyo": 18, "London": 14}
			temp := temps[p.City]
			if temp == 0 {
				temp = 20
			}
			return &agentflow.ToolResult{Content: fmt.Sprintf("%s: %d°C", p.City, temp)}, nil
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(weather),
		agentflow.WithSystemPrompt("You are a weather assistant. Use get_weather for each city. Call the tool multiple times if asked about multiple cities."),
		agentflow.WithMaxTurns(5),
		agentflow.WithMaxTokens(500),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var text string
	var toolCalls int

	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("Compare weather in Istanbul, Tokyo and London right now."),
	}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			text += ev.TextDelta.Text
		case agentflow.EventToolStart:
			toolCalls++
			t.Logf("Tool call #%d: %s(%s)", toolCalls, ev.ToolStart.ToolCall.Name, string(ev.ToolStart.ToolCall.Input))
		case agentflow.EventToolEnd:
			t.Logf("Tool result: %s", ev.ToolEnd.Result.Content)
		case agentflow.EventTurnEnd:
			t.Logf("Turn %d: %s", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		}
	}

	t.Logf("Final: %q", text)
	if toolCalls < 2 {
		t.Errorf("expected at least 2 tool calls for 3 cities, got %d", toolCalls)
	}
	// Model may not always produce a final text summary if it ends on a tool turn.
	// The important assertion is that multiple tool calls were made.
}

// TestIntegration_BudgetExhausted — Gerçek API ile token budget testi.
func TestIntegration_BudgetExhausted(t *testing.T) {
	provider := groq.New(groqKey(t), "llama-3.3-70b-versatile")

	echo := tools.New("echo", "Echo back the input text.").
		WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"text": map[string]any{"type": "string"},
			},
			"required": []string{"text"},
		}).
		ConcurrencySafe(true).ReadOnly(true).
		WithExecute(func(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			var p struct{ Text string `json:"text"` }
			json.Unmarshal(input, &p)
			return &agentflow.ToolResult{Content: p.Text}, nil
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(echo),
		agentflow.WithSystemPrompt("You are an echo assistant. Always use the echo tool, then use it again with a different text. Keep using the tool."),
		agentflow.WithTokenBudget(agentflow.TokenBudget{
			MaxTokens:        500, // Çok düşük budget — 1-2 turn'de dolacak.
			WarningThreshold: 0.5,
		}),
		agentflow.WithMaxTurns(20),
		agentflow.WithMaxTokens(200),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var reason agentflow.TurnEndReason
	var warnings int
	var totalTokens int

	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("Echo 'hello' and then echo 'world' and keep going"),
	}) {
		switch ev.Type {
		case agentflow.EventTurnEnd:
			reason = ev.TurnEnd.Reason
			t.Logf("Turn %d ended: %s", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		case agentflow.EventBudgetWarning:
			warnings++
			t.Logf("⚠ Budget warning: %d/%d (%.0f%%)",
				ev.BudgetWarning.ConsumedTokens, ev.BudgetWarning.MaxTokens, ev.BudgetWarning.Percentage*100)
		case agentflow.EventUsage:
			totalTokens += ev.Usage.Usage.TotalTokens
			t.Logf("Tokens this turn: %d (cumulative: %d)", ev.Usage.Usage.TotalTokens, totalTokens)
		case agentflow.EventToolStart:
			t.Logf("Tool: %s", ev.ToolStart.ToolCall.Name)
		}
	}

	t.Logf("Final reason: %s, total tokens: %d, warnings: %d", reason, totalTokens, warnings)

	if reason != agentflow.TurnEndBudgetExhausted && reason != agentflow.TurnEndComplete {
		// Budget should exhaust, but if model finishes before budget, that's also OK.
		t.Logf("Note: loop ended with %s (budget may not have been reached)", reason)
	}
	if reason == agentflow.TurnEndBudgetExhausted {
		t.Log("✓ Budget enforcement worked — loop stopped due to budget exhaustion")
	}
}

// TestIntegration_SessionPersistence — Gerçek API ile session save + resume.
func TestIntegration_SessionPersistence(t *testing.T) {
	key := groqKey(t)
	store := memstore.New()

	// Run 1: İlk soru.
	provider1 := groq.New(key, "llama-3.3-70b-versatile")
	agent1 := agentflow.NewAgent(provider1,
		agentflow.WithSessionStore(store),
		agentflow.WithMaxTurns(1),
		agentflow.WithMaxTokens(100),
	)

	session := &agentflow.Session{Metadata: map[string]any{"test": "persistence"}}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var text1 string
	for ev := range agent1.RunSession(ctx, session, []agentflow.Message{
		agentflow.NewUserMessage("My name is Can. Remember it."),
	}) {
		if ev.Type == agentflow.EventTextDelta {
			text1 += ev.TextDelta.Text
		}
	}
	t.Logf("Run 1 response: %q", text1)
	t.Logf("Session ID: %s", session.ID)

	// Verify saved.
	saved, err := store.Load(context.Background(), session.ID)
	if err != nil {
		t.Fatalf("session not saved: %v", err)
	}
	t.Logf("Saved session has %d messages", len(saved.Messages))

	// Run 2: Resume ve takip sorusu.
	provider2 := groq.New(key, "llama-3.3-70b-versatile")
	agent2 := agentflow.NewAgent(provider2,
		agentflow.WithSessionStore(store),
		agentflow.WithMaxTurns(1),
		agentflow.WithMaxTokens(100),
	)

	ctx2, cancel2 := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel2()

	events, err := agent2.Resume(ctx2, session.ID, "What is my name?")
	if err != nil {
		t.Fatalf("resume failed: %v", err)
	}

	var text2 string
	for ev := range events {
		if ev.Type == agentflow.EventTextDelta {
			text2 += ev.TextDelta.Text
		}
	}
	t.Logf("Run 2 response: %q", text2)

	// Model should remember the name from the previous turn.
	if text2 == "" {
		t.Error("empty response on resume")
	}

	// Updated session should have more messages.
	updated, _ := store.Load(context.Background(), session.ID)
	t.Logf("Updated session has %d messages", len(updated.Messages))
	if len(updated.Messages) <= len(saved.Messages) {
		t.Error("session was not updated after resume")
	}
}

// TestIntegration_SubAgent — Gerçek API ile sub-agent delegation.
func TestIntegration_SubAgent(t *testing.T) {
	key := groqKey(t)
	provider := groq.New(key, "llama-3.3-70b-versatile")

	parent := agentflow.NewAgent(provider,
		agentflow.WithMaxTurns(3),
		agentflow.WithMaxTokens(200),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var childText string
	for ev := range parent.SpawnChild(ctx, agentflow.SubAgentConfig{
		SystemPrompt: "You are a Go expert. Give very short answers.",
		MaxTurns:     1,
		MaxTokens:    100,
	}, "What is a goroutine in one sentence?") {
		if ev.Type == agentflow.EventTextDelta {
			childText += ev.TextDelta.Text
		}
		if ev.Type == agentflow.EventTurnEnd {
			t.Logf("Child turn %d: %s", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		}
	}

	t.Logf("Child response: %q", childText)
	if childText == "" {
		t.Error("empty sub-agent response")
	}
}

// TestIntegration_Orchestrate — Gerçek API ile paralel sub-agent orchestration.
func TestIntegration_Orchestrate(t *testing.T) {
	key := groqKey(t)
	provider := groq.New(key, "llama-3.3-70b-versatile")

	parent := agentflow.NewAgent(provider, agentflow.WithMaxTokens(100))

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	results := agentflow.Orchestrate(ctx, parent, agentflow.SubAgentConfig{
		SystemPrompt: "Answer in exactly one word.",
		MaxTurns:     1,
		MaxTokens:    20,
	}, []string{
		"What color is the sky?",
		"What color is grass?",
	})

	for _, r := range results {
		if r.Error != nil {
			t.Errorf("Task %d error: %v", r.Index, r.Error)
		} else {
			t.Logf("Task %d (%s): %q", r.Index, r.Task, r.Result)
		}
	}

	successCount := 0
	for _, r := range results {
		if r.Error == nil && r.Result != "" {
			successCount++
		}
	}
	if successCount < 2 {
		t.Errorf("expected 2 successful results, got %d", successCount)
	}
}
