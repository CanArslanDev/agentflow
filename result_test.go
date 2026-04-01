package agentflow_test

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/groq"
	"github.com/CanArslanDev/agentflow/provider/mock"
	"github.com/CanArslanDev/agentflow/tools"
)

// TestTruncateLimiter_SmallResult — küçük sonuç kesilmez.
func TestTruncateLimiter_SmallResult(t *testing.T) {
	limiter := agentflow.TruncateLimiter{}
	result := &agentflow.ToolResult{Content: "short"}
	got := limiter.Limit(result, 1000)
	if got.Content != "short" {
		t.Errorf("expected unchanged, got %q", got.Content)
	}
}

// TestTruncateLimiter_LargeResult — büyük sonuç kesilir, notice eklenir.
func TestTruncateLimiter_LargeResult(t *testing.T) {
	limiter := agentflow.TruncateLimiter{}
	content := strings.Repeat("x", 10000)
	result := &agentflow.ToolResult{Content: content}

	got := limiter.Limit(result, 1000)
	if len(got.Content) > 1100 { // allow some slack for notice
		t.Errorf("expected ~1000 chars, got %d", len(got.Content))
	}
	if !strings.Contains(got.Content, "truncated") {
		t.Error("expected truncation notice")
	}
	// Head should be preserved.
	if !strings.HasPrefix(got.Content, "xxx") {
		t.Error("expected head to be preserved")
	}
	// Tail should be preserved.
	if !strings.HasSuffix(got.Content, "xxx") {
		t.Error("expected tail to be preserved")
	}
	// Metadata should have truncation info.
	if got.Metadata["truncated_chars"] == nil {
		t.Error("expected truncated_chars in metadata")
	}
	t.Logf("Truncated from %d to %d chars, omitted %v",
		len(content), len(got.Content), got.Metadata["truncated_chars"])
}

// TestHeadTailLimiter — head/tail ratio çalışır.
func TestHeadTailLimiter(t *testing.T) {
	limiter := agentflow.HeadTailLimiter{HeadRatio: 0.5}
	content := strings.Repeat("A", 5000) + strings.Repeat("B", 5000)
	result := &agentflow.ToolResult{Content: content}

	got := limiter.Limit(result, 1000)
	if len(got.Content) > 1100 {
		t.Errorf("expected ~1000 chars, got %d", len(got.Content))
	}
	if !strings.Contains(got.Content, "omitted") {
		t.Error("expected omission notice")
	}
	// With 0.5 ratio, head should be ~half A's, tail ~half B's.
	if !strings.HasPrefix(got.Content, "AAA") {
		t.Error("expected head to start with A's")
	}
	if !strings.HasSuffix(got.Content, "BBB") {
		t.Error("expected tail to end with B's")
	}
}

// TestNoLimiter — hiçbir şey değişmez.
func TestNoLimiter(t *testing.T) {
	limiter := agentflow.NoLimiter{}
	content := strings.Repeat("x", 100000)
	result := &agentflow.ToolResult{Content: content}

	got := limiter.Limit(result, 1000)
	if len(got.Content) != 100000 {
		t.Errorf("expected unchanged 100000 chars, got %d", len(got.Content))
	}
}

// TestAgentResultLimiting — agent loop'ta büyük tool result'lar kesilir.
func TestAgentResultLimiting(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(
			mock.ToolCallEvent("tc_1", "big_output", `{}`),
		),
		mock.WithResponse(
			mock.TextDelta("Got it."),
		),
	)

	bigTool := tools.New("big_output", "Returns huge output").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(true).ReadOnly(true).
		WithExecute(func(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{Content: strings.Repeat("data", 25000)}, nil // 100K chars
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(bigTool),
		agentflow.WithMaxResultSize(500),
	)

	var toolEndContent string
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("get big data"),
	}) {
		if ev.Type == agentflow.EventToolEnd {
			toolEndContent = ev.ToolEnd.Result.Content
		}
	}

	if len(toolEndContent) > 600 {
		t.Errorf("expected truncated result ~500 chars, got %d", len(toolEndContent))
	}
	if !strings.Contains(toolEndContent, "truncated") {
		t.Error("expected truncation notice in result")
	}
	t.Logf("Result truncated to %d chars", len(toolEndContent))
}

// TestAgentErrorResultNotTruncated — hata sonuçları kesilmez.
func TestAgentErrorResultNotTruncated(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(
			mock.ToolCallEvent("tc_1", "failing", `{}`),
		),
		mock.WithResponse(
			mock.TextDelta("OK"),
		),
	)

	failing := tools.New("failing", "Fails with long error").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(true).ReadOnly(true).
		WithExecute(func(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{
				Content: strings.Repeat("error details ", 1000),
				IsError: true,
			}, nil
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(failing),
		agentflow.WithMaxResultSize(100), // Very small limit.
	)

	var toolEndContent string
	var isError bool
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("do it"),
	}) {
		if ev.Type == agentflow.EventToolEnd {
			toolEndContent = ev.ToolEnd.Result.Content
			isError = ev.ToolEnd.Result.IsError
		}
	}

	// Error results should NOT be truncated — model needs full error info.
	if len(toolEndContent) < 1000 {
		t.Errorf("expected error result to NOT be truncated, got %d chars", len(toolEndContent))
	}
	if !isError {
		t.Error("expected IsError=true")
	}
}

// TestIntegration_ResultLimiting — gerçek API ile büyük tool output kesilir.
func TestIntegration_ResultLimiting(t *testing.T) {
	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		t.Skip("GROQ_API_KEY not set")
	}

	provider := groq.New(key, "llama-3.3-70b-versatile")

	bigSearch := tools.New("search_database", "Search the database and return matching records.").
		WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{"type": "string"},
			},
			"required": []string{"query"},
		}).
		ConcurrencySafe(true).ReadOnly(true).
		WithExecute(func(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			// Simulate a huge database result.
			var records []string
			for i := 0; i < 500; i++ {
				records = append(records, fmt.Sprintf(`{"id":%d,"name":"Record %d","data":"%s"}`, i, i, strings.Repeat("x", 100)))
			}
			return &agentflow.ToolResult{Content: "[" + strings.Join(records, ",") + "]"}, nil
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(bigSearch),
		agentflow.WithSystemPrompt("You are a database assistant. Use search_database to answer questions. Summarize the results briefly."),
		agentflow.WithMaxTurns(3),
		agentflow.WithMaxTokens(300),
		agentflow.WithMaxResultSize(2000), // Limit to 2K chars.
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var text string
	var toolResultSize int

	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("Search for all records"),
	}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			text += ev.TextDelta.Text
		case agentflow.EventToolEnd:
			toolResultSize = len(ev.ToolEnd.Result.Content)
			t.Logf("Tool result size: %d chars (truncated: %v)",
				toolResultSize, strings.Contains(ev.ToolEnd.Result.Content, "truncated"))
		case agentflow.EventTurnEnd:
			t.Logf("Turn %d: %s", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		}
	}

	if toolResultSize > 2500 {
		t.Errorf("expected result truncated to ~2000 chars, got %d", toolResultSize)
	}
	t.Logf("Response: %q", text)
}
