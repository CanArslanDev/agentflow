package agentflow_test

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/groq"
	"github.com/CanArslanDev/agentflow/provider/mock"
	"github.com/CanArslanDev/agentflow/tools"
	"github.com/CanArslanDev/agentflow/tools/builtin"
)

// --- Unit tests ---

// TestModeLocal_AllToolsVisible — local modda tüm tool'lar model'e gönderilir.
func TestModeLocal_AllToolsVisible(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(mock.TextDelta("OK")),
	)

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(builtin.All()...),
		agentflow.WithExecutionMode(agentflow.ModeLocal),
	)

	toolNames := agent.Tools()
	if len(toolNames) != 11 { // bash, read_file, write_file, edit_file, list_dir, glob, grep, http_request, web_search, deep_search, sleep
		t.Errorf("expected 11 tools in local mode, got %d: %v", len(toolNames), toolNames)
	}
}

// TestModeRemote_LocalToolsHidden — remote modda local tool'lar model'e gönderilmez.
func TestModeRemote_LocalToolsHidden(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(
			// Model tries to call bash (shouldn't be in schema, but testing guard).
			mock.ToolCallEvent("tc_1", "bash", `{"command": "ls"}`),
		),
		mock.WithResponse(mock.TextDelta("OK")),
	)

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(builtin.All()...),
		agentflow.WithExecutionMode(agentflow.ModeRemote),
		agentflow.WithMaxTurns(3),
	)

	var toolEndResult *agentflow.ToolResult
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("test"),
	}) {
		if ev.Type == agentflow.EventToolEnd {
			toolEndResult = &ev.ToolEnd.Result
		}
	}

	// bash should be blocked in remote mode.
	if toolEndResult == nil || !toolEndResult.IsError {
		t.Error("expected bash to be blocked in remote mode")
	}
	if toolEndResult != nil {
		t.Logf("Blocked: %s", toolEndResult.Content)
	}
}

// TestModeRemote_RemoteToolsWork — remote modda http_request ve web_search çalışır.
func TestModeRemote_RemoteToolsWork(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(
			mock.ToolCallEvent("tc_1", "http_request", `{"url": "https://httpbin.org/get"}`),
		),
		mock.WithResponse(mock.TextDelta("Got it")),
	)

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(builtin.All()...),
		agentflow.WithExecutionMode(agentflow.ModeRemote),
		agentflow.WithMaxTurns(3),
	)

	var toolEndResult *agentflow.ToolResult
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("test"),
	}) {
		if ev.Type == agentflow.EventToolEnd {
			toolEndResult = &ev.ToolEnd.Result
		}
	}

	if toolEndResult == nil {
		t.Fatal("expected tool result")
	}
	if toolEndResult.IsError {
		t.Errorf("http_request should work in remote mode, got error: %s", toolEndResult.Content)
	}
}

// TestModeRemote_SleepAllowed — ToolAny tool'lar remote'da çalışır.
func TestModeRemote_SleepAllowed(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(
			mock.ToolCallEvent("tc_1", "sleep", `{"seconds": 0.01}`),
		),
		mock.WithResponse(mock.TextDelta("Done")),
	)

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(builtin.All()...),
		agentflow.WithExecutionMode(agentflow.ModeRemote),
		agentflow.WithMaxTurns(3),
	)

	var isError bool
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("wait"),
	}) {
		if ev.Type == agentflow.EventToolEnd {
			isError = ev.ToolEnd.Result.IsError
		}
	}

	if isError {
		t.Error("sleep (ToolAny) should work in remote mode")
	}
}

// TestCustomToolDefaultLocality — LocalityAware implement etmeyen tool = LocalOnly.
func TestCustomToolDefaultLocality(t *testing.T) {
	// Bu tool LocalityAware implement etmiyor — default LocalOnly olmalı.
	customTool := tools.New("custom", "Custom tool").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(true).ReadOnly(true).
		WithExecute(func(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{Content: "custom result"}, nil
		}).Build()

	// ToolBuilder builtTool implements LocalityAware with default ToolLocalOnly.
	locality := agentflow.ToolLocalOnly
	if la, ok := customTool.(agentflow.LocalityAware); ok {
		locality = la.Locality()
	}

	if locality != agentflow.ToolLocalOnly {
		t.Errorf("expected ToolLocalOnly default, got %d", locality)
	}
}

// TestCustomToolRemoteSafe — Builder ile RemoteSafe() işaretlenmiş tool.
func TestCustomToolRemoteSafe(t *testing.T) {
	remoteTool := tools.New("api_call", "Call an API").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(true).ReadOnly(true).
		RemoteSafe().
		WithExecute(func(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{Content: "api result"}, nil
		}).Build()

	la, ok := remoteTool.(agentflow.LocalityAware)
	if !ok {
		t.Fatal("expected LocalityAware implementation")
	}
	if la.Locality() != agentflow.ToolRemoteSafe {
		t.Errorf("expected ToolRemoteSafe, got %d", la.Locality())
	}

	// Should be allowed in remote mode.
	if !agentflow.IsToolAllowed(remoteTool, agentflow.ModeRemote) {
		t.Error("RemoteSafe tool should be allowed in ModeRemote")
	}
}

// TestRemoteRegistry — builtin.Remote() sadece remote-safe tool'lar döner.
func TestRemoteRegistry(t *testing.T) {
	remote := builtin.Remote()
	for _, tool := range remote {
		if !agentflow.IsToolAllowed(tool, agentflow.ModeRemote) {
			t.Errorf("tool %s in Remote() set but not allowed in ModeRemote", tool.Name())
		}
	}
	if len(remote) < 2 {
		t.Errorf("expected at least 2 remote tools, got %d", len(remote))
	}

	// Verify specific tools.
	names := make(map[string]bool)
	for _, tool := range remote {
		names[tool.Name()] = true
	}
	if !names["http_request"] {
		t.Error("missing http_request in Remote()")
	}
	if !names["web_search"] {
		t.Error("missing web_search in Remote()")
	}
	if names["bash"] {
		t.Error("bash should NOT be in Remote()")
	}
}

// TestWebSearch_Unit — web search tool basic functionality.
func TestWebSearch_Unit(t *testing.T) {
	tool := builtin.WebSearch()
	if tool.Name() != "web_search" {
		t.Errorf("expected 'web_search', got %q", tool.Name())
	}

	la, ok := tool.(agentflow.LocalityAware)
	if !ok || la.Locality() != agentflow.ToolRemoteSafe {
		t.Error("web_search should be RemoteSafe")
	}
}

// --- Integration tests ---

// TestIntegration_WebSearch — gerçek DuckDuckGo araması.
func TestIntegration_WebSearch(t *testing.T) {
	tool := builtin.WebSearch()

	result, err := tool.Execute(
		context.Background(),
		json.RawMessage(`{"query": "Go programming language", "max_results": 3}`),
		nil,
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("search error: %s", result.Content)
	}

	t.Logf("Search results:\n%s", result.Content)
	if result.Content == "" || result.Content == "no results found for: Go programming language" {
		t.Error("expected search results")
	}
}

// TestIntegration_RemoteModeWithGroq — Groq + remote mode + web search.
func TestIntegration_RemoteModeWithGroq(t *testing.T) {
	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		t.Skip("GROQ_API_KEY not set")
	}

	provider := groq.New(key, "llama-3.3-70b-versatile")

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(builtin.All()...), // Tüm tool'lar register edilir...
		agentflow.WithExecutionMode(agentflow.ModeRemote), // ...ama remote modda sadece remote-safe olanlar görünür.
		agentflow.WithSystemPrompt("You are a research assistant. Use web_search to find information. You can also use http_request for APIs."),
		agentflow.WithMaxTurns(5),
		agentflow.WithMaxTokens(500),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var text string
	var toolCalls []string
	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("Search for the latest Go programming language version"),
	}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			text += ev.TextDelta.Text
		case agentflow.EventToolStart:
			toolCalls = append(toolCalls, ev.ToolStart.ToolCall.Name)
			t.Logf("Tool: %s", ev.ToolStart.ToolCall.Name)
		case agentflow.EventToolEnd:
			t.Logf("Result (%d chars): %s...", len(ev.ToolEnd.Result.Content),
				truncate(ev.ToolEnd.Result.Content, 100))
		case agentflow.EventTurnEnd:
			t.Logf("Turn %d: %s", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		}
	}

	t.Logf("Response: %s", truncate(text, 200))

	// Verify that only remote-safe tools were called (no bash, read_file, etc.).
	for _, name := range toolCalls {
		if name == "bash" || name == "read_file" || name == "write_file" || name == "edit_file" {
			t.Errorf("local-only tool %q should not be called in remote mode", name)
		}
	}
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}
