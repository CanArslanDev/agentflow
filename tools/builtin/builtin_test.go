package builtin_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/groq"
	"github.com/CanArslanDev/agentflow/tools/builtin"
)

func call(t *testing.T, tool agentflow.Tool, input string) *agentflow.ToolResult {
	t.Helper()
	result, err := tool.Execute(context.Background(), json.RawMessage(input), nil)
	if err != nil {
		t.Fatalf("%s: unexpected error: %v", tool.Name(), err)
	}
	return result
}

// --- Bash ---

func TestBash_Echo(t *testing.T) {
	r := call(t, builtin.Bash(), `{"command": "echo hello"}`)
	if r.IsError {
		t.Fatalf("expected success, got error: %s", r.Content)
	}
	if strings.TrimSpace(r.Content) != "hello" {
		t.Errorf("expected 'hello', got %q", r.Content)
	}
}

func TestBash_Failure(t *testing.T) {
	r := call(t, builtin.Bash(), `{"command": "exit 1"}`)
	if !r.IsError {
		t.Error("expected error for exit 1")
	}
}

func TestBash_Timeout(t *testing.T) {
	r := call(t, builtin.Bash(), `{"command": "sleep 10", "timeout": 500}`)
	if !r.IsError {
		t.Error("expected timeout error")
	}
}

// --- ReadFile / WriteFile / EditFile ---

func TestReadFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.txt")
	os.WriteFile(path, []byte("line1\nline2\nline3"), 0644)

	r := call(t, builtin.ReadFile(), `{"path": "`+path+`"}`)
	if r.IsError {
		t.Fatalf("read error: %s", r.Content)
	}
	if !strings.Contains(r.Content, "line2") {
		t.Errorf("expected content, got %q", r.Content)
	}
}

func TestReadFile_WithOffset(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.txt")
	os.WriteFile(path, []byte("line0\nline1\nline2\nline3"), 0644)

	r := call(t, builtin.ReadFile(), `{"path": "`+path+`", "offset": 1, "limit": 2}`)
	if !strings.Contains(r.Content, "line1") || !strings.Contains(r.Content, "line2") {
		t.Errorf("expected lines 1-2, got %q", r.Content)
	}
	if strings.Contains(r.Content, "line0") {
		t.Error("should not contain line0 (before offset)")
	}
}

func TestReadFile_NotFound(t *testing.T) {
	r := call(t, builtin.ReadFile(), `{"path": "/nonexistent/file.txt"}`)
	if !r.IsError {
		t.Error("expected error for missing file")
	}
}

func TestWriteFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "subdir", "output.txt")

	r := call(t, builtin.WriteFile(), `{"path": "`+path+`", "content": "hello world"}`)
	if r.IsError {
		t.Fatalf("write error: %s", r.Content)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read written file: %v", err)
	}
	if string(data) != "hello world" {
		t.Errorf("expected 'hello world', got %q", string(data))
	}
}

func TestEditFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "edit.txt")
	os.WriteFile(path, []byte("foo bar baz"), 0644)

	r := call(t, builtin.EditFile(), `{"path": "`+path+`", "old_string": "bar", "new_string": "qux"}`)
	if r.IsError {
		t.Fatalf("edit error: %s", r.Content)
	}

	data, _ := os.ReadFile(path)
	if string(data) != "foo qux baz" {
		t.Errorf("expected 'foo qux baz', got %q", string(data))
	}
}

func TestEditFile_NotFound(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "edit.txt")
	os.WriteFile(path, []byte("hello"), 0644)

	r := call(t, builtin.EditFile(), `{"path": "`+path+`", "old_string": "missing", "new_string": "x"}`)
	if !r.IsError {
		t.Error("expected error for missing old_string")
	}
}

func TestEditFile_Ambiguous(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "edit.txt")
	os.WriteFile(path, []byte("aaa bbb aaa"), 0644)

	r := call(t, builtin.EditFile(), `{"path": "`+path+`", "old_string": "aaa", "new_string": "x"}`)
	if !r.IsError {
		t.Error("expected error for ambiguous match")
	}
}

// --- ListDir ---

func TestListDir(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "a.txt"), []byte("a"), 0644)
	os.Mkdir(filepath.Join(dir, "subdir"), 0755)

	r := call(t, builtin.ListDir(), `{"path": "`+dir+`"}`)
	if r.IsError {
		t.Fatalf("error: %s", r.Content)
	}
	if !strings.Contains(r.Content, "a.txt") || !strings.Contains(r.Content, "subdir") {
		t.Errorf("expected file and dir listing, got %q", r.Content)
	}
}

// --- Glob ---

func TestGlob(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "a.go"), []byte(""), 0644)
	os.WriteFile(filepath.Join(dir, "b.go"), []byte(""), 0644)
	os.WriteFile(filepath.Join(dir, "c.txt"), []byte(""), 0644)

	r := call(t, builtin.Glob(), `{"pattern": "*.go", "path": "`+dir+`"}`)
	if r.IsError {
		t.Fatalf("error: %s", r.Content)
	}
	if !strings.Contains(r.Content, "a.go") || !strings.Contains(r.Content, "b.go") {
		t.Errorf("expected .go files, got %q", r.Content)
	}
	if strings.Contains(r.Content, "c.txt") {
		t.Error("should not match .txt")
	}
}

// --- Grep ---

func TestGrep(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "a.go"), []byte("func main() {\n\tprintln(\"hello\")\n}"), 0644)
	os.WriteFile(filepath.Join(dir, "b.txt"), []byte("no match here"), 0644)

	r := call(t, builtin.Grep(), `{"pattern": "println", "path": "`+dir+`"}`)
	if r.IsError {
		t.Fatalf("error: %s", r.Content)
	}
	if !strings.Contains(r.Content, "println") {
		t.Errorf("expected match, got %q", r.Content)
	}
	if !strings.Contains(r.Content, "a.go:2") {
		t.Errorf("expected file:line format, got %q", r.Content)
	}
}

func TestGrep_WithGlob(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "a.go"), []byte("hello"), 0644)
	os.WriteFile(filepath.Join(dir, "b.txt"), []byte("hello"), 0644)

	r := call(t, builtin.Grep(), `{"pattern": "hello", "path": "`+dir+`", "glob": "*.go"}`)
	if !strings.Contains(r.Content, "a.go") {
		t.Error("expected .go match")
	}
	if strings.Contains(r.Content, "b.txt") {
		t.Error("should not match .txt with glob filter")
	}
}

// --- HTTPRequest ---

func TestHTTPRequest(t *testing.T) {
	r := call(t, builtin.HTTPRequest(), `{"url": "https://httpbin.org/get", "method": "GET"}`)
	if r.IsError {
		t.Fatalf("error: %s", r.Content)
	}
	if !strings.Contains(r.Content, "200") {
		t.Errorf("expected 200 status, got %q", r.Content[:100])
	}
}

// --- Sleep ---

func TestSleep(t *testing.T) {
	start := time.Now()
	r := call(t, builtin.Sleep(), `{"seconds": 0.1}`)
	elapsed := time.Since(start)

	if r.IsError {
		t.Fatalf("error: %s", r.Content)
	}
	if elapsed < 80*time.Millisecond {
		t.Errorf("slept too little: %v", elapsed)
	}
}

func TestSleep_MaxCap(t *testing.T) {
	// Should cap at 60s, but we cancel via context before that.
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	result, _ := builtin.Sleep().Execute(ctx, json.RawMessage(`{"seconds": 999}`), nil)
	if !result.IsError {
		t.Error("expected cancellation error")
	}
}

// --- AskUser ---

func TestAskUser(t *testing.T) {
	tool := builtin.AskUser(func(_ context.Context, question string) (string, error) {
		if question != "What is your name?" {
			t.Errorf("unexpected question: %q", question)
		}
		return "Alice", nil
	})

	r := call(t, tool, `{"question": "What is your name?"}`)
	if r.IsError {
		t.Fatalf("error: %s", r.Content)
	}
	if r.Content != "Alice" {
		t.Errorf("expected 'Alice', got %q", r.Content)
	}
}

// --- Registry ---

func TestAll(t *testing.T) {
	all := builtin.All()
	if len(all) != 10 {
		t.Errorf("expected 10 tools, got %d", len(all))
	}

	names := make(map[string]bool)
	for _, tool := range all {
		names[tool.Name()] = true
	}

	expected := []string{"bash", "read_file", "write_file", "edit_file", "list_dir", "glob", "grep", "http_request", "web_search", "sleep"}
	for _, name := range expected {
		if !names[name] {
			t.Errorf("missing tool: %s", name)
		}
	}
}

func TestReadOnly(t *testing.T) {
	ro := builtin.ReadOnly()
	for _, tool := range ro {
		if !tool.IsReadOnly(nil) {
			t.Errorf("tool %s in ReadOnly set but IsReadOnly=false", tool.Name())
		}
	}
}

// --- Integration: Groq agent with built-in tools ---

func TestIntegration_AgentWithBuiltinTools(t *testing.T) {
	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		t.Skip("GROQ_API_KEY not set")
	}

	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "data.txt"), []byte("The secret number is 42."), 0644)

	provider := groq.New(key, "llama-3.3-70b-versatile")

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(builtin.ReadOnly()...),
		agentflow.WithSystemPrompt("You are a file assistant. Use tools to answer questions. IMPORTANT: Always use absolute paths. The files are in: "+dir),
		agentflow.WithMaxTurns(5),
		agentflow.WithMaxTokens(300),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var text string
	var toolCalls int
	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("Read the file data.txt and tell me the secret number."),
	}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			text += ev.TextDelta.Text
		case agentflow.EventToolStart:
			toolCalls++
			t.Logf("Tool: %s(%s)", ev.ToolStart.ToolCall.Name, string(ev.ToolStart.ToolCall.Input))
		case agentflow.EventToolEnd:
			t.Logf("Result: %s", ev.ToolEnd.Result.Content)
		case agentflow.EventTurnEnd:
			t.Logf("Turn %d: %s", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		}
	}

	t.Logf("Response: %q", text)
	if toolCalls == 0 {
		t.Error("expected tool calls")
	}
	if !strings.Contains(text, "42") {
		t.Errorf("expected '42' in response, got %q", text)
	}
}
