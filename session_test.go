package agentflow_test

import (
	"context"
	"os"
	"testing"

	"github.com/canarslan/agentflow"
	"github.com/canarslan/agentflow/provider/mock"
	"github.com/canarslan/agentflow/session/filestore"
	"github.com/canarslan/agentflow/session/memstore"
)

// TestMemstoreRoundTrip verifies save → load → list → delete cycle.
func TestMemstoreRoundTrip(t *testing.T) {
	store := memstore.New()
	ctx := context.Background()

	session := &agentflow.Session{
		ID:       "test-123",
		Messages: []agentflow.Message{agentflow.NewUserMessage("Hello")},
		Metadata: map[string]any{"user": "alice"},
	}

	// Save.
	if err := store.Save(ctx, session); err != nil {
		t.Fatalf("save: %v", err)
	}

	// Load.
	loaded, err := store.Load(ctx, "test-123")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if loaded.ID != "test-123" {
		t.Errorf("expected ID test-123, got %s", loaded.ID)
	}
	if len(loaded.Messages) != 1 {
		t.Errorf("expected 1 message, got %d", len(loaded.Messages))
	}
	if loaded.Messages[0].TextContent() != "Hello" {
		t.Errorf("expected 'Hello', got %q", loaded.Messages[0].TextContent())
	}

	// List.
	infos, err := store.List(ctx)
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(infos) != 1 {
		t.Fatalf("expected 1 session, got %d", len(infos))
	}
	if infos[0].Preview != "Hello" {
		t.Errorf("expected preview 'Hello', got %q", infos[0].Preview)
	}

	// Delete.
	if err := store.Delete(ctx, "test-123"); err != nil {
		t.Fatalf("delete: %v", err)
	}
	_, err = store.Load(ctx, "test-123")
	if err != agentflow.ErrSessionNotFound {
		t.Errorf("expected ErrSessionNotFound, got %v", err)
	}
}

// TestFilestoreRoundTrip verifies file-based save → load → list → delete.
func TestFilestoreRoundTrip(t *testing.T) {
	dir := t.TempDir()
	store, err := filestore.New(dir)
	if err != nil {
		t.Fatalf("new filestore: %v", err)
	}
	ctx := context.Background()

	session := &agentflow.Session{
		ID:       "file-test-456",
		Messages: []agentflow.Message{agentflow.NewUserMessage("File test")},
		Metadata: map[string]any{"env": "test"},
		ModelID:  "test-model",
	}

	// Save.
	if err := store.Save(ctx, session); err != nil {
		t.Fatalf("save: %v", err)
	}

	// Verify file exists on disk.
	entries, _ := os.ReadDir(dir)
	if len(entries) != 1 {
		t.Fatalf("expected 1 file, got %d", len(entries))
	}
	t.Logf("Session file: %s", entries[0].Name())

	// Load.
	loaded, err := store.Load(ctx, "file-test-456")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if loaded.ModelID != "test-model" {
		t.Errorf("expected model 'test-model', got %q", loaded.ModelID)
	}
	if loaded.Messages[0].TextContent() != "File test" {
		t.Errorf("expected 'File test', got %q", loaded.Messages[0].TextContent())
	}

	// List.
	infos, err := store.List(ctx)
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(infos) != 1 || infos[0].ID != "file-test-456" {
		t.Errorf("unexpected list result: %v", infos)
	}

	// Delete.
	if err := store.Delete(ctx, "file-test-456"); err != nil {
		t.Fatalf("delete: %v", err)
	}
	_, err = store.Load(ctx, "file-test-456")
	if err != agentflow.ErrSessionNotFound {
		t.Errorf("expected ErrSessionNotFound after delete, got %v", err)
	}
}

// TestRunSessionAutoSave verifies that RunSession auto-saves after each turn.
func TestRunSessionAutoSave(t *testing.T) {
	store := memstore.New()

	provider := mock.New(
		mock.WithResponse(mock.TextDelta("First response")),
	)

	agent := agentflow.NewAgent(provider,
		agentflow.WithSessionStore(store),
	)

	session := &agentflow.Session{
		Metadata: map[string]any{"test": true},
	}

	for ev := range agent.RunSession(context.Background(), session, []agentflow.Message{
		agentflow.NewUserMessage("Hello"),
	}) {
		_ = ev
	}

	// Session should have been auto-saved.
	if session.ID == "" {
		t.Fatal("expected session ID to be generated")
	}

	loaded, err := store.Load(context.Background(), session.ID)
	if err != nil {
		t.Fatalf("load after auto-save: %v", err)
	}
	if len(loaded.Messages) < 2 {
		t.Errorf("expected at least 2 messages (user+assistant), got %d", len(loaded.Messages))
	}
	if loaded.TurnCount != 1 {
		t.Errorf("expected turn count 1, got %d", loaded.TurnCount)
	}
	t.Logf("Auto-saved session %s with %d messages", loaded.ID, len(loaded.Messages))
}

// TestResumeSession verifies loading a session and continuing the conversation.
func TestResumeSession(t *testing.T) {
	store := memstore.New()

	// First run: save a session.
	provider1 := mock.New(
		mock.WithResponse(mock.TextDelta("Initial response")),
	)
	agent1 := agentflow.NewAgent(provider1, agentflow.WithSessionStore(store))

	session := &agentflow.Session{ID: "resume-test"}
	for range agent1.RunSession(context.Background(), session, []agentflow.Message{
		agentflow.NewUserMessage("First message"),
	}) {
	}

	// Verify saved.
	saved, _ := store.Load(context.Background(), "resume-test")
	if saved == nil {
		t.Fatal("session not saved")
	}
	t.Logf("Saved session has %d messages", len(saved.Messages))

	// Second run: resume with a new message.
	provider2 := mock.New(
		mock.WithResponse(mock.TextDelta("Continued response")),
	)
	agent2 := agentflow.NewAgent(provider2, agentflow.WithSessionStore(store))

	events, err := agent2.Resume(context.Background(), "resume-test", "Follow up question")
	if err != nil {
		t.Fatalf("resume: %v", err)
	}

	var text string
	for ev := range events {
		if ev.Type == agentflow.EventTextDelta {
			text += ev.TextDelta.Text
		}
	}

	if text != "Continued response" {
		t.Errorf("expected 'Continued response', got %q", text)
	}

	// Verify session was updated with new messages.
	updated, _ := store.Load(context.Background(), "resume-test")
	if len(updated.Messages) <= len(saved.Messages) {
		t.Errorf("expected more messages after resume, saved=%d updated=%d",
			len(saved.Messages), len(updated.Messages))
	}
	t.Logf("Resumed session now has %d messages", len(updated.Messages))
}

// TestResumeWithoutStore verifies that Resume returns an error when no store is set.
func TestResumeWithoutStore(t *testing.T) {
	agent := agentflow.NewAgent(mock.New())
	_, err := agent.Resume(context.Background(), "any-id", "")
	if err == nil {
		t.Error("expected error when resuming without session store")
	}
}

// TestResumeNotFound verifies that Resume returns ErrSessionNotFound.
func TestResumeNotFound(t *testing.T) {
	store := memstore.New()
	agent := agentflow.NewAgent(mock.New(), agentflow.WithSessionStore(store))
	_, err := agent.Resume(context.Background(), "nonexistent", "")
	if err != agentflow.ErrSessionNotFound {
		t.Errorf("expected ErrSessionNotFound, got %v", err)
	}
}
