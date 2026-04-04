package groq_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/groq"
)

func sseResponse(chunks ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for _, chunk := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", chunk)
		}
		fmt.Fprintf(w, "data: [DONE]\n\n")
	}
}

func TestGroq_MaxCompletionTokens(t *testing.T) {
	var receivedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&receivedBody)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0}]}\n\n")
		fmt.Fprintf(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	provider := groq.New("test-key", "llama-3.3-70b-versatile", groq.WithBaseURL(server.URL))

	stream, err := provider.CreateStream(context.Background(), &agentflow.Request{
		Messages:  []agentflow.Message{agentflow.NewUserMessage("Hi")},
		MaxTokens: 1024,
	})
	if err != nil {
		t.Fatalf("create stream: %v", err)
	}
	defer stream.Close()
	// Drain.
	for {
		_, err := stream.Next()
		if err == io.EOF {
			break
		}
	}

	// max_completion_tokens should be set, max_tokens should NOT be set.
	if mct, ok := receivedBody["max_completion_tokens"]; !ok {
		t.Error("max_completion_tokens not found in request body")
	} else if mct != float64(1024) {
		t.Errorf("max_completion_tokens: expected 1024, got %v", mct)
	}

	if mt, ok := receivedBody["max_tokens"]; ok && mt != float64(0) {
		t.Errorf("max_tokens should not be set (or zero), got %v", mt)
	}
}

func TestGroq_NoMaxTokens(t *testing.T) {
	var receivedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&receivedBody)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0}]}\n\n")
		fmt.Fprintf(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	provider := groq.New("test-key", "llama-3.3-70b-versatile", groq.WithBaseURL(server.URL))

	stream, err := provider.CreateStream(context.Background(), &agentflow.Request{
		Messages: []agentflow.Message{agentflow.NewUserMessage("Hi")},
		// No MaxTokens set.
	})
	if err != nil {
		t.Fatalf("create stream: %v", err)
	}
	defer stream.Close()
	for {
		_, err := stream.Next()
		if err == io.EOF {
			break
		}
	}

	// Neither field should be present when MaxTokens is 0.
	if mct, ok := receivedBody["max_completion_tokens"]; ok && mct != float64(0) {
		t.Errorf("max_completion_tokens should not be set, got %v", mct)
	}
}

func TestGroq_SimpleTextResponse(t *testing.T) {
	server := httptest.NewServer(sseResponse(
		`{"choices":[{"delta":{"content":"Hello"},"index":0}]}`,
		`{"choices":[{"delta":{"content":" World"},"index":0}]}`,
		`{"choices":[{"finish_reason":"stop","delta":{},"index":0}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
	))
	defer server.Close()

	provider := groq.New("test-key", "llama-3.3-70b-versatile", groq.WithBaseURL(server.URL))

	stream, err := provider.CreateStream(context.Background(), &agentflow.Request{
		Messages: []agentflow.Message{agentflow.NewUserMessage("Hi")},
	})
	if err != nil {
		t.Fatalf("create stream: %v", err)
	}
	defer stream.Close()

	var text string
	for {
		ev, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		if ev.Type == agentflow.StreamEventDelta && ev.Delta != nil {
			text += ev.Delta.Text
		}
	}

	if text != "Hello World" {
		t.Errorf("expected 'Hello World', got %q", text)
	}
}

func TestGroq_ModelID(t *testing.T) {
	provider := groq.New("key", "llama-3.3-70b-versatile")
	if provider.ModelID() != "llama-3.3-70b-versatile" {
		t.Errorf("model ID: %s", provider.ModelID())
	}
}
