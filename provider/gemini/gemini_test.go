package gemini_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/gemini"
)

func geminiSSE(chunks ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for _, chunk := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", chunk)
		}
	}
}

func TestGemini_SimpleTextResponse(t *testing.T) {
	server := httptest.NewServer(geminiSSE(
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]}}]}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":" World"}]}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}`,
	))
	defer server.Close()

	provider := gemini.New("test-key", "gemini-2.0-flash", gemini.WithBaseURL(server.URL))

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

	usage := stream.Usage()
	if usage == nil {
		t.Fatal("expected usage")
	}
	if usage.TotalTokens != 15 {
		t.Errorf("total tokens: %d", usage.TotalTokens)
	}
}

func TestGemini_ToolCallResponse(t *testing.T) {
	server := httptest.NewServer(geminiSSE(
		`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"NYC"}}}]}}]}`,
	))
	defer server.Close()

	provider := gemini.New("test-key", "gemini-2.0-flash", gemini.WithBaseURL(server.URL))

	stream, err := provider.CreateStream(context.Background(), &agentflow.Request{
		Messages: []agentflow.Message{agentflow.NewUserMessage("Weather?")},
	})
	if err != nil {
		t.Fatalf("create stream: %v", err)
	}
	defer stream.Close()

	var toolCall *agentflow.ToolCall
	for {
		ev, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		if ev.Type == agentflow.StreamEventToolCall && ev.ToolCall != nil {
			toolCall = ev.ToolCall
		}
	}

	if toolCall == nil {
		t.Fatal("expected tool call")
	}
	if toolCall.Name != "get_weather" {
		t.Errorf("tool name: %s", toolCall.Name)
	}
}

func TestGemini_ErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		w.Write([]byte(`{"error":{"message":"API key invalid"}}`))
	}))
	defer server.Close()

	provider := gemini.New("bad-key", "gemini-2.0-flash", gemini.WithBaseURL(server.URL))

	_, err := provider.CreateStream(context.Background(), &agentflow.Request{
		Messages: []agentflow.Message{agentflow.NewUserMessage("Hi")},
	})
	if err == nil {
		t.Fatal("expected error for 403")
	}
}

func TestGemini_DocumentMessage(t *testing.T) {
	var receivedBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"ok\"}]}}]}\n\n")
	}))
	defer server.Close()

	provider := gemini.New("test-key", "gemini-2.0-flash", gemini.WithBaseURL(server.URL))

	msg := agentflow.NewDocumentMessage("Summarize",
		agentflow.DocumentContent{
			Filename:  "data.csv",
			MediaType: "text/csv",
			Data:      "Y29sMSxjb2wy",
		},
	)

	stream, err := provider.CreateStream(context.Background(), &agentflow.Request{
		Messages: []agentflow.Message{msg},
	})
	if err != nil {
		t.Fatalf("create stream: %v", err)
	}
	defer stream.Close()
	// Drain stream.
	for {
		_, err := stream.Next()
		if err == io.EOF {
			break
		}
	}

	bodyStr := string(receivedBody)
	// Verify inlineData with document MIME type is present.
	if !containsSubstr(bodyStr, `"mimeType":"text/csv"`) {
		t.Errorf("request body should contain mimeType, got: %s", bodyStr)
	}
	if !containsSubstr(bodyStr, `"data":"Y29sMSxjb2wy"`) {
		t.Errorf("request body should contain base64 data, got: %s", bodyStr)
	}
}

func containsSubstr(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestGemini_ModelID(t *testing.T) {
	provider := gemini.New("key", "gemini-2.0-flash")
	if provider.ModelID() != "gemini-2.0-flash" {
		t.Errorf("model ID: %s", provider.ModelID())
	}
}
