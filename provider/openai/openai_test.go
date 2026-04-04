package openai_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/openai"
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

func TestOpenAI_SimpleTextResponse(t *testing.T) {
	server := httptest.NewServer(sseResponse(
		`{"choices":[{"delta":{"content":"Hello"},"index":0}]}`,
		`{"choices":[{"delta":{"content":" World"},"index":0}]}`,
		`{"choices":[{"finish_reason":"stop","delta":{},"index":0}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
	))
	defer server.Close()

	provider := openai.New("test-key", "gpt-4o", openai.WithBaseURL(server.URL))

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
		t.Errorf("expected 15 total tokens, got %d", usage.TotalTokens)
	}
}

func TestOpenAI_ToolCallResponse(t *testing.T) {
	server := httptest.NewServer(sseResponse(
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":""}}]},"index":0}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]},"index":0}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"NYC\"}"}}]},"index":0}]}`,
		`{"choices":[{"finish_reason":"tool_calls","delta":{},"index":0}]}`,
	))
	defer server.Close()

	provider := openai.New("test-key", "gpt-4o", openai.WithBaseURL(server.URL))

	stream, err := provider.CreateStream(context.Background(), &agentflow.Request{
		Messages: []agentflow.Message{agentflow.NewUserMessage("Weather in NYC?")},
		Tools: []agentflow.ToolDefinition{{
			Name:        "get_weather",
			Description: "Get weather",
			InputSchema: map[string]any{"type": "object"},
		}},
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
	if toolCall.ID != "call_1" {
		t.Errorf("tool ID: %s", toolCall.ID)
	}
	if string(toolCall.Input) != `{"city":"NYC"}` {
		t.Errorf("tool input: %s", string(toolCall.Input))
	}
}

func TestOpenAI_ErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`{"error":{"message":"rate limited"}}`))
	}))
	defer server.Close()

	provider := openai.New("test-key", "gpt-4o", openai.WithBaseURL(server.URL))

	_, err := provider.CreateStream(context.Background(), &agentflow.Request{
		Messages: []agentflow.Message{agentflow.NewUserMessage("Hi")},
	})
	if err == nil {
		t.Fatal("expected error for 429")
	}

	if !agentflow.IsRetryableError(err) {
		t.Error("429 should be retryable")
	}
}

func TestOpenAI_Timeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(5 * time.Second)
	}))
	defer server.Close()

	provider := openai.New("test-key", "gpt-4o",
		openai.WithBaseURL(server.URL),
		openai.WithHTTPClient(&http.Client{Timeout: 100 * time.Millisecond}),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	_, err := provider.CreateStream(ctx, &agentflow.Request{
		Messages: []agentflow.Message{agentflow.NewUserMessage("Hi")},
	})
	if err == nil {
		t.Fatal("expected timeout error")
	}
}

func TestOpenAI_ModelID(t *testing.T) {
	provider := openai.New("key", "gpt-4o-mini")
	if provider.ModelID() != "gpt-4o-mini" {
		t.Errorf("model ID: %s", provider.ModelID())
	}
}

func TestOpenAI_DocumentMessage(t *testing.T) {
	var receivedBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0}]}\n\n")
		fmt.Fprintf(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	provider := openai.New("test-key", "gpt-4o", openai.WithBaseURL(server.URL))

	msg := agentflow.NewDocumentMessage("Summarize this",
		agentflow.DocumentContent{
			Filename:  "report.pdf",
			MediaType: "application/pdf",
			Data:      "JVBERi0xLjQ=",
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
	// Verify the request body contains file content part.
	if !containsSubstr(bodyStr, `"type":"file"`) {
		t.Errorf("request body should contain file type, got: %s", bodyStr)
	}
	if !containsSubstr(bodyStr, `"filename":"report.pdf"`) {
		t.Errorf("request body should contain filename, got: %s", bodyStr)
	}
	if !containsSubstr(bodyStr, `data:application/pdf;base64,JVBERi0xLjQ=`) {
		t.Errorf("request body should contain file_data as data URI, got: %s", bodyStr)
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

func TestOpenAI_MetadataPropagation(t *testing.T) {
	var receivedHeaders http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedHeaders = r.Header
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0}]}\n\n")
		fmt.Fprintf(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	provider := openai.New("test-key", "gpt-4o", openai.WithBaseURL(server.URL))

	stream, err := provider.CreateStream(context.Background(), &agentflow.Request{
		Messages: []agentflow.Message{agentflow.NewUserMessage("Hi")},
		Metadata: map[string]string{
			"traceparent": "00-abc123-def456-01",
			"x-custom":    "value",
		},
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

	if receivedHeaders.Get("traceparent") != "00-abc123-def456-01" {
		t.Errorf("traceparent header: %s", receivedHeaders.Get("traceparent"))
	}
	if receivedHeaders.Get("x-custom") != "value" {
		t.Errorf("x-custom header: %s", receivedHeaders.Get("x-custom"))
	}
}
