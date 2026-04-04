package ollama_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/ollama"
)

// jsonlResponse creates a handler that writes JSONL lines (one JSON per line).
func jsonlResponse(lines ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.WriteHeader(http.StatusOK)
		for _, line := range lines {
			fmt.Fprintln(w, line)
		}
	}
}

func TestOllama_TextResponse(t *testing.T) {
	server := httptest.NewServer(jsonlResponse(
		`{"model":"llama3.1","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":"Hello"},"done":false}`,
		`{"model":"llama3.1","created_at":"2024-01-01T00:00:01Z","message":{"role":"assistant","content":" World"},"done":false}`,
		`{"model":"llama3.1","created_at":"2024-01-01T00:00:02Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","eval_count":10,"prompt_eval_count":25}`,
	))
	defer server.Close()

	provider := ollama.New(server.URL, "llama3.1")

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
	if usage.PromptTokens != 25 {
		t.Errorf("expected 25 prompt tokens, got %d", usage.PromptTokens)
	}
	if usage.CompletionTokens != 10 {
		t.Errorf("expected 10 completion tokens, got %d", usage.CompletionTokens)
	}
	if usage.TotalTokens != 35 {
		t.Errorf("expected 35 total tokens, got %d", usage.TotalTokens)
	}
}

func TestOllama_ToolCall(t *testing.T) {
	server := httptest.NewServer(jsonlResponse(
		`{"model":"llama3.1","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"NYC"}}}]},"done":true,"done_reason":"stop","eval_count":15,"prompt_eval_count":30}`,
	))
	defer server.Close()

	provider := ollama.New(server.URL, "llama3.1")

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
	if toolCall.ID != "ollama-call-0" {
		t.Errorf("tool ID: %s", toolCall.ID)
	}

	var args map[string]any
	if err := json.Unmarshal(toolCall.Input, &args); err != nil {
		t.Fatalf("unmarshal args: %v", err)
	}
	if args["city"] != "NYC" {
		t.Errorf("expected city=NYC, got %v", args["city"])
	}

	usage := stream.Usage()
	if usage == nil {
		t.Fatal("expected usage")
	}
	if usage.CompletionTokens != 15 {
		t.Errorf("completion tokens: %d", usage.CompletionTokens)
	}
}

func TestOllama_MultipleToolCalls(t *testing.T) {
	server := httptest.NewServer(jsonlResponse(
		`{"model":"llama3.1","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"NYC"}}},{"function":{"name":"get_weather","arguments":{"city":"London"}}},{"function":{"name":"get_weather","arguments":{"city":"Tokyo"}}}]},"done":true,"eval_count":20,"prompt_eval_count":40}`,
	))
	defer server.Close()

	provider := ollama.New(server.URL, "llama3.1")

	stream, err := provider.CreateStream(context.Background(), &agentflow.Request{
		Messages: []agentflow.Message{agentflow.NewUserMessage("Weather in NYC, London, Tokyo?")},
	})
	if err != nil {
		t.Fatalf("create stream: %v", err)
	}
	defer stream.Close()

	var toolCalls []*agentflow.ToolCall
	for {
		ev, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		if ev.Type == agentflow.StreamEventToolCall && ev.ToolCall != nil {
			toolCalls = append(toolCalls, ev.ToolCall)
		}
	}

	if len(toolCalls) != 3 {
		t.Fatalf("expected 3 tool calls, got %d", len(toolCalls))
	}

	expectedCities := []string{"NYC", "London", "Tokyo"}
	for i, tc := range toolCalls {
		if tc.ID != fmt.Sprintf("ollama-call-%d", i) {
			t.Errorf("tool call %d: expected ID ollama-call-%d, got %s", i, i, tc.ID)
		}
		var args map[string]any
		json.Unmarshal(tc.Input, &args)
		if args["city"] != expectedCities[i] {
			t.Errorf("tool call %d: expected city=%s, got %v", i, expectedCities[i], args["city"])
		}
	}
}

func TestOllama_UsageTracking(t *testing.T) {
	server := httptest.NewServer(jsonlResponse(
		`{"model":"llama3.1","message":{"role":"assistant","content":"Hi"},"done":false}`,
		`{"model":"llama3.1","message":{"role":"assistant","content":""},"done":true,"eval_count":42,"prompt_eval_count":100}`,
	))
	defer server.Close()

	provider := ollama.New(server.URL, "llama3.1")

	stream, err := provider.CreateStream(context.Background(), &agentflow.Request{
		Messages: []agentflow.Message{agentflow.NewUserMessage("Hi")},
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
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
	}

	usage := stream.Usage()
	if usage == nil {
		t.Fatal("expected usage")
	}
	if usage.PromptTokens != 100 {
		t.Errorf("prompt tokens: expected 100, got %d", usage.PromptTokens)
	}
	if usage.CompletionTokens != 42 {
		t.Errorf("completion tokens: expected 42, got %d", usage.CompletionTokens)
	}
	if usage.TotalTokens != 142 {
		t.Errorf("total tokens: expected 142, got %d", usage.TotalTokens)
	}
}

func TestOllama_ErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		w.Write([]byte(`{"error":"model 'nonexistent' not found"}`))
	}))
	defer server.Close()

	provider := ollama.New(server.URL, "nonexistent")

	_, err := provider.CreateStream(context.Background(), &agentflow.Request{
		Messages: []agentflow.Message{agentflow.NewUserMessage("Hi")},
	})
	if err == nil {
		t.Fatal("expected error for 404")
	}

	// 404 should not be retryable.
	if agentflow.IsRetryableError(err) {
		t.Error("404 should not be retryable")
	}
}

func TestOllama_ServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":"internal server error"}`))
	}))
	defer server.Close()

	provider := ollama.New(server.URL, "llama3.1")

	_, err := provider.CreateStream(context.Background(), &agentflow.Request{
		Messages: []agentflow.Message{agentflow.NewUserMessage("Hi")},
	})
	if err == nil {
		t.Fatal("expected error for 500")
	}

	if !agentflow.IsRetryableError(err) {
		t.Error("500 should be retryable")
	}
}

func TestOllama_ContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(5 * time.Second)
	}))
	defer server.Close()

	provider := ollama.New(server.URL, "llama3.1",
		ollama.WithHTTPClient(&http.Client{Timeout: 100 * time.Millisecond}),
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

func TestOllama_ModelID(t *testing.T) {
	provider := ollama.New("http://localhost:11434", "llama3.1:70b")
	if provider.ModelID() != "llama3.1:70b" {
		t.Errorf("model ID: %s", provider.ModelID())
	}
}

func TestOllama_MetadataPropagation(t *testing.T) {
	var receivedHeaders http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedHeaders = r.Header
		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, `{"model":"llama3.1","message":{"role":"assistant","content":"ok"},"done":true,"eval_count":1,"prompt_eval_count":1}`)
	}))
	defer server.Close()

	provider := ollama.New(server.URL, "llama3.1")

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

func TestOllama_MessageConversion(t *testing.T) {
	var receivedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&receivedBody)
		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, `{"model":"llama3.1","message":{"role":"assistant","content":"ok"},"done":true,"eval_count":1,"prompt_eval_count":1}`)
	}))
	defer server.Close()

	provider := ollama.New(server.URL, "llama3.1")

	temp := 0.5
	stream, err := provider.CreateStream(context.Background(), &agentflow.Request{
		SystemPrompt: "You are helpful.",
		Messages: []agentflow.Message{
			agentflow.NewUserMessage("Hello"),
			agentflow.NewAssistantMessage("Hi there!"),
			agentflow.NewUserMessage("How are you?"),
		},
		MaxTokens:   100,
		Temperature: &temp,
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

	// Verify request structure.
	if receivedBody["model"] != "llama3.1" {
		t.Errorf("model: %v", receivedBody["model"])
	}
	if receivedBody["stream"] != true {
		t.Errorf("stream: %v", receivedBody["stream"])
	}

	messages, ok := receivedBody["messages"].([]any)
	if !ok {
		t.Fatalf("messages not an array: %T", receivedBody["messages"])
	}
	// system + user + assistant + user = 4 messages
	if len(messages) != 4 {
		t.Errorf("expected 4 messages, got %d", len(messages))
	}

	// First message should be system.
	first := messages[0].(map[string]any)
	if first["role"] != "system" {
		t.Errorf("first message role: %v", first["role"])
	}
	if first["content"] != "You are helpful." {
		t.Errorf("first message content: %v", first["content"])
	}

	// Check options.
	opts, ok := receivedBody["options"].(map[string]any)
	if !ok {
		t.Fatal("options not found")
	}
	if opts["num_predict"] != float64(100) {
		t.Errorf("num_predict: %v", opts["num_predict"])
	}
	if opts["temperature"] != 0.5 {
		t.Errorf("temperature: %v", opts["temperature"])
	}
}

func TestOllama_ImageMessage(t *testing.T) {
	var receivedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&receivedBody)
		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, `{"model":"llava","message":{"role":"assistant","content":"A cat"},"done":true,"eval_count":3,"prompt_eval_count":20}`)
	}))
	defer server.Close()

	provider := ollama.New(server.URL, "llava")

	msg := agentflow.NewImageMessage("What is this?",
		agentflow.ImageContent{MediaType: "image/png", Data: "iVBOR"},
	)

	stream, err := provider.CreateStream(context.Background(), &agentflow.Request{
		Messages: []agentflow.Message{msg},
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

	// Verify images are included in the message.
	messages := receivedBody["messages"].([]any)
	userMsg := messages[0].(map[string]any)
	images, ok := userMsg["images"].([]any)
	if !ok || len(images) == 0 {
		t.Fatal("expected images in user message")
	}
	if images[0] != "iVBOR" {
		t.Errorf("image data: %v", images[0])
	}
}

func TestOllama_DocumentFallback(t *testing.T) {
	var receivedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&receivedBody)
		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, `{"model":"llama3.1","message":{"role":"assistant","content":"ok"},"done":true,"eval_count":1,"prompt_eval_count":1}`)
	}))
	defer server.Close()

	provider := ollama.New(server.URL, "llama3.1")

	// text/csv is a text MIME type, so it should be decoded and included as text.
	msg := agentflow.NewDocumentMessage("Analyze this CSV",
		agentflow.DocumentContent{
			Filename:  "data.csv",
			MediaType: "text/csv",
			Data:      "Y29sMSxjb2wy", // base64 of "col1,col2"
		},
	)

	stream, err := provider.CreateStream(context.Background(), &agentflow.Request{
		Messages: []agentflow.Message{msg},
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

	messages := receivedBody["messages"].([]any)
	userMsg := messages[0].(map[string]any)
	content := userMsg["content"].(string)

	if !strings.Contains(content, "col1,col2") {
		t.Errorf("expected decoded CSV content, got: %s", content)
	}
	if !strings.Contains(content, "[Document: data.csv]") {
		t.Errorf("expected document label, got: %s", content)
	}
}

func TestOllama_BinaryDocumentFallback(t *testing.T) {
	var receivedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&receivedBody)
		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, `{"model":"llama3.1","message":{"role":"assistant","content":"ok"},"done":true,"eval_count":1,"prompt_eval_count":1}`)
	}))
	defer server.Close()

	provider := ollama.New(server.URL, "llama3.1")

	// application/pdf is not a text MIME type, so it should show a placeholder.
	msg := agentflow.NewDocumentMessage("Read this PDF",
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
	for {
		_, err := stream.Next()
		if err == io.EOF {
			break
		}
	}

	messages := receivedBody["messages"].([]any)
	userMsg := messages[0].(map[string]any)
	content := userMsg["content"].(string)

	if !strings.Contains(content, "binary, not displayable") {
		t.Errorf("expected binary fallback message, got: %s", content)
	}
	if !strings.Contains(content, "report.pdf") {
		t.Errorf("expected filename in message, got: %s", content)
	}
}

func TestOllama_EmptyResponse(t *testing.T) {
	server := httptest.NewServer(jsonlResponse(
		`{"model":"llama3.1","message":{"role":"assistant","content":""},"done":true,"eval_count":0,"prompt_eval_count":5}`,
	))
	defer server.Close()

	provider := ollama.New(server.URL, "llama3.1")

	stream, err := provider.CreateStream(context.Background(), &agentflow.Request{
		Messages: []agentflow.Message{agentflow.NewUserMessage("Hi")},
	})
	if err != nil {
		t.Fatalf("create stream: %v", err)
	}
	defer stream.Close()

	var events int
	for {
		_, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		events++
	}

	if events != 0 {
		t.Errorf("expected 0 content events for empty response, got %d", events)
	}
}

func TestOllama_HealthCheck(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" && r.Method == http.MethodGet {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"models":[]}`))
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	provider := ollama.New(server.URL, "llama3.1")

	if err := agentflow.IsHealthy(context.Background(), provider); err != nil {
		t.Errorf("expected healthy, got: %v", err)
	}
}

func TestOllama_HealthCheckFail(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	provider := ollama.New(server.URL, "llama3.1")

	if err := agentflow.IsHealthy(context.Background(), provider); err == nil {
		t.Error("expected unhealthy")
	}
}

// --- Integration test ---

func TestIntegration_Ollama(t *testing.T) {
	baseURL := os.Getenv("OLLAMA_BASE_URL")
	if baseURL == "" {
		t.Skip("OLLAMA_BASE_URL not set")
	}
	model := os.Getenv("OLLAMA_MODEL")
	if model == "" {
		model = "llama3.2:1b"
	}

	provider := ollama.New(baseURL, model)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	stream, err := provider.CreateStream(ctx, &agentflow.Request{
		Messages:    []agentflow.Message{agentflow.NewUserMessage("Say 'pong' and nothing else.")},
		MaxTokens:   10,
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

	t.Logf("Ollama response: %q", text)
	if text == "" {
		t.Error("empty response from Ollama")
	}
	if !strings.Contains(strings.ToLower(text), "pong") {
		t.Logf("Note: model may not have followed instructions exactly, but pipeline works")
	}

	usage := stream.Usage()
	if usage != nil {
		t.Logf("Usage: prompt=%d, completion=%d, total=%d", usage.PromptTokens, usage.CompletionTokens, usage.TotalTokens)
	}
}
