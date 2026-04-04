package anthropic_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/anthropic"
)

func anthropicSSE(events ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for _, ev := range events {
			fmt.Fprint(w, ev)
		}
	}
}

func TestAnthropic_SimpleTextResponse(t *testing.T) {
	server := httptest.NewServer(anthropicSSE(
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":15}}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\" World\"}}\n\n",
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":5}}\n\n",
		"event: message_stop\ndata: {}\n\n",
	))
	defer server.Close()

	provider := anthropic.New("test-key", "claude-sonnet-4-20250514", anthropic.WithBaseURL(server.URL))

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
	if usage.PromptTokens != 15 {
		t.Errorf("prompt tokens: %d", usage.PromptTokens)
	}
	if usage.CompletionTokens != 5 {
		t.Errorf("completion tokens: %d", usage.CompletionTokens)
	}
}

func TestAnthropic_ToolCallResponse(t *testing.T) {
	server := httptest.NewServer(anthropicSSE(
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":20}}}\n\n",
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"NYC\\\"}\"}}\n\n",
		"event: content_block_stop\ndata: {}\n\n",
		"event: message_stop\ndata: {}\n\n",
	))
	defer server.Close()

	provider := anthropic.New("test-key", "claude-sonnet-4-20250514", anthropic.WithBaseURL(server.URL))

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
	if toolCall.ID != "toolu_1" {
		t.Errorf("tool ID: %s", toolCall.ID)
	}
	if string(toolCall.Input) != `{"city":"NYC"}` {
		t.Errorf("tool input: %s", string(toolCall.Input))
	}
}

func TestAnthropic_ErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"error":{"message":"invalid api key"}}`))
	}))
	defer server.Close()

	provider := anthropic.New("bad-key", "claude-sonnet-4-20250514", anthropic.WithBaseURL(server.URL))

	_, err := provider.CreateStream(context.Background(), &agentflow.Request{
		Messages: []agentflow.Message{agentflow.NewUserMessage("Hi")},
	})
	if err == nil {
		t.Fatal("expected error for 401")
	}

	// 401 should NOT be retryable.
	if agentflow.IsRetryableError(err) {
		t.Error("401 should not be retryable")
	}
}

func TestAnthropic_DocumentMessage(t *testing.T) {
	var receivedBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w,
			"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10}}}\n\n",
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Summary here\"}}\n\n",
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":3}}\n\n",
			"event: message_stop\ndata: {}\n\n",
		)
	}))
	defer server.Close()

	provider := anthropic.New("test-key", "claude-sonnet-4-20250514", anthropic.WithBaseURL(server.URL))

	msg := agentflow.NewDocumentMessage("Summarize this PDF",
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
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
	}

	bodyStr := string(receivedBody)
	// Verify document block is present in the request.
	if !contains(bodyStr, `"type":"document"`) {
		t.Errorf("request body should contain document type, got: %s", bodyStr)
	}
	if !contains(bodyStr, `"media_type":"application/pdf"`) {
		t.Errorf("request body should contain media_type, got: %s", bodyStr)
	}
	if !contains(bodyStr, `"data":"JVBERi0xLjQ="`) {
		t.Errorf("request body should contain base64 data, got: %s", bodyStr)
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && searchString(s, substr)
}

func searchString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestAnthropic_ModelID(t *testing.T) {
	provider := anthropic.New("key", "claude-sonnet-4-20250514")
	if provider.ModelID() != "claude-sonnet-4-20250514" {
		t.Errorf("model ID: %s", provider.ModelID())
	}
}
