// Streaming example: production-ready HTTP SSE endpoint for an agentic AI.
//
// Endpoints:
//   POST /chat          — send a message, receive SSE stream
//   POST /chat/sync     — send a message, receive JSON response
//
// Usage:
//   GROQ_API_KEY=gsk-... go run ./_examples/streaming/
//   curl -X POST http://localhost:8080/chat -d '{"message":"Search for Go 1.23 features"}' -H 'Content-Type: application/json'
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/groq"
	"github.com/CanArslanDev/agentflow/tools/builtin"
)

type chatRequest struct {
	Message      string `json:"message"`
	SystemPrompt string `json:"system_prompt,omitempty"`
}

type chatResponse struct {
	Response  string `json:"response"`
	ToolCalls int    `json:"tool_calls"`
	Turns     int    `json:"turns"`
}

func main() {
	apiKey := os.Getenv("GROQ_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "GROQ_API_KEY environment variable required")
		os.Exit(1)
	}

	provider := groq.New(apiKey, "llama-3.3-70b-versatile")

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(builtin.Remote()...),
		agentflow.WithExecutionMode(agentflow.ModeRemote),
		agentflow.WithSystemPrompt("You are a helpful assistant. Use web_search for current information and http_request for APIs."),
		agentflow.WithMaxTurns(10),
		agentflow.WithMaxTokens(2048),
		agentflow.WithMaxResultSize(5000),
		agentflow.WithTokenBudget(agentflow.TokenBudget{
			MaxTokens:        50000,
			WarningThreshold: 0.8,
		}),
	)

	mux := http.NewServeMux()

	mux.HandleFunc("/chat", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}

		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
			return
		}
		if req.Message == "" {
			http.Error(w, "message is required", http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming not supported", http.StatusInternalServerError)
			return
		}

		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
		defer cancel()

		messages := []agentflow.Message{agentflow.NewUserMessage(req.Message)}

		for ev := range agent.Run(ctx, messages) {
			data := marshalSSE(ev)
			if data == "" {
				continue
			}
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}
	})

	mux.HandleFunc("/chat/sync", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}

		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
			return
		}
		if req.Message == "" {
			http.Error(w, "message is required", http.StatusBadRequest)
			return
		}

		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
		defer cancel()

		messages := []agentflow.Message{agentflow.NewUserMessage(req.Message)}

		var text string
		var toolCalls, turns int
		for ev := range agent.Run(ctx, messages) {
			switch ev.Type {
			case agentflow.EventTextDelta:
				text += ev.TextDelta.Text
			case agentflow.EventToolStart:
				toolCalls++
			case agentflow.EventTurnEnd:
				turns = ev.TurnEnd.TurnNumber
			}
		}

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		json.NewEncoder(w).Encode(chatResponse{
			Response:  text,
			ToolCalls: toolCalls,
			Turns:     turns,
		})
	})

	// CORS preflight handler.
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodOptions {
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
			w.WriteHeader(http.StatusNoContent)
			return
		}
		http.NotFound(w, r)
	})

	fmt.Println("agentflow HTTP server")
	fmt.Println("  POST /chat       — SSE streaming response")
	fmt.Println("  POST /chat/sync  — JSON response")
	fmt.Println("  Listening on :8080")
	http.ListenAndServe(":8080", mux)
}

func marshalSSE(ev agentflow.Event) string {
	var payload map[string]any

	switch ev.Type {
	case agentflow.EventTextDelta:
		payload = map[string]any{"type": "text_delta", "text": ev.TextDelta.Text}
	case agentflow.EventToolStart:
		payload = map[string]any{"type": "tool_start", "tool": ev.ToolStart.ToolCall.Name, "call_id": ev.ToolStart.ToolCall.ID}
	case agentflow.EventToolProgress:
		payload = map[string]any{"type": "tool_progress", "message": ev.ToolProgress.Message}
	case agentflow.EventToolEnd:
		payload = map[string]any{"type": "tool_end", "tool": ev.ToolEnd.ToolCall.Name, "is_error": ev.ToolEnd.Result.IsError, "duration_ms": ev.ToolEnd.Duration.Milliseconds()}
	case agentflow.EventTurnStart:
		payload = map[string]any{"type": "turn_start", "turn": ev.TurnStart.TurnNumber}
	case agentflow.EventTurnEnd:
		payload = map[string]any{"type": "turn_end", "reason": string(ev.TurnEnd.Reason), "turn": ev.TurnEnd.TurnNumber}
	case agentflow.EventUsage:
		payload = map[string]any{"type": "usage", "prompt_tokens": ev.Usage.Usage.PromptTokens, "completion_tokens": ev.Usage.Usage.CompletionTokens}
	case agentflow.EventBudgetWarning:
		payload = map[string]any{"type": "budget_warning", "consumed": ev.BudgetWarning.ConsumedTokens, "max": ev.BudgetWarning.MaxTokens}
	case agentflow.EventError:
		payload = map[string]any{"type": "error", "message": ev.Error.Err.Error(), "retrying": ev.Error.Retrying}
	default:
		return ""
	}

	b, _ := json.Marshal(payload)
	return string(b)
}
