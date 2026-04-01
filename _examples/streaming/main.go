// Streaming example: expose an agent as an HTTP SSE endpoint.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/openrouter"
	"github.com/CanArslanDev/agentflow/tools"
)

func main() {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "OPENROUTER_API_KEY environment variable required")
		os.Exit(1)
	}

	provider := openrouter.New(apiKey, "anthropic/claude-sonnet-4-20250514")

	searchTool := tools.New("web_search", "Search the web for current information.").
		WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{
					"type":        "string",
					"description": "The search query",
				},
			},
			"required": []string{"query"},
		}).
		ReadOnly(true).
		ConcurrencySafe(true).
		WithExecute(func(_ context.Context, input json.RawMessage, progress agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			var params struct {
				Query string `json:"query"`
			}
			json.Unmarshal(input, &params)

			if progress != nil {
				progress(agentflow.ProgressEvent{Message: "Searching: " + params.Query})
			}

			// Demo result — replace with actual search API.
			return &agentflow.ToolResult{
				Content: fmt.Sprintf("Search results for '%s':\n1. Example result 1\n2. Example result 2", params.Query),
			}, nil
		}).
		Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(searchTool),
		agentflow.WithSystemPrompt("You are a helpful assistant with web search capability."),
		agentflow.WithMaxTurns(5),
	)

	http.HandleFunc("/chat", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			Message string `json:"message"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		// Set SSE headers.
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming not supported", http.StatusInternalServerError)
			return
		}

		messages := []agentflow.Message{
			agentflow.NewUserMessage(req.Message),
		}

		ctx := r.Context()
		for ev := range agent.Run(ctx, messages) {
			data := marshalSSEEvent(ev)
			if data == "" {
				continue
			}
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}
	})

	fmt.Println("Server listening on :8080")
	fmt.Println("POST /chat with {\"message\": \"...\"}")
	http.ListenAndServe(":8080", nil)
}

// marshalSSEEvent converts an agentflow Event into a JSON string for SSE.
func marshalSSEEvent(ev agentflow.Event) string {
	var payload map[string]any

	switch ev.Type {
	case agentflow.EventTextDelta:
		payload = map[string]any{
			"type": "text_delta",
			"text": ev.TextDelta.Text,
		}
	case agentflow.EventToolStart:
		payload = map[string]any{
			"type":      "tool_start",
			"tool_name": ev.ToolStart.ToolCall.Name,
			"call_id":   ev.ToolStart.ToolCall.ID,
		}
	case agentflow.EventToolProgress:
		payload = map[string]any{
			"type":    "tool_progress",
			"message": ev.ToolProgress.Message,
		}
	case agentflow.EventToolEnd:
		payload = map[string]any{
			"type":      "tool_end",
			"tool_name": ev.ToolEnd.ToolCall.Name,
			"is_error":  ev.ToolEnd.Result.IsError,
			"duration":  ev.ToolEnd.Duration.String(),
		}
	case agentflow.EventTurnEnd:
		payload = map[string]any{
			"type":   "turn_end",
			"reason": string(ev.TurnEnd.Reason),
			"turn":   ev.TurnEnd.TurnNumber,
		}
	case agentflow.EventError:
		payload = map[string]any{
			"type":     "error",
			"message":  ev.Error.Err.Error(),
			"retrying": ev.Error.Retrying,
		}
	default:
		return ""
	}

	b, _ := json.Marshal(payload)
	return string(b)
}
