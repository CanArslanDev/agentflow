package ollama

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/CanArslanDev/agentflow"
)

// ollamaStream parses Ollama's JSONL streaming format into agentflow StreamEvents.
// Unlike OpenAI SSE, Ollama writes one JSON object per line (no "data: " prefix,
// no "[DONE]" marker). The stream ends when a response with Done=true is received.
type ollamaStream struct {
	resp         *http.Response
	scanner      *bufio.Scanner
	usage        *agentflow.Usage
	done         bool
	pendingCalls []responseToolCall // buffered tool calls for multi-call responses
	pendingIdx   int               // next pending call to emit
}

func newStream(resp *http.Response) *ollamaStream {
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 256*1024), 10*1024*1024)
	return &ollamaStream{
		resp:    resp,
		scanner: scanner,
	}
}

// Next returns the next StreamEvent from the Ollama JSONL stream.
// Returns io.EOF when the stream is complete.
func (s *ollamaStream) Next() (agentflow.StreamEvent, error) {
	// Drain any pending tool calls from a previous multi-call response.
	if s.pendingIdx < len(s.pendingCalls) {
		ev := s.buildToolCallEvent(s.pendingCalls, s.pendingIdx)
		s.pendingIdx++
		if s.pendingIdx >= len(s.pendingCalls) {
			s.done = true
		}
		return ev, nil
	}

	for {
		if s.done {
			return agentflow.StreamEvent{}, io.EOF
		}

		if !s.scanner.Scan() {
			if err := s.scanner.Err(); err != nil {
				return agentflow.StreamEvent{}, err
			}
			return agentflow.StreamEvent{}, io.EOF
		}

		line := strings.TrimSpace(s.scanner.Text())
		if line == "" {
			continue
		}

		var resp ollamaResponse
		if err := json.Unmarshal([]byte(line), &resp); err != nil {
			continue
		}

		// Handle tool calls (arrive in the final message).
		if len(resp.Message.ToolCalls) > 0 {
			s.done = true
			// Record usage from the final response.
			if resp.PromptEvalCount > 0 || resp.EvalCount > 0 {
				s.usage = &agentflow.Usage{
					PromptTokens:     resp.PromptEvalCount,
					CompletionTokens: resp.EvalCount,
					TotalTokens:      resp.PromptEvalCount + resp.EvalCount,
				}
			}
			// Store all tool calls; emit the first now, rest via pending buffer.
			s.pendingCalls = resp.Message.ToolCalls
			s.pendingIdx = 1 // first one returned immediately below
			if len(resp.Message.ToolCalls) <= 1 {
				s.done = true
			}
			return s.buildToolCallEvent(resp.Message.ToolCalls, 0), nil
		}

		// Handle the final done=true message (no tool calls).
		if resp.Done {
			s.done = true
			if resp.PromptEvalCount > 0 || resp.EvalCount > 0 {
				s.usage = &agentflow.Usage{
					PromptTokens:     resp.PromptEvalCount,
					CompletionTokens: resp.EvalCount,
					TotalTokens:      resp.PromptEvalCount + resp.EvalCount,
				}
			}
			// If the final message has content, emit it before EOF.
			if resp.Message.Content != "" {
				return agentflow.StreamEvent{
					Type:  agentflow.StreamEventDelta,
					Delta: &agentflow.ContentDelta{Text: resp.Message.Content},
				}, nil
			}
			return agentflow.StreamEvent{}, io.EOF
		}

		// Regular content delta.
		if resp.Message.Content != "" {
			return agentflow.StreamEvent{
				Type:  agentflow.StreamEventDelta,
				Delta: &agentflow.ContentDelta{Text: resp.Message.Content},
			}, nil
		}
	}
}

func (s *ollamaStream) buildToolCallEvent(calls []responseToolCall, index int) agentflow.StreamEvent {
	tc := calls[index]
	argsJSON, _ := json.Marshal(tc.Function.Arguments)
	if !json.Valid(argsJSON) {
		argsJSON = json.RawMessage("{}")
	}
	return agentflow.StreamEvent{
		Type: agentflow.StreamEventToolCall,
		ToolCall: &agentflow.ToolCall{
			ID:    fmt.Sprintf("ollama-call-%d", index),
			Name:  tc.Function.Name,
			Input: argsJSON,
		},
	}
}

// Close releases the HTTP response body.
func (s *ollamaStream) Close() error {
	if s.resp != nil && s.resp.Body != nil {
		return s.resp.Body.Close()
	}
	return nil
}

// Usage returns token usage statistics collected during streaming.
func (s *ollamaStream) Usage() *agentflow.Usage {
	return s.usage
}

// --- Response types ---

type ollamaResponse struct {
	Model     string          `json:"model"`
	CreatedAt string          `json:"created_at"`
	Message   responseMessage `json:"message"`
	Done      bool            `json:"done"`
	DoneReason string         `json:"done_reason"`

	// Token counts (only present in the final message where done=true).
	EvalCount       int `json:"eval_count"`        // completion tokens
	PromptEvalCount int `json:"prompt_eval_count"` // prompt tokens
}

type responseMessage struct {
	Role      string             `json:"role"`
	Content   string             `json:"content"`
	ToolCalls []responseToolCall `json:"tool_calls,omitempty"`
}

type responseToolCall struct {
	Function responseFunctionCall `json:"function"`
}

type responseFunctionCall struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}
