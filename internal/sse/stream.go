package sse

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/CanArslanDev/agentflow"
)

// Stream parses an OpenAI-compatible SSE response into agentflow StreamEvents.
// It handles text deltas, thinking deltas, and tool call accumulation.
type Stream struct {
	resp    *http.Response
	scanner *bufio.Scanner
	usage   *agentflow.Usage
	done    bool

	// Tool call accumulators keyed by index.
	toolCalls map[int]*ToolCallAccumulator
}

// NewStream creates an SSE stream parser from an HTTP response.
func NewStream(resp *http.Response) *Stream {
	return &Stream{
		resp:      resp,
		scanner:   bufio.NewScanner(resp.Body),
		toolCalls: make(map[int]*ToolCallAccumulator),
	}
}

// Next returns the next StreamEvent. Returns io.EOF when the stream is complete.
func (s *Stream) Next() (agentflow.StreamEvent, error) {
	for {
		if s.done {
			return agentflow.StreamEvent{}, io.EOF
		}

		if !s.scanner.Scan() {
			if events := s.flushToolCalls(); len(events) > 0 {
				s.done = true
				return events[0], nil
			}
			if err := s.scanner.Err(); err != nil {
				return agentflow.StreamEvent{}, err
			}
			return agentflow.StreamEvent{}, io.EOF
		}

		line := s.scanner.Text()

		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		if data == "[DONE]" {
			if events := s.flushToolCalls(); len(events) > 0 {
				s.done = true
				return events[0], nil
			}
			s.done = true
			return agentflow.StreamEvent{}, io.EOF
		}

		var chunk StreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		if chunk.Usage != nil {
			s.usage = &agentflow.Usage{
				PromptTokens:     chunk.Usage.PromptTokens,
				CompletionTokens: chunk.Usage.CompletionTokens,
				TotalTokens:      chunk.Usage.TotalTokens,
			}
		}

		if len(chunk.Choices) == 0 {
			continue
		}
		choice := chunk.Choices[0]

		if choice.FinishReason != nil && *choice.FinishReason == "tool_calls" {
			if events := s.flushToolCalls(); len(events) > 0 {
				return events[0], nil
			}
		}

		if choice.Delta.Reasoning != nil && *choice.Delta.Reasoning != "" {
			return agentflow.StreamEvent{
				Type:          agentflow.StreamEventThinkingDelta,
				ThinkingDelta: &agentflow.ContentDelta{Text: *choice.Delta.Reasoning},
			}, nil
		}

		if choice.Delta.Content != nil && *choice.Delta.Content != "" {
			return agentflow.StreamEvent{
				Type:  agentflow.StreamEventDelta,
				Delta: &agentflow.ContentDelta{Text: *choice.Delta.Content},
			}, nil
		}

		if len(choice.Delta.ToolCalls) > 0 {
			for _, tc := range choice.Delta.ToolCalls {
				acc, ok := s.toolCalls[tc.Index]
				if !ok {
					acc = &ToolCallAccumulator{}
					s.toolCalls[tc.Index] = acc
				}
				if tc.ID != "" {
					acc.ID = tc.ID
				}
				if tc.Function.Name != "" {
					acc.Name = tc.Function.Name
				}
				acc.Arguments += tc.Function.Arguments
			}
			continue
		}
	}
}

// flushToolCalls converts accumulated tool call deltas into complete StreamEvents.
func (s *Stream) flushToolCalls() []agentflow.StreamEvent {
	if len(s.toolCalls) == 0 {
		return nil
	}

	events := make([]agentflow.StreamEvent, 0, len(s.toolCalls))
	for _, acc := range s.toolCalls {
		events = append(events, agentflow.StreamEvent{
			Type: agentflow.StreamEventToolCall,
			ToolCall: &agentflow.ToolCall{
				ID:    acc.ID,
				Name:  acc.Name,
				Input: acc.ToJSON(),
			},
		})
	}
	s.toolCalls = make(map[int]*ToolCallAccumulator)
	return events
}

// Close releases the HTTP response body.
func (s *Stream) Close() error {
	if s.resp != nil && s.resp.Body != nil {
		return s.resp.Body.Close()
	}
	return nil
}

// Usage returns token usage statistics collected during streaming.
func (s *Stream) Usage() *agentflow.Usage {
	return s.usage
}
