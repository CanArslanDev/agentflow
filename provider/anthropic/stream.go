package anthropic

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/CanArslanDev/agentflow"
)

// anthropicStream parses Anthropic's SSE format into agentflow StreamEvents.
// Anthropic uses a different SSE event structure than OpenAI:
//   - event: message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop
//   - Tool calls arrive as content_block_start (type: tool_use) followed by
//     content_block_delta (type: input_json_delta) chunks.
//   - Thinking blocks arrive as content_block_start (type: thinking) followed by
//     content_block_delta (type: thinking_delta) chunks.
type anthropicStream struct {
	resp    *http.Response
	scanner *bufio.Scanner
	usage   *agentflow.Usage
	done    bool

	// Tool call accumulation.
	currentToolID   string
	currentToolName string
	currentToolJSON string

	// Current content block type tracking.
	currentBlockType string
}

func newAnthropicStream(resp *http.Response) *anthropicStream {
	return &anthropicStream{
		resp:    resp,
		scanner: bufio.NewScanner(resp.Body),
	}
}

func (s *anthropicStream) Next() (agentflow.StreamEvent, error) {
	var eventType string

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

		line := s.scanner.Text()

		// Parse SSE event type.
		if strings.HasPrefix(line, "event: ") {
			eventType = strings.TrimPrefix(line, "event: ")
			continue
		}

		// Parse SSE data.
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		switch eventType {
		case "message_start":
			var ev sseEvent
			if err := json.Unmarshal([]byte(data), &ev); err != nil {
				continue
			}
			if ev.Message != nil && ev.Message.Usage != nil {
				s.usage = &agentflow.Usage{
					PromptTokens: ev.Message.Usage.InputTokens,
				}
			}
			continue

		case "content_block_start":
			var ev sseEvent
			if err := json.Unmarshal([]byte(data), &ev); err != nil {
				continue
			}
			if ev.ContentBlock != nil {
				s.currentBlockType = ev.ContentBlock.Type
				if ev.ContentBlock.Type == "tool_use" {
					s.currentToolID = ev.ContentBlock.ID
					s.currentToolName = ev.ContentBlock.Name
					s.currentToolJSON = ""
				}
			}
			continue

		case "content_block_delta":
			var ev sseEvent
			if err := json.Unmarshal([]byte(data), &ev); err != nil {
				continue
			}
			if ev.Delta == nil {
				continue
			}

			switch ev.Delta.Type {
			case "text_delta":
				if ev.Delta.Text != "" {
					return agentflow.StreamEvent{
						Type:  agentflow.StreamEventDelta,
						Delta: &agentflow.ContentDelta{Text: ev.Delta.Text},
					}, nil
				}
			case "thinking_delta":
				if ev.Delta.Thinking != "" {
					return agentflow.StreamEvent{
						Type:          agentflow.StreamEventThinkingDelta,
						ThinkingDelta: &agentflow.ContentDelta{Text: ev.Delta.Thinking},
					}, nil
				}
			case "input_json_delta":
				s.currentToolJSON += ev.Delta.PartialJSON
			}
			continue

		case "content_block_stop":
			blockType := s.currentBlockType
			s.currentBlockType = ""

			// If we were accumulating a tool call, emit it now.
			if blockType == "tool_use" && s.currentToolID != "" {
				toolJSON := s.currentToolJSON
				if toolJSON == "" || !json.Valid([]byte(toolJSON)) {
					toolJSON = "{}"
				}
				ev := agentflow.StreamEvent{
					Type: agentflow.StreamEventToolCall,
					ToolCall: &agentflow.ToolCall{
						ID:    s.currentToolID,
						Name:  s.currentToolName,
						Input: json.RawMessage(toolJSON),
					},
				}
				s.currentToolID = ""
				s.currentToolName = ""
				s.currentToolJSON = ""
				return ev, nil
			}
			continue

		case "message_delta":
			var ev sseEvent
			if err := json.Unmarshal([]byte(data), &ev); err != nil {
				continue
			}
			if ev.Usage != nil {
				if s.usage == nil {
					s.usage = &agentflow.Usage{}
				}
				s.usage.CompletionTokens = ev.Usage.OutputTokens
				s.usage.TotalTokens = s.usage.PromptTokens + s.usage.CompletionTokens
			}
			continue

		case "message_stop":
			s.done = true
			return agentflow.StreamEvent{}, io.EOF

		case "error":
			var errResp struct {
				Error struct {
					Message string `json:"message"`
				} `json:"error"`
			}
			json.Unmarshal([]byte(data), &errResp)
			return agentflow.StreamEvent{
				Type:  agentflow.StreamEventError,
				Error: &agentflow.ProviderError{Message: errResp.Error.Message},
			}, nil
		}
	}
}

func (s *anthropicStream) Close() error {
	if s.resp != nil && s.resp.Body != nil {
		return s.resp.Body.Close()
	}
	return nil
}

func (s *anthropicStream) Usage() *agentflow.Usage {
	return s.usage
}
