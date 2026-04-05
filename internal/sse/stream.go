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
// It handles text deltas, thinking deltas, think tag parsing, and tool call accumulation.
type Stream struct {
	resp    *http.Response
	scanner *bufio.Scanner
	usage   *agentflow.Usage
	done    bool

	// Tool call accumulators keyed by index.
	toolCalls map[int]*ToolCallAccumulator

	// Think tag parser for content field (<think>...</think> inline tags).
	thinkParser thinkTagParser

	// Strips model-internal tags from thinking content (<tool>, <output>).
	thinkingStripper thinkTagStripper

	// Strips model-internal tags from reasoning field (<think>, <tool>, <output>).
	reasoningStripper thinkTagStripper

	// Pending events from a single chunk that produced multiple segments.
	pending []agentflow.StreamEvent
}

// NewStream creates an SSE stream parser from an HTTP response.
func NewStream(resp *http.Response) *Stream {
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 256*1024), 10*1024*1024) // 256KB initial, 10MB max
	return &Stream{
		resp:      resp,
		scanner:   scanner,
		toolCalls: make(map[int]*ToolCallAccumulator),
	}
}

// Next returns the next StreamEvent. Returns io.EOF when the stream is complete.
func (s *Stream) Next() (agentflow.StreamEvent, error) {
	// Drain any pending events from a previous multi-segment chunk.
	if len(s.pending) > 0 {
		ev := s.pending[0]
		s.pending = s.pending[1:]
		return ev, nil
	}

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
			// Strip <think>/<​/think> tags from reasoning content — some models
			// (e.g. Groq compound-beta) include them in the reasoning field.
			// Uses a stateful stripper to handle tags split across chunks.
			text := s.reasoningStripper.strip(*choice.Delta.Reasoning)
			if text != "" {
				return agentflow.StreamEvent{
					Type:          agentflow.StreamEventThinkingDelta,
					ThinkingDelta: &agentflow.ContentDelta{Text: text},
				}, nil
			}
			continue
		}

		if choice.Delta.Content != nil && *choice.Delta.Content != "" {
			segments := s.thinkParser.process(*choice.Delta.Content)
			events := make([]agentflow.StreamEvent, 0, len(segments))
			for _, seg := range segments {
				if seg.text == "" {
					continue
				}
				if seg.thinking {
					// Strip model-internal tags (<tool>, <output>) from thinking content.
					cleaned := s.thinkingStripper.strip(seg.text)
					if cleaned == "" {
						continue
					}
					events = append(events, agentflow.StreamEvent{
						Type:          agentflow.StreamEventThinkingDelta,
						ThinkingDelta: &agentflow.ContentDelta{Text: cleaned},
					})
				} else {
					events = append(events, agentflow.StreamEvent{
						Type:  agentflow.StreamEventDelta,
						Delta: &agentflow.ContentDelta{Text: seg.text},
					})
				}
			}
			if len(events) == 0 {
				continue
			}
			if len(events) > 1 {
				s.pending = events[1:]
			}
			return events[0], nil
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

// reasoningTagStripper is the list of tags to strip from reasoning field content.
// These are model-internal formatting tags that should not reach the client.
var reasoningStripTags = []string{
	"<think>", "</think>",
	"<tool>", "</tool>",
	"<output>", "</output>",
}

// thinkTagStripper removes model-internal tags (<think>, <tool>, <output> and
// their closing counterparts) from streaming text, handling tags split across
// chunk boundaries. Used for the reasoning field where content is already
// classified as thinking but may contain raw formatting tags.
type thinkTagStripper struct {
	tagBuf string
}

// strip removes model-internal tags from text, returning the cleaned text.
func (s *thinkTagStripper) strip(text string) string {
	if s.tagBuf != "" {
		text = s.tagBuf + text
		s.tagBuf = ""
	}

	var result strings.Builder
	for len(text) > 0 {
		// Find the earliest tag match.
		bestIdx := -1
		bestTag := ""
		for _, tag := range reasoningStripTags {
			if idx := strings.Index(text, tag); idx >= 0 {
				if bestIdx < 0 || idx < bestIdx {
					bestIdx = idx
					bestTag = tag
				}
			}
		}
		if bestIdx >= 0 {
			result.WriteString(text[:bestIdx])
			text = text[bestIdx+len(bestTag):]
			continue
		}

		// Check for partial tag at end (any of the strip tags).
		if partial := matchPartialSuffixAny(text, reasoningStripTags); partial > 0 {
			result.WriteString(text[:len(text)-partial])
			s.tagBuf = text[len(text)-partial:]
			return result.String()
		}

		// No tags found.
		result.WriteString(text)
		break
	}
	return result.String()
}

// matchPartialSuffixAny checks if the end of text matches a prefix of any tag.
// Returns the longest partial match length.
func matchPartialSuffixAny(text string, tags []string) int {
	best := 0
	for _, tag := range tags {
		if n := matchPartialSuffix(text, tag); n > best {
			best = n
		}
	}
	return best
}

// --- Think tag parser ---

// thinkTagParser is a stateful parser that detects <think>...</think> tags in
// streaming text content. Tags may be split across chunk boundaries.
//
// State machine: NORMAL → <think> → THINKING → </think> → NORMAL
type thinkTagParser struct {
	thinking bool   // true when inside <think>...</think>
	tagBuf   string // partial tag buffer for boundary-split tags
}

// textSegment is a piece of text with its thinking/normal classification.
type textSegment struct {
	text     string
	thinking bool
}

// process takes a chunk of text and returns classified segments.
// It handles <think> and </think> tags, including when they are split across chunks.
func (p *thinkTagParser) process(text string) []textSegment {
	var segments []textSegment

	// Prepend any buffered partial tag from previous chunk.
	if p.tagBuf != "" {
		text = p.tagBuf + text
		p.tagBuf = ""
	}

	for len(text) > 0 {
		if p.thinking {
			// Looking for </think>
			idx := strings.Index(text, "</think>")
			if idx >= 0 {
				// Found closing tag.
				if idx > 0 {
					segments = append(segments, textSegment{text: text[:idx], thinking: true})
				}
				p.thinking = false
				text = text[idx+len("</think>"):]
				continue
			}
			// Check if text ends with a partial </think> tag.
			if partial := matchPartialSuffix(text, "</think>"); partial > 0 {
				if len(text)-partial > 0 {
					segments = append(segments, textSegment{text: text[:len(text)-partial], thinking: true})
				}
				p.tagBuf = text[len(text)-partial:]
				return segments
			}
			// No tag found, all text is thinking.
			segments = append(segments, textSegment{text: text, thinking: true})
			return segments
		}

		// NORMAL state: looking for <think>
		idx := strings.Index(text, "<think>")
		if idx >= 0 {
			// Found opening tag.
			if idx > 0 {
				segments = append(segments, textSegment{text: text[:idx], thinking: false})
			}
			p.thinking = true
			text = text[idx+len("<think>"):]
			continue
		}
		// Check if text ends with a partial <think> tag.
		if partial := matchPartialSuffix(text, "<think>"); partial > 0 {
			if len(text)-partial > 0 {
				segments = append(segments, textSegment{text: text[:len(text)-partial], thinking: false})
			}
			p.tagBuf = text[len(text)-partial:]
			return segments
		}
		// No tag found, all text is normal.
		segments = append(segments, textSegment{text: text, thinking: false})
		return segments
	}

	return segments
}

// matchPartialSuffix checks if the end of text matches a prefix of tag.
// Returns the length of the partial match (0 if no match).
// For example, text="hello<thi" and tag="<think>" returns 4 ("<thi").
func matchPartialSuffix(text, tag string) int {
	maxLen := len(tag) - 1
	if maxLen > len(text) {
		maxLen = len(text)
	}
	for i := maxLen; i > 0; i-- {
		if strings.HasSuffix(text, tag[:i]) {
			return i
		}
	}
	return 0
}
