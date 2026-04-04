package compactor

import (
	"context"
	"fmt"

	"github.com/CanArslanDev/agentflow"
)

// SlidingWindowCompactor keeps the most recent N messages and discards older ones.
// The first message (typically the initial user prompt) is always preserved to
// maintain task context. A system summary of discarded messages is inserted
// after the first message.
//
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithCompactor(compactor.NewSlidingWindow(20, 0)),
//	)
type SlidingWindowCompactor struct {
	// keepLast is the number of recent messages to retain.
	keepLast int

	// triggerAt is the message count that triggers compaction.
	// Zero means triggerAt = keepLast * 2.
	triggerAt int
}

// NewSlidingWindow creates a compactor that retains the last keepLast
// messages. Compaction triggers when the history exceeds triggerAt messages.
// If triggerAt is 0, it defaults to keepLast * 2.
func NewSlidingWindow(keepLast, triggerAt int) *SlidingWindowCompactor {
	if keepLast < 2 {
		keepLast = 2
	}
	if triggerAt <= 0 {
		triggerAt = keepLast * 2
	}
	return &SlidingWindowCompactor{
		keepLast:  keepLast,
		triggerAt: triggerAt,
	}
}

// ShouldCompact returns true when the message count exceeds the trigger threshold.
func (c *SlidingWindowCompactor) ShouldCompact(messages []agentflow.Message, _ *agentflow.Usage) bool {
	return len(messages) > c.triggerAt
}

// Compact keeps the first message (initial user prompt) and the last keepLast
// messages, inserting a system note about discarded context in between.
func (c *SlidingWindowCompactor) Compact(_ context.Context, messages []agentflow.Message) ([]agentflow.Message, error) {
	if len(messages) <= c.keepLast+1 {
		return messages, nil
	}

	discarded := len(messages) - c.keepLast - 1 // -1 for the preserved first message
	if discarded <= 0 {
		return messages, nil
	}

	result := make([]agentflow.Message, 0, c.keepLast+2) // first + note + keepLast

	// Preserve the first message (initial user prompt / task description).
	result = append(result, messages[0])

	// Insert a system note about the compaction.
	result = append(result, NewCompactionNotice(discarded, len(messages)))

	// Append the most recent messages.
	recent := messages[len(messages)-c.keepLast:]
	result = append(result, recent...)

	return result, nil
}

// TokenWindowCompactor triggers compaction based on estimated token usage rather
// than message count. This is more accurate for conversations with variable
// message sizes (large tool results vs short text).
type TokenWindowCompactor struct {
	// maxTokens is the estimated token threshold that triggers compaction.
	maxTokens int

	// keepLast is the number of recent messages to retain.
	keepLast int

	// charsPerToken is the estimated characters per token for size estimation.
	// Default: 4.
	charsPerToken int
}

// NewTokenWindow creates a compactor that triggers when the estimated
// token count of the conversation exceeds maxTokens. It retains the last
// keepLast messages after compaction.
func NewTokenWindow(maxTokens, keepLast int) *TokenWindowCompactor {
	if keepLast < 2 {
		keepLast = 2
	}
	return &TokenWindowCompactor{
		maxTokens:     maxTokens,
		keepLast:      keepLast,
		charsPerToken: 4,
	}
}

// ShouldCompact estimates the token count from message content and compares
// against the threshold. Uses the last known usage if available for calibration.
func (c *TokenWindowCompactor) ShouldCompact(messages []agentflow.Message, usage *agentflow.Usage) bool {
	// If we have real usage data, use it directly.
	if usage != nil && usage.PromptTokens > 0 {
		return usage.PromptTokens > c.maxTokens
	}

	// Estimate from content length.
	totalChars := estimateChars(messages)
	estimated := totalChars / c.charsPerToken
	return estimated > c.maxTokens
}

// Compact uses the same sliding window strategy as SlidingWindowCompactor.
func (c *TokenWindowCompactor) Compact(ctx context.Context, messages []agentflow.Message) ([]agentflow.Message, error) {
	sw := &SlidingWindowCompactor{keepLast: c.keepLast}
	return sw.Compact(ctx, messages)
}

// NewCompactionNotice creates a system message indicating that older context
// was compacted. This helps the model understand there may be missing context.
func NewCompactionNotice(discardedCount, originalCount int) agentflow.Message {
	return agentflow.Message{
		Role: agentflow.RoleSystem,
		Content: []agentflow.ContentBlock{{
			Type: agentflow.ContentText,
			Text: fmt.Sprintf("[Context compacted: %d of %d earlier messages were removed to fit context limits. "+
				"The conversation continues from the most recent messages below.]",
				discardedCount, originalCount),
		}},
	}
}

// estimateChars counts the total characters in message content.
func estimateChars(messages []agentflow.Message) int {
	totalChars := 0
	for _, msg := range messages {
		for _, block := range msg.Content {
			switch block.Type {
			case agentflow.ContentText:
				totalChars += len(block.Text)
			case agentflow.ContentToolResult:
				if block.ToolResult != nil {
					totalChars += len(block.ToolResult.Content)
				}
			}
		}
	}
	return totalChars
}
