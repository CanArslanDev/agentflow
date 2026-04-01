package agentflow

import "context"

// SlidingWindowCompactor keeps the most recent N messages and discards older ones.
// The first message (typically the initial user prompt) is always preserved to
// maintain task context. A system summary of discarded messages is inserted
// after the first message.
//
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithCompactor(agentflow.NewSlidingWindowCompactor(20, 0)),
//	)
type SlidingWindowCompactor struct {
	// keepLast is the number of recent messages to retain.
	keepLast int

	// triggerAt is the message count that triggers compaction.
	// Zero means triggerAt = keepLast * 2.
	triggerAt int
}

// NewSlidingWindowCompactor creates a compactor that retains the last keepLast
// messages. Compaction triggers when the history exceeds triggerAt messages.
// If triggerAt is 0, it defaults to keepLast * 2.
func NewSlidingWindowCompactor(keepLast, triggerAt int) *SlidingWindowCompactor {
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
func (c *SlidingWindowCompactor) ShouldCompact(messages []Message, _ *Usage) bool {
	return len(messages) > c.triggerAt
}

// Compact keeps the first message (initial user prompt) and the last keepLast
// messages, inserting a system note about discarded context in between.
func (c *SlidingWindowCompactor) Compact(_ context.Context, messages []Message) ([]Message, error) {
	if len(messages) <= c.keepLast+1 {
		return messages, nil
	}

	discarded := len(messages) - c.keepLast - 1 // -1 for the preserved first message
	if discarded <= 0 {
		return messages, nil
	}

	result := make([]Message, 0, c.keepLast+2) // first + note + keepLast

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

// NewTokenWindowCompactor creates a compactor that triggers when the estimated
// token count of the conversation exceeds maxTokens. It retains the last
// keepLast messages after compaction.
func NewTokenWindowCompactor(maxTokens, keepLast int) *TokenWindowCompactor {
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
func (c *TokenWindowCompactor) ShouldCompact(messages []Message, usage *Usage) bool {
	// If we have real usage data, use it directly.
	if usage != nil && usage.PromptTokens > 0 {
		return usage.PromptTokens > c.maxTokens
	}

	// Estimate from content length.
	totalChars := 0
	for _, msg := range messages {
		for _, block := range msg.Content {
			switch block.Type {
			case ContentText:
				totalChars += len(block.Text)
			case ContentToolResult:
				if block.ToolResult != nil {
					totalChars += len(block.ToolResult.Content)
				}
			}
		}
	}

	estimated := totalChars / c.charsPerToken
	return estimated > c.maxTokens
}

// Compact uses the same sliding window strategy as SlidingWindowCompactor.
func (c *TokenWindowCompactor) Compact(ctx context.Context, messages []Message) ([]Message, error) {
	sw := &SlidingWindowCompactor{keepLast: c.keepLast}
	return sw.Compact(ctx, messages)
}

// NewCompactionNotice creates a system message indicating that older context
// was compacted. This helps the model understand there may be missing context.
func NewCompactionNotice(discardedCount, originalCount int) Message {
	return Message{
		Role: RoleSystem,
		Content: []ContentBlock{{
			Type: ContentText,
			Text: "[Context compacted: " +
				itoa(discardedCount) + " of " + itoa(originalCount) +
				" earlier messages were removed to fit context limits. " +
				"The conversation continues from the most recent messages below.]",
		}},
	}
}

// itoa is a minimal int-to-string without importing strconv.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	if n < 0 {
		return "-" + itoa(-n)
	}
	digits := make([]byte, 0, 10)
	for n > 0 {
		digits = append(digits, byte('0'+n%10))
		n /= 10
	}
	// Reverse.
	for i, j := 0, len(digits)-1; i < j; i, j = i+1, j-1 {
		digits[i], digits[j] = digits[j], digits[i]
	}
	return string(digits)
}
