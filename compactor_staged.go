package agentflow

import "context"

// StagedCompactor applies multiple compaction strategies in sequence.
// First tries a lightweight strategy; if the result is still too large,
// falls through to heavier strategies.
//
//	compactor := agentflow.NewStagedCompactor(
//	    agentflow.NewSlidingWindowCompactor(30, 40),  // Stage 1: keep 30
//	    agentflow.NewSlidingWindowCompactor(15, 20),  // Stage 2: keep 15
//	    agentflow.NewSummaryCompactor(provider, 10, 0), // Stage 3: summarize
//	)
type StagedCompactor struct {
	stages       []Compactor
	tokenPerChar int // Estimated chars per token for size checking. Default: 4.
	targetTokens int // Target token count after compaction. Default: 4000.
}

// NewStagedCompactor creates a compactor that tries each stage in order.
// Stages should be ordered from lightest (fastest) to heaviest (most aggressive).
func NewStagedCompactor(stages ...Compactor) *StagedCompactor {
	return &StagedCompactor{
		stages:       stages,
		tokenPerChar: 4,
		targetTokens: 4000,
	}
}

// WithTarget sets the target token count after compaction.
func (c *StagedCompactor) WithTarget(tokens int) *StagedCompactor {
	c.targetTokens = tokens
	return c
}

// ShouldCompact returns true if any stage thinks compaction is needed.
func (c *StagedCompactor) ShouldCompact(messages []Message, usage *Usage) bool {
	for _, stage := range c.stages {
		if stage.ShouldCompact(messages, usage) {
			return true
		}
	}
	return false
}

// Compact tries each stage in order. After each stage, checks if the result
// is small enough. Stops at the first stage that brings messages under target.
func (c *StagedCompactor) Compact(ctx context.Context, messages []Message) ([]Message, error) {
	current := messages

	for _, stage := range c.stages {
		if !stage.ShouldCompact(current, nil) {
			continue
		}

		compacted, err := stage.Compact(ctx, current)
		if err != nil {
			continue // Try next stage on error.
		}

		current = compacted

		// Check if we're under target.
		if c.estimateTokens(current) <= c.targetTokens {
			break
		}
	}

	return current, nil
}

// estimateTokens gives a rough token count from message content lengths.
func (c *StagedCompactor) estimateTokens(messages []Message) int {
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
	cpt := c.tokenPerChar
	if cpt <= 0 {
		cpt = 4
	}
	return totalChars / cpt
}

// ContextCollapser merges consecutive tool call + result pairs into a
// single summary message. This preserves the semantic content while
// dramatically reducing message count.
type ContextCollapser struct {
	// triggerAt is the number of tool-result pairs before collapsing.
	triggerAt int
}

// NewContextCollapser creates a collapser that triggers after N tool-result pairs.
func NewContextCollapser(triggerAt int) *ContextCollapser {
	if triggerAt < 2 {
		triggerAt = 2
	}
	return &ContextCollapser{triggerAt: triggerAt}
}

// ShouldCompact counts tool-result pairs in the conversation.
func (c *ContextCollapser) ShouldCompact(messages []Message, _ *Usage) bool {
	pairs := 0
	for _, msg := range messages {
		for _, block := range msg.Content {
			if block.Type == ContentToolResult {
				pairs++
			}
		}
	}
	return pairs > c.triggerAt
}

// Compact collapses consecutive assistant(tool_call) + user(tool_result) pairs
// into a single system message summarizing what happened.
func (c *ContextCollapser) Compact(_ context.Context, messages []Message) ([]Message, error) {
	if len(messages) < 4 {
		return messages, nil
	}

	var result []Message
	// Always keep the first message.
	result = append(result, messages[0])

	i := 1
	var collapsed int
	for i < len(messages) {
		msg := messages[i]

		// Check if this is an assistant message with tool calls followed by tool results.
		if msg.Role == RoleAssistant && len(msg.ToolCalls()) > 0 && i+1 < len(messages) {
			nextMsg := messages[i+1]
			if nextMsg.Role == RoleUser && len(nextMsg.ToolResults()) > 0 {
				// Collapse this pair into a summary.
				calls := msg.ToolCalls()
				results := nextMsg.ToolResults()

				var summary string
				for j, call := range calls {
					summary += "[" + call.Name + "]"
					if j < len(results) && !results[j].IsError {
						content := results[j].Content
						if len(content) > 100 {
							content = content[:100] + "..."
						}
						summary += " -> " + content
					}
					summary += "\n"
				}

				result = append(result, Message{
					Role: RoleSystem,
					Content: []ContentBlock{{
						Type: ContentText,
						Text: "[Collapsed tool execution]\n" + summary,
					}},
				})

				collapsed++
				i += 2
				continue
			}
		}

		// Keep recent messages as-is (last few should not be collapsed).
		if i >= len(messages)-4 {
			result = append(result, msg)
			i++
			continue
		}

		result = append(result, msg)
		i++
	}

	return result, nil
}
