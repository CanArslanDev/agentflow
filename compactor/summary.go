package compactor

import (
	"context"
	"fmt"
	"io"

	"github.com/CanArslanDev/agentflow"
)

// SummaryCompactor uses an AI provider to summarize older messages before
// discarding them. This preserves more context than a simple sliding window
// by creating a condensed summary of the discarded conversation.
//
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithCompactor(compactor.NewSummary(provider, 20, 0)),
//	)
type SummaryCompactor struct {
	// provider is used to generate the summary (can be the same or different
	// provider than the main agent — e.g., a cheaper/faster model).
	provider agentflow.Provider

	// keepLast is the number of recent messages to retain after compaction.
	keepLast int

	// triggerAt is the message count threshold. Zero = keepLast * 2.
	triggerAt int

	// maxSummaryTokens limits the summary generation response.
	maxSummaryTokens int
}

// NewSummary creates a compactor that summarizes older messages using
// the given provider. keepLast messages are retained; the rest are summarized.
func NewSummary(provider agentflow.Provider, keepLast, triggerAt int) *SummaryCompactor {
	if keepLast < 2 {
		keepLast = 2
	}
	if triggerAt <= 0 {
		triggerAt = keepLast * 2
	}
	return &SummaryCompactor{
		provider:         provider,
		keepLast:         keepLast,
		triggerAt:        triggerAt,
		maxSummaryTokens: 500,
	}
}

// WithMaxSummaryTokens sets the token limit for the summary generation.
func (c *SummaryCompactor) WithMaxSummaryTokens(n int) *SummaryCompactor {
	c.maxSummaryTokens = n
	return c
}

// ShouldCompact returns true when message count exceeds the trigger threshold.
func (c *SummaryCompactor) ShouldCompact(messages []agentflow.Message, _ *agentflow.Usage) bool {
	return len(messages) > c.triggerAt
}

// Compact summarizes the older messages and retains the most recent ones.
func (c *SummaryCompactor) Compact(ctx context.Context, messages []agentflow.Message) ([]agentflow.Message, error) {
	if len(messages) <= c.keepLast+1 {
		return messages, nil
	}

	// Split: first message + messages to summarize + recent messages.
	first := messages[0]
	cutoff := len(messages) - c.keepLast
	toSummarize := messages[1:cutoff]
	recent := messages[cutoff:]

	if len(toSummarize) == 0 {
		return messages, nil
	}

	// Build the summarization request.
	summary, err := c.generateSummary(ctx, toSummarize)
	if err != nil {
		// Fallback to sliding window if summarization fails.
		sw := &SlidingWindowCompactor{keepLast: c.keepLast}
		return sw.Compact(ctx, messages)
	}

	result := make([]agentflow.Message, 0, c.keepLast+2)
	result = append(result, first)
	result = append(result, agentflow.Message{
		Role: agentflow.RoleSystem,
		Content: []agentflow.ContentBlock{{
			Type: agentflow.ContentText,
			Text: fmt.Sprintf("[Summary of %d earlier messages]\n%s", len(toSummarize), summary),
		}},
	})
	result = append(result, recent...)

	return result, nil
}

// generateSummary calls the provider to create a condensed summary of messages.
func (c *SummaryCompactor) generateSummary(ctx context.Context, messages []agentflow.Message) (string, error) {
	// Build a conversation asking the model to summarize.
	var summaryContent string
	for _, msg := range messages {
		text := msg.TextContent()
		if text == "" {
			continue
		}
		prefix := string(msg.Role)
		summaryContent += prefix + ": " + text + "\n"
	}

	req := &agentflow.Request{
		Messages: []agentflow.Message{
			{Role: agentflow.RoleUser, Content: []agentflow.ContentBlock{{
				Type: agentflow.ContentText,
				Text: "Summarize the following conversation concisely. " +
					"Preserve key facts, decisions, tool results, and context that would be needed to continue the conversation. " +
					"Keep it under 200 words.\n\n" + summaryContent,
			}}},
		},
		SystemPrompt: "You are a conversation summarizer. Be concise and preserve essential context.",
		MaxTokens:    c.maxSummaryTokens,
	}

	stream, err := c.provider.CreateStream(ctx, req)
	if err != nil {
		return "", err
	}
	defer stream.Close()

	var result string
	for {
		ev, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
		if ev.Type == agentflow.StreamEventDelta && ev.Delta != nil {
			result += ev.Delta.Text
		}
	}

	return result, nil
}
