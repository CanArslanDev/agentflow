package agentflow

import "context"

// Compactor manages conversation history when the context window limit is
// approached. Implementations decide when compaction is needed and how to
// reduce the message history while preserving essential context.
type Compactor interface {
	// ShouldCompact reports whether the current message history needs compaction.
	// Called at the beginning of each agentic loop iteration before the model call.
	ShouldCompact(messages []Message, usage *Usage) bool

	// Compact reduces the message history to fit within context limits.
	// Returns the compacted messages that replace the original history.
	// The implementation may use summarization, truncation, or any other strategy.
	Compact(ctx context.Context, messages []Message) ([]Message, error)
}
