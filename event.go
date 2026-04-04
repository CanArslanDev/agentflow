package agentflow

import "time"

// EventType discriminates the variant within an Event.
type EventType int

const (
	// EventTextDelta carries streaming text from the model's response.
	EventTextDelta EventType = iota

	// EventThinkingDelta carries streaming thinking/reasoning content from the model.
	EventThinkingDelta

	// EventToolStart signals that a tool execution is beginning.
	EventToolStart

	// EventToolProgress carries incremental progress from a running tool.
	EventToolProgress

	// EventToolEnd signals that a tool execution has completed.
	EventToolEnd

	// EventTurnStart signals the beginning of a new agentic loop iteration.
	EventTurnStart

	// EventTurnEnd signals the end of a loop iteration, or the loop itself.
	EventTurnEnd

	// EventMessage carries a complete message added to the conversation history.
	EventMessage

	// EventError carries a recoverable error (e.g., a retry is in progress).
	EventError

	// EventUsage carries token usage statistics for the current turn.
	EventUsage

	// EventSubAgentStart signals that a sub-agent has been spawned.
	EventSubAgentStart

	// EventSubAgentEnd signals that a sub-agent has completed.
	EventSubAgentEnd

	// EventBudgetWarning signals that token consumption crossed the warning threshold.
	EventBudgetWarning

	// EventCompaction signals that conversation history was compacted.
	EventCompaction

	// EventRetry signals that a provider call is being retried after a transient error.
	EventRetry

	// EventPermissionDenied signals that a tool call was blocked by the permission checker.
	EventPermissionDenied

	// EventHookBlocked signals that a hook blocked tool execution or the model call.
	EventHookBlocked
)

// Event is the primary output type of the agentic loop. It is a discriminated union
// where exactly one of the typed fields is set, determined by the Type field.
//
// Events are delivered through the channel returned by [Agent.Run]. Consumers
// switch on Type to handle each variant.
type Event struct {
	Type EventType

	TextDelta     *TextDeltaEvent      // Type == EventTextDelta
	ThinkDelta    *TextDeltaEvent      // Type == EventThinkingDelta
	ToolStart     *ToolStartEvent      // Type == EventToolStart
	ToolProgress  *ProgressEvent       // Type == EventToolProgress
	ToolEnd       *ToolEndEvent        // Type == EventToolEnd
	TurnStart     *TurnStartEvent      // Type == EventTurnStart
	TurnEnd       *TurnEndEvent        // Type == EventTurnEnd
	Message       *Message             // Type == EventMessage
	Error         *ErrorEvent          // Type == EventError
	Usage         *UsageEvent          // Type == EventUsage
	SubAgentStart *SubAgentStartEvent  // Type == EventSubAgentStart
	SubAgentEnd   *SubAgentEndEvent    // Type == EventSubAgentEnd
	BudgetWarning    *BudgetWarningEvent    // Type == EventBudgetWarning
	Compaction       *CompactionEvent       // Type == EventCompaction
	Retry            *RetryEvent            // Type == EventRetry
	PermissionDenied *PermissionDeniedEvent // Type == EventPermissionDenied
	HookBlocked      *HookBlockedEvent      // Type == EventHookBlocked

	// SubAgentIndex identifies which child agent emitted this event.
	// Zero for the parent agent's own events. Set by SpawnChildren.
	SubAgentIndex int
}

// TextDeltaEvent carries an incremental text chunk from the model.
type TextDeltaEvent struct {
	Text string
}

// ToolStartEvent signals that a tool invocation is about to execute.
type ToolStartEvent struct {
	ToolCall ToolCall
}

// ToolEndEvent signals that a tool invocation has completed.
type ToolEndEvent struct {
	ToolCall ToolCall
	Result   ToolResult
	Duration time.Duration
}

// TurnStartEvent signals the beginning of a new iteration in the agentic loop.
type TurnStartEvent struct {
	TurnNumber int
}

// TurnEndEvent signals the end of an iteration or the entire loop.
type TurnEndEvent struct {
	TurnNumber int
	Reason     TurnEndReason
	Messages   []Message // Final conversation history at loop termination.
}

// TurnEndReason describes why the agentic loop iteration or the loop itself ended.
type TurnEndReason string

const (
	// TurnEndComplete means the model finished without requesting tool calls.
	TurnEndComplete TurnEndReason = "completed"

	// TurnEndMaxTurns means the configured maximum turn limit was reached.
	TurnEndMaxTurns TurnEndReason = "max_turns"

	// TurnEndAborted means the context was cancelled by the caller.
	TurnEndAborted TurnEndReason = "aborted"

	// TurnEndError means an unrecoverable error terminated the loop.
	TurnEndError TurnEndReason = "error"

	// TurnEndHookBlock means a hook prevented the loop from continuing.
	TurnEndHookBlock TurnEndReason = "hook_blocked"

	// TurnEndBudgetExhausted means the token budget was fully consumed.
	TurnEndBudgetExhausted TurnEndReason = "budget_exhausted"
)

// ErrorEvent carries a recoverable error from the agentic loop.
type ErrorEvent struct {
	Err       error
	Retrying  bool
	TurnCount int
}

// UsageEvent carries token usage statistics for a single model call.
type UsageEvent struct {
	Usage     Usage
	TurnCount int
}

// BudgetWarningEvent signals that token consumption crossed the configured threshold.
type BudgetWarningEvent struct {
	ConsumedTokens int     // Total tokens consumed so far.
	MaxTokens      int     // The configured budget limit.
	Percentage     float64 // Consumption as a fraction (0.0–1.0).
}

// CompactionEvent signals that conversation history was compacted.
type CompactionEvent struct {
	BeforeCount int // Message count before compaction.
	AfterCount  int // Message count after compaction.
	TurnCount   int
}

// RetryEvent signals that a provider call is being retried.
type RetryEvent struct {
	Attempt   int           // Current retry attempt (1-based).
	Delay     time.Duration // Delay before this retry.
	Err       error         // The error that triggered the retry.
	TurnCount int
}

// PermissionDeniedEvent signals that a tool call was blocked by permissions.
type PermissionDeniedEvent struct {
	ToolCall ToolCall
}

// HookBlockedEvent signals that a hook blocked execution.
type HookBlockedEvent struct {
	Phase      HookPhase
	ToolCall   *ToolCall // Non-nil for PreToolUse blocks.
	Reason     string
	TurnCount  int
}

// FilterEvents returns a channel that only delivers events matching the specified types.
// The returned channel is closed when the input channel is closed.
//
//	for ev := range agentflow.FilterEvents(agent.Run(ctx, msgs), agentflow.EventTextDelta, agentflow.EventTurnEnd) {
//	    // only text deltas and turn ends
//	}
func FilterEvents(ch <-chan Event, types ...EventType) <-chan Event {
	allowed := make(map[EventType]bool, len(types))
	for _, t := range types {
		allowed[t] = true
	}

	out := make(chan Event, cap(ch))
	go func() {
		defer close(out)
		for ev := range ch {
			if allowed[ev.Type] {
				out <- ev
			}
		}
	}()
	return out
}

// SubAgentStartEvent signals that a sub-agent has been spawned.
type SubAgentStartEvent struct {
	Index int    // Child index (0-based) within SpawnChildren.
	Task  string // The task delegated to the sub-agent.
}

// SubAgentEndEvent signals that a sub-agent has completed its work.
type SubAgentEndEvent struct {
	Index  int    // Child index matching SubAgentStartEvent.
	Task   string // The original task.
	Result string // The sub-agent's final text response.
}
