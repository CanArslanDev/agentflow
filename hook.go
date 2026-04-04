package agentflow

import (
	"context"
	"encoding/json"
)

// HookPhase defines when a hook fires in the execution pipeline.
type HookPhase int

const (
	// HookPreToolUse fires after input validation, before permission check.
	// Can block execution or modify input.
	HookPreToolUse HookPhase = iota

	// HookPostToolUse fires after tool execution completes.
	// Can inspect results, log metrics, or inject follow-up messages.
	HookPostToolUse

	// HookPreModelCall fires before each model API call.
	// Can modify messages, inject system context, or block the call.
	HookPreModelCall

	// HookPostModelCall fires after the model response is fully received.
	// Can inspect the response, log usage, or trigger side effects.
	HookPostModelCall

	// HookOnTurnEnd fires when the agentic loop would normally terminate
	// (no more tool calls). Can inject messages to force continuation.
	HookOnTurnEnd
)

// Hook intercepts the agent execution pipeline at defined lifecycle phases.
// Hooks execute synchronously in registration order. Keep hook execution fast
// to avoid blocking the agentic loop.
type Hook interface {
	// Phase returns when this hook should fire.
	Phase() HookPhase

	// Execute runs the hook logic. Returning a non-nil HookAction modifies
	// the pipeline behavior (block execution, modify input, inject messages).
	// Return nil action to proceed without modification.
	Execute(ctx context.Context, hc *HookContext) (*HookAction, error)
}

// HookContext provides read access to the current execution state when a hook fires.
type HookContext struct {
	// Phase indicates which lifecycle phase triggered this hook.
	Phase HookPhase

	// ToolCall is set for tool-phase hooks (PreToolUse, PostToolUse).
	ToolCall *ToolCall

	// ToolResult is set for PostToolUse hooks.
	ToolResult *ToolResult

	// Messages is the current conversation history at the time the hook fires.
	Messages []Message

	// TurnCount is the current iteration number of the agentic loop.
	TurnCount int

	// Metadata is a mutable key-value bag that persists across hooks within a
	// single agent run. Hooks can use it to communicate state to each other.
	Metadata map[string]any
}

// HookAction tells the execution pipeline how to proceed after a hook fires.
// All fields are optional; a nil HookAction means "continue normally."
type HookAction struct {
	// Block, if true, prevents the current operation from proceeding.
	// For PreToolUse: the tool is not executed; BlockReason is sent to the model.
	// For PreModelCall: the model call is skipped; the loop terminates.
	Block       bool
	BlockReason string

	// ModifiedInput, if non-nil, replaces the tool's input before execution.
	// Only effective for PreToolUse hooks.
	ModifiedInput json.RawMessage

	// InjectMessages appends additional messages to the conversation before
	// the next model call. Effective for PostToolUse and OnTurnEnd hooks.
	InjectMessages []Message
}

// MultiPhaseHook is an optional interface for hooks that fire at multiple phases.
// Hooks implementing this interface have their Execute called for each phase
// returned by Phases(), eliminating the need to register duplicate hooks.
type MultiPhaseHook interface {
	// Phases returns all phases this hook should fire at.
	Phases() []HookPhase

	// Execute runs the hook logic for the given phase.
	Execute(ctx context.Context, hc *HookContext) (*HookAction, error)
}

// HookFunc is an adapter that allows ordinary functions to be used as hooks.
// It pairs a phase with a function, eliminating the need for a struct that
// implements the Hook interface for simple cases.
type HookFunc struct {
	HookPhase HookPhase
	Fn        func(ctx context.Context, hc *HookContext) (*HookAction, error)
}

// Phase returns the hook's firing phase.
func (h HookFunc) Phase() HookPhase {
	return h.HookPhase
}

// Execute delegates to the wrapped function.
func (h HookFunc) Execute(ctx context.Context, hc *HookContext) (*HookAction, error) {
	return h.Fn(ctx, hc)
}
