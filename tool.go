package agentflow

import (
	"context"
	"encoding/json"
	"time"
)

// Tool defines a capability that an agent can invoke. Tools are registered with
// an Agent and presented to the AI model as callable functions. The model decides
// when and how to invoke tools based on their name, description, and input schema.
//
// Implementations must be safe for concurrent use if IsConcurrencySafe returns true.
type Tool interface {
	// Name returns the unique identifier for this tool. Must be alphanumeric
	// with underscores, matching the pattern [a-zA-Z_][a-zA-Z0-9_]*.
	Name() string

	// Description returns a human-readable description sent to the model.
	// A clear, specific description significantly improves tool selection accuracy.
	Description() string

	// InputSchema returns the JSON Schema object describing the tool's input parameters.
	// This schema is sent to the model and used for input validation before execution.
	InputSchema() map[string]any

	// Execute runs the tool with the given validated input. The context carries
	// cancellation signals from the agent loop. Implementations should respect
	// context cancellation for long-running operations.
	//
	// The progress function, if non-nil, can be called to report incremental
	// progress updates that are forwarded to the event stream.
	Execute(ctx context.Context, input json.RawMessage, progress ProgressFunc) (*ToolResult, error)

	// IsConcurrencySafe reports whether this tool can safely execute in parallel
	// with other concurrency-safe tools given the specific input. Read-only tools
	// should return true. Tools that mutate shared state should return false.
	IsConcurrencySafe(input json.RawMessage) bool

	// IsReadOnly reports whether the tool performs only read operations for the
	// given input. Used by permission checkers to make access control decisions.
	IsReadOnly(input json.RawMessage) bool
}

// ToolDefinition is the serializable schema sent to the AI provider. It describes
// a tool's interface so the model knows how to invoke it.
type ToolDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
}

// ToolResult is the outcome of a tool execution. The Content field is sent back
// to the model, while Metadata is included only in events for observability.
type ToolResult struct {
	// Content is the textual result sent to the model. For successful operations,
	// this contains the output data. For failures, this contains the error description.
	Content string

	// IsError indicates the tool execution failed. The model receives Content as
	// an error message and can adapt its approach (retry, use a different tool, etc.).
	IsError bool

	// Metadata is not sent to the model but is included in ToolEndEvent for
	// observability. Use it for execution metrics, debug data, or audit information.
	Metadata map[string]any
}

// ProgressFunc is called by tools to report incremental progress during execution.
// Progress events are forwarded to the agent's event stream in real time.
type ProgressFunc func(ProgressEvent)

// ProgressEvent describes incremental progress from a tool execution.
type ProgressEvent struct {
	// ToolCallID identifies which tool invocation this progress belongs to.
	// Set automatically by the executor; tools do not need to set this.
	ToolCallID string

	// Message is a human-readable progress description.
	Message string

	// Data carries tool-specific structured progress information.
	Data any
}

// --- Execution Mode & Tool Locality ---

// ExecutionMode determines which environment the agent is running in.
// This controls which tools are visible to the model.
type ExecutionMode int

const (
	// ModeLocal allows all tools. Use when the agent runs on the user's machine
	// where filesystem and shell access are expected.
	ModeLocal ExecutionMode = iota

	// ModeRemote restricts tools to only those marked as remote-safe.
	// Use when the agent runs on a server where local filesystem/shell access
	// is inappropriate or dangerous. The model only sees remote-safe tools —
	// it cannot call tools it doesn't know about.
	ModeRemote
)

// ToolLocality declares which execution environments a tool supports.
type ToolLocality int

const (
	// ToolLocalOnly means the tool requires local machine access (filesystem,
	// shell, local processes). Blocked in ModeRemote.
	ToolLocalOnly ToolLocality = iota

	// ToolRemoteSafe means the tool is safe for server execution. It does not
	// access the local filesystem or run local commands. Allowed in both modes.
	ToolRemoteSafe

	// ToolAny means the tool has no environment dependency. It works identically
	// in both local and remote modes (e.g., sleep, pure computation).
	ToolAny
)

// LocalityAware is an optional interface that tools can implement to declare
// their execution environment compatibility. Tools that do not implement this
// interface are treated as ToolLocalOnly — the safe default that prevents
// accidental remote execution of local-only tools.
type LocalityAware interface {
	Locality() ToolLocality
}

// TimeoutAware is an optional interface that tools can implement to declare
// their own execution timeout. When implemented and the returned duration is
// positive, this timeout takes precedence over the global WithToolTimeout
// configuration. Tools that do not implement this interface use the global timeout.
type TimeoutAware interface {
	Timeout() time.Duration
}

// toolLocality returns the locality of a tool. If the tool implements
// LocalityAware, its declaration is used. Otherwise, ToolLocalOnly is assumed.
func toolLocality(t Tool) ToolLocality {
	if la, ok := t.(LocalityAware); ok {
		return la.Locality()
	}
	return ToolLocalOnly
}

// IsToolAllowed checks whether a tool is permitted in the given execution mode.
// In ModeLocal, all tools are allowed. In ModeRemote, only ToolRemoteSafe
// and ToolAny tools are permitted.
func IsToolAllowed(t Tool, mode ExecutionMode) bool {
	if mode == ModeLocal {
		return true
	}
	locality := toolLocality(t)
	return locality == ToolRemoteSafe || locality == ToolAny
}

// toolDefinitionFrom converts a Tool interface into the serializable ToolDefinition
// that is sent to the AI provider.
func toolDefinitionFrom(t Tool) ToolDefinition {
	return ToolDefinition{
		Name:        t.Name(),
		Description: t.Description(),
		InputSchema: t.InputSchema(),
	}
}
