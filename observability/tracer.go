// Package observability provides tracing and cost tracking for agent runs.
// The Tracer captures OpenTelemetry-style spans from model calls and tool
// executions. The CostTracker estimates API costs based on token usage.
package observability

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/CanArslanDev/agentflow"
)

// Trace represents a complete agent run with all spans (turns, tool calls).
type Trace struct {
	ID        string        `json:"id"`
	StartTime time.Time     `json:"start_time"`
	EndTime   time.Time     `json:"end_time"`
	Duration  time.Duration `json:"duration"`
	Spans     []Span        `json:"spans"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

// Span represents a single operation within a trace (model call, tool execution).
type Span struct {
	ID        string         `json:"id"`
	ParentID  string         `json:"parent_id,omitempty"`
	Name      string         `json:"name"`
	Kind      SpanKind       `json:"kind"`
	StartTime time.Time      `json:"start_time"`
	EndTime   time.Time      `json:"end_time"`
	Duration  time.Duration  `json:"duration"`
	Status    SpanStatus     `json:"status"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

// SpanKind identifies the type of operation.
type SpanKind string

const (
	SpanKindModelCall     SpanKind = "model_call"
	SpanKindToolExecution SpanKind = "tool_execution"
	SpanKindTurn          SpanKind = "turn"
	SpanKindSubAgent      SpanKind = "sub_agent"
	SpanKindSkill         SpanKind = "skill"
)

// SpanStatus indicates the outcome of a span.
type SpanStatus string

const (
	SpanStatusOK    SpanStatus = "ok"
	SpanStatusError SpanStatus = "error"
)

// Tracer collects spans from an agent run into a structured trace.
// Register it as hooks to capture all lifecycle events automatically.
//
//	tracer := observability.NewTracer()
//	hooks := tracer.Hooks()
//	agent := agentflow.NewAgent(provider, agentflow.WithHook(hooks[0]), agentflow.WithHook(hooks[1]), ...)
//	// after run:
//	trace := tracer.Finish()
type Tracer struct {
	mu        sync.Mutex
	traceID   string
	startTime time.Time
	spans     []Span
	spanCount atomic.Int64
	metadata  map[string]any
}

// NewTracer creates a tracer for capturing a single agent run.
func NewTracer() *Tracer {
	return &Tracer{
		traceID:   agentflow.GenerateSessionID()[:16],
		startTime: time.Now(),
		metadata:  make(map[string]any),
	}
}

// Hooks returns the hooks that should be registered with the agent.
func (t *Tracer) Hooks() []agentflow.Hook {
	return []agentflow.Hook{
		agentflow.HookFunc{
			HookPhase: agentflow.HookPreModelCall,
			Fn:        t.preModelCall,
		},
		agentflow.HookFunc{
			HookPhase: agentflow.HookPostModelCall,
			Fn:        t.postModelCall,
		},
		agentflow.HookFunc{
			HookPhase: agentflow.HookPreToolUse,
			Fn:        t.preToolUse,
		},
		agentflow.HookFunc{
			HookPhase: agentflow.HookPostToolUse,
			Fn:        t.postToolUse,
		},
	}
}

func (t *Tracer) preModelCall(_ context.Context, hc *agentflow.HookContext) (*agentflow.HookAction, error) {
	hc.Metadata["_trace_model_start"] = time.Now()
	return nil, nil
}

func (t *Tracer) postModelCall(_ context.Context, hc *agentflow.HookContext) (*agentflow.HookAction, error) {
	start, ok := hc.Metadata["_trace_model_start"].(time.Time)
	if !ok {
		return nil, nil
	}
	delete(hc.Metadata, "_trace_model_start")

	t.addSpan(Span{
		Name:      fmt.Sprintf("model_call_turn_%d", hc.TurnCount),
		Kind:      SpanKindModelCall,
		StartTime: start,
		EndTime:   time.Now(),
		Duration:  time.Since(start),
		Status:    SpanStatusOK,
		Metadata:  map[string]any{"turn": hc.TurnCount},
	})
	return nil, nil
}

func (t *Tracer) preToolUse(_ context.Context, hc *agentflow.HookContext) (*agentflow.HookAction, error) {
	if hc.ToolCall != nil {
		hc.Metadata["_trace_tool_start_"+hc.ToolCall.ID] = time.Now()
	}
	return nil, nil
}

func (t *Tracer) postToolUse(_ context.Context, hc *agentflow.HookContext) (*agentflow.HookAction, error) {
	if hc.ToolCall == nil {
		return nil, nil
	}
	start, ok := hc.Metadata["_trace_tool_start_"+hc.ToolCall.ID].(time.Time)
	if !ok {
		return nil, nil
	}
	delete(hc.Metadata, "_trace_tool_start_"+hc.ToolCall.ID)

	status := SpanStatusOK
	if hc.ToolResult != nil && hc.ToolResult.IsError {
		status = SpanStatusError
	}

	t.addSpan(Span{
		Name:      "tool:" + hc.ToolCall.Name,
		Kind:      SpanKindToolExecution,
		StartTime: start,
		EndTime:   time.Now(),
		Duration:  time.Since(start),
		Status:    status,
		Metadata: map[string]any{
			"tool_name": hc.ToolCall.Name,
			"call_id":   hc.ToolCall.ID,
			"is_error":  hc.ToolResult != nil && hc.ToolResult.IsError,
		},
	})
	return nil, nil
}

func (t *Tracer) addSpan(span Span) {
	span.ID = fmt.Sprintf("span_%d", t.spanCount.Add(1))
	t.mu.Lock()
	t.spans = append(t.spans, span)
	t.mu.Unlock()
}

// Finish completes the trace and returns the collected data.
func (t *Tracer) Finish() *Trace {
	t.mu.Lock()
	defer t.mu.Unlock()

	now := time.Now()
	return &Trace{
		ID:        t.traceID,
		StartTime: t.startTime,
		EndTime:   now,
		Duration:  now.Sub(t.startTime),
		Spans:     append([]Span{}, t.spans...),
		Metadata:  t.metadata,
	}
}

// SpanCount returns the number of spans collected so far.
func (t *Tracer) SpanCount() int {
	return int(t.spanCount.Load())
}
