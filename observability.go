package agentflow

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// --- OpenTelemetry-style Observability ---

// Trace represents a complete agent run with all spans (turns, tool calls).
type Trace struct {
	ID        string      `json:"id"`
	StartTime time.Time   `json:"start_time"`
	EndTime   time.Time   `json:"end_time"`
	Duration  time.Duration `json:"duration"`
	Spans     []Span      `json:"spans"`
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
//	tracer := agentflow.NewTracer()
//	agent := agentflow.NewAgent(provider, agentflow.WithHook(tracer.Hooks()...))
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
		traceID:   GenerateSessionID()[:16],
		startTime: time.Now(),
		metadata:  make(map[string]any),
	}
}

// Hooks returns the hooks that should be registered with the agent.
func (t *Tracer) Hooks() []Hook {
	return []Hook{
		HookFunc{
			HookPhase: HookPreModelCall,
			Fn:        t.preModelCall,
		},
		HookFunc{
			HookPhase: HookPostModelCall,
			Fn:        t.postModelCall,
		},
		HookFunc{
			HookPhase: HookPreToolUse,
			Fn:        t.preToolUse,
		},
		HookFunc{
			HookPhase: HookPostToolUse,
			Fn:        t.postToolUse,
		},
	}
}

func (t *Tracer) preModelCall(_ context.Context, hc *HookContext) (*HookAction, error) {
	hc.Metadata["_trace_model_start"] = time.Now()
	return nil, nil
}

func (t *Tracer) postModelCall(_ context.Context, hc *HookContext) (*HookAction, error) {
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

func (t *Tracer) preToolUse(_ context.Context, hc *HookContext) (*HookAction, error) {
	if hc.ToolCall != nil {
		hc.Metadata["_trace_tool_start_"+hc.ToolCall.ID] = time.Now()
	}
	return nil, nil
}

func (t *Tracer) postToolUse(_ context.Context, hc *HookContext) (*HookAction, error) {
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

// --- Provider Cost Tracking ---

// CostTracker estimates API costs based on token usage and model pricing.
//
//	tracker := agentflow.NewCostTracker()
//	tracker.SetPrice("llama-3.3-70b-versatile", 0.59, 0.79) // per 1M tokens
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithOnEvent(tracker.OnEvent),
//	)
//	// after run:
//	fmt.Printf("Total cost: $%.4f\n", tracker.TotalCost())
type CostTracker struct {
	mu     sync.Mutex
	prices map[string]ModelPrice
	usage  []UsageRecord
}

// ModelPrice defines the cost per million tokens for a model.
type ModelPrice struct {
	PromptPerMillion     float64 `json:"prompt_per_million"`
	CompletionPerMillion float64 `json:"completion_per_million"`
}

// UsageRecord captures a single API call's token usage.
type UsageRecord struct {
	Timestamp        time.Time `json:"timestamp"`
	PromptTokens     int       `json:"prompt_tokens"`
	CompletionTokens int       `json:"completion_tokens"`
	EstimatedCost    float64   `json:"estimated_cost"`
}

// NewCostTracker creates a tracker with default pricing for common models.
func NewCostTracker() *CostTracker {
	ct := &CostTracker{
		prices: map[string]ModelPrice{
			"llama-3.3-70b-versatile":                      {PromptPerMillion: 0.59, CompletionPerMillion: 0.79},
			"llama-3.1-8b-instant":                         {PromptPerMillion: 0.05, CompletionPerMillion: 0.08},
			"mixtral-8x7b-32768":                           {PromptPerMillion: 0.24, CompletionPerMillion: 0.24},
			"anthropic/claude-sonnet-4-20250514":            {PromptPerMillion: 3.00, CompletionPerMillion: 15.00},
			"anthropic/claude-haiku-4-5-20251001":           {PromptPerMillion: 0.80, CompletionPerMillion: 4.00},
			"openai/gpt-4o":                                {PromptPerMillion: 2.50, CompletionPerMillion: 10.00},
			"openai/gpt-4o-mini":                           {PromptPerMillion: 0.15, CompletionPerMillion: 0.60},
			"meta-llama/llama-4-scout-17b-16e-instruct":    {PromptPerMillion: 0.11, CompletionPerMillion: 0.34},
		},
	}
	return ct
}

// SetPrice sets the pricing for a model.
func (ct *CostTracker) SetPrice(model string, promptPerMillion, completionPerMillion float64) {
	ct.mu.Lock()
	ct.prices[model] = ModelPrice{
		PromptPerMillion:     promptPerMillion,
		CompletionPerMillion: completionPerMillion,
	}
	ct.mu.Unlock()
}

// OnEvent is the callback to register with WithOnEvent. It captures usage events.
func (ct *CostTracker) OnEvent(ev Event) {
	if ev.Type != EventUsage || ev.Usage == nil {
		return
	}

	ct.mu.Lock()
	defer ct.mu.Unlock()

	// Estimate cost using default pricing (model-specific lookup requires model ID
	// which isn't in the usage event — use a generic rate).
	cost := ct.estimateCost(ev.Usage.Usage.PromptTokens, ev.Usage.Usage.CompletionTokens)

	ct.usage = append(ct.usage, UsageRecord{
		Timestamp:        time.Now(),
		PromptTokens:     ev.Usage.Usage.PromptTokens,
		CompletionTokens: ev.Usage.Usage.CompletionTokens,
		EstimatedCost:    cost,
	})
}

// TrackUsage manually records a usage event with a specific model for accurate pricing.
func (ct *CostTracker) TrackUsage(model string, promptTokens, completionTokens int) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	cost := ct.estimateCostForModel(model, promptTokens, completionTokens)
	ct.usage = append(ct.usage, UsageRecord{
		Timestamp:        time.Now(),
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		EstimatedCost:    cost,
	})
}

// TotalCost returns the total estimated cost across all tracked API calls.
func (ct *CostTracker) TotalCost() float64 {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	var total float64
	for _, u := range ct.usage {
		total += u.EstimatedCost
	}
	return total
}

// TotalTokens returns aggregate token counts.
func (ct *CostTracker) TotalTokens() (prompt, completion int) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	for _, u := range ct.usage {
		prompt += u.PromptTokens
		completion += u.CompletionTokens
	}
	return
}

// Records returns all usage records.
func (ct *CostTracker) Records() []UsageRecord {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	return append([]UsageRecord{}, ct.usage...)
}

func (ct *CostTracker) estimateCost(promptTokens, completionTokens int) float64 {
	// Use a generic average rate when model is unknown.
	return (float64(promptTokens) * 0.50 / 1_000_000) + (float64(completionTokens) * 1.00 / 1_000_000)
}

func (ct *CostTracker) estimateCostForModel(model string, promptTokens, completionTokens int) float64 {
	price, ok := ct.prices[model]
	if !ok {
		return ct.estimateCost(promptTokens, completionTokens)
	}
	return (float64(promptTokens) * price.PromptPerMillion / 1_000_000) +
		(float64(completionTokens) * price.CompletionPerMillion / 1_000_000)
}
