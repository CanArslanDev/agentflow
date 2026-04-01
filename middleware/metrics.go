package middleware

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/CanArslanDev/agentflow"
)

// Metrics collects execution statistics from the agent loop. It provides
// a thread-safe, in-memory view of tool call counts, durations, and error rates.
//
//	m := middleware.NewMetrics()
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithHook(m.Hooks()...),
//	)
//	// After agent run:
//	fmt.Println(m.Snapshot())
type Metrics struct {
	mu           sync.RWMutex
	toolCalls    map[string]*toolMetrics
	totalCalls   atomic.Int64
	totalErrors  atomic.Int64
	totalTurns   atomic.Int64
}

type toolMetrics struct {
	calls    atomic.Int64
	errors   atomic.Int64
	totalMs  atomic.Int64
}

// MetricsSnapshot is a point-in-time view of collected metrics.
type MetricsSnapshot struct {
	TotalCalls  int64                      `json:"total_calls"`
	TotalErrors int64                      `json:"total_errors"`
	TotalTurns  int64                      `json:"total_turns"`
	ByTool      map[string]ToolSnapshot    `json:"by_tool"`
}

// ToolSnapshot is per-tool metrics.
type ToolSnapshot struct {
	Calls      int64         `json:"calls"`
	Errors     int64         `json:"errors"`
	AvgLatency time.Duration `json:"avg_latency"`
}

// NewMetrics creates a new Metrics collector.
func NewMetrics() *Metrics {
	return &Metrics{
		toolCalls: make(map[string]*toolMetrics),
	}
}

// Hooks returns the hooks that should be registered with the agent.
func (m *Metrics) Hooks() []agentflow.Hook {
	return []agentflow.Hook{
		agentflow.HookFunc{
			HookPhase: agentflow.HookPreToolUse,
			Fn: func(_ context.Context, hc *agentflow.HookContext) (*agentflow.HookAction, error) {
				if hc.ToolCall != nil {
					hc.Metadata["metrics_start_"+hc.ToolCall.ID] = time.Now()
				}
				return nil, nil
			},
		},
		agentflow.HookFunc{
			HookPhase: agentflow.HookPostToolUse,
			Fn: func(_ context.Context, hc *agentflow.HookContext) (*agentflow.HookAction, error) {
				if hc.ToolCall == nil || hc.ToolResult == nil {
					return nil, nil
				}

				tm := m.getOrCreate(hc.ToolCall.Name)
				tm.calls.Add(1)
				m.totalCalls.Add(1)

				if hc.ToolResult.IsError {
					tm.errors.Add(1)
					m.totalErrors.Add(1)
				}

				if start, ok := hc.Metadata["metrics_start_"+hc.ToolCall.ID].(time.Time); ok {
					tm.totalMs.Add(time.Since(start).Milliseconds())
					delete(hc.Metadata, "metrics_start_"+hc.ToolCall.ID)
				}

				return nil, nil
			},
		},
		agentflow.HookFunc{
			HookPhase: agentflow.HookPreModelCall,
			Fn: func(_ context.Context, _ *agentflow.HookContext) (*agentflow.HookAction, error) {
				m.totalTurns.Add(1)
				return nil, nil
			},
		},
	}
}

// Snapshot returns a point-in-time copy of all metrics.
func (m *Metrics) Snapshot() MetricsSnapshot {
	m.mu.RLock()
	defer m.mu.RUnlock()

	snap := MetricsSnapshot{
		TotalCalls:  m.totalCalls.Load(),
		TotalErrors: m.totalErrors.Load(),
		TotalTurns:  m.totalTurns.Load(),
		ByTool:      make(map[string]ToolSnapshot, len(m.toolCalls)),
	}

	for name, tm := range m.toolCalls {
		calls := tm.calls.Load()
		var avgLatency time.Duration
		if calls > 0 {
			avgLatency = time.Duration(tm.totalMs.Load()/calls) * time.Millisecond
		}
		snap.ByTool[name] = ToolSnapshot{
			Calls:      calls,
			Errors:     tm.errors.Load(),
			AvgLatency: avgLatency,
		}
	}

	return snap
}

func (m *Metrics) getOrCreate(name string) *toolMetrics {
	m.mu.RLock()
	tm, ok := m.toolCalls[name]
	m.mu.RUnlock()
	if ok {
		return tm
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if tm, ok = m.toolCalls[name]; ok {
		return tm
	}
	tm = &toolMetrics{}
	m.toolCalls[name] = tm
	return tm
}
