package middleware

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/CanArslanDev/agentflow"
)

// CircuitState represents the current state of a circuit breaker.
type CircuitState int

const (
	// CircuitClosed allows all requests through (normal operation).
	CircuitClosed CircuitState = iota

	// CircuitOpen blocks all requests (too many failures).
	CircuitOpen

	// CircuitHalfOpen allows a single probe request to test recovery.
	CircuitHalfOpen
)

// CircuitBreaker blocks tool execution after consecutive failures exceed a
// threshold. After a reset duration, the circuit enters half-open state and
// allows a single probe request. If it succeeds, the circuit closes; if it
// fails, it re-opens.
//
//	cb := middleware.NewCircuitBreaker(3, 30*time.Second)
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithHook(cb.Hooks()[0]),
//	    agentflow.WithHook(cb.Hooks()[1]),
//	)
type CircuitBreaker struct {
	threshold     int
	resetDuration time.Duration

	mu       sync.Mutex
	circuits map[string]*circuit
}

type circuit struct {
	state       CircuitState
	failures    int
	lastFailure time.Time
}

// NewCircuitBreaker creates a circuit breaker that opens after threshold
// consecutive failures and attempts recovery after resetDuration.
func NewCircuitBreaker(threshold int, resetDuration time.Duration) *CircuitBreaker {
	if threshold < 1 {
		threshold = 1
	}
	return &CircuitBreaker{
		threshold:     threshold,
		resetDuration: resetDuration,
		circuits:      make(map[string]*circuit),
	}
}

// Hooks returns PreToolUse (to block) and PostToolUse (to track) hooks.
func (cb *CircuitBreaker) Hooks() []agentflow.Hook {
	return []agentflow.Hook{
		agentflow.HookFunc{
			HookPhase: agentflow.HookPreToolUse,
			Fn:        cb.preToolUse,
		},
		agentflow.HookFunc{
			HookPhase: agentflow.HookPostToolUse,
			Fn:        cb.postToolUse,
		},
	}
}

func (cb *CircuitBreaker) preToolUse(_ context.Context, hc *agentflow.HookContext) (*agentflow.HookAction, error) {
	if hc.ToolCall == nil {
		return nil, nil
	}

	cb.mu.Lock()
	c, ok := cb.circuits[hc.ToolCall.Name]
	if !ok {
		cb.mu.Unlock()
		return nil, nil
	}

	switch c.state {
	case CircuitOpen:
		// Check if reset duration has elapsed.
		if time.Since(c.lastFailure) >= cb.resetDuration {
			c.state = CircuitHalfOpen
			cb.mu.Unlock()
			return nil, nil // Allow probe request.
		}
		cb.mu.Unlock()
		return &agentflow.HookAction{
			Block:       true,
			BlockReason: fmt.Sprintf("circuit breaker open for tool %q (%d consecutive failures)", hc.ToolCall.Name, c.failures),
		}, nil

	case CircuitHalfOpen:
		cb.mu.Unlock()
		return nil, nil // Allow probe request.

	default:
		cb.mu.Unlock()
		return nil, nil
	}
}

func (cb *CircuitBreaker) postToolUse(_ context.Context, hc *agentflow.HookContext) (*agentflow.HookAction, error) {
	if hc.ToolCall == nil || hc.ToolResult == nil {
		return nil, nil
	}

	cb.mu.Lock()
	defer cb.mu.Unlock()

	c, ok := cb.circuits[hc.ToolCall.Name]
	if !ok {
		c = &circuit{}
		cb.circuits[hc.ToolCall.Name] = c
	}

	if hc.ToolResult.IsError {
		c.failures++
		c.lastFailure = time.Now()
		if c.failures >= cb.threshold {
			c.state = CircuitOpen
		}
	} else {
		// Success resets the circuit.
		c.failures = 0
		c.state = CircuitClosed
	}

	return nil, nil
}

// State returns the current circuit state for a tool. Returns CircuitClosed
// if the tool has no circuit breaker state.
func (cb *CircuitBreaker) State(toolName string) CircuitState {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	c, ok := cb.circuits[toolName]
	if !ok {
		return CircuitClosed
	}
	return c.state
}

// Reset clears the circuit breaker state for all tools.
func (cb *CircuitBreaker) Reset() {
	cb.mu.Lock()
	cb.circuits = make(map[string]*circuit)
	cb.mu.Unlock()
}
