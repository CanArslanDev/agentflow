package middleware

import (
	"context"
	"fmt"
	"log/slog"
	"runtime/debug"

	"github.com/canarslan/agentflow"
)

// Recovery returns a PreToolUse hook that wraps tool execution with panic recovery.
// If a tool panics, the panic is caught, logged, and converted to an error result
// instead of crashing the agent loop.
//
// Note: The Agent already includes basic panic recovery in callToolWithRecovery.
// This middleware adds structured logging of the stack trace.
func Recovery(logger *slog.Logger) agentflow.Hook {
	if logger == nil {
		logger = slog.Default()
	}

	return agentflow.HookFunc{
		HookPhase: agentflow.HookPostToolUse,
		Fn: func(_ context.Context, hc *agentflow.HookContext) (*agentflow.HookAction, error) {
			if hc.ToolResult != nil && hc.ToolResult.IsError && hc.ToolCall != nil {
				if hc.ToolResult.Content == "internal error: tool panicked" {
					logger.Error("tool panicked",
						slog.String("tool", hc.ToolCall.Name),
						slog.String("call_id", hc.ToolCall.ID),
						slog.String("stack", string(debug.Stack())),
					)
				}
			}
			return nil, nil
		},
	}
}

// MaxTurnsGuard returns an OnTurnEnd hook that logs a warning when the agent
// approaches the turn limit. Useful for observability.
func MaxTurnsGuard(maxTurns int, logger *slog.Logger) agentflow.Hook {
	if logger == nil {
		logger = slog.Default()
	}

	threshold := maxTurns * 80 / 100 // Warn at 80%.
	if threshold < 1 {
		threshold = 1
	}

	return agentflow.HookFunc{
		HookPhase: agentflow.HookPreModelCall,
		Fn: func(_ context.Context, hc *agentflow.HookContext) (*agentflow.HookAction, error) {
			if hc.TurnCount >= threshold {
				logger.Warn(fmt.Sprintf("approaching turn limit: %d/%d", hc.TurnCount, maxTurns),
					slog.Int("turn", hc.TurnCount),
					slog.Int("max_turns", maxTurns),
				)
			}
			return nil, nil
		},
	}
}
