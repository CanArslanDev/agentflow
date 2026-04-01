// Package middleware provides reusable hooks for common cross-cutting concerns
// like logging, metrics collection, and panic recovery.
package middleware

import (
	"context"
	"log/slog"
	"time"

	"github.com/CanArslanDev/agentflow"
)

// Logging returns a pair of hooks that log tool execution start and completion
// using the provided slog.Logger. If logger is nil, slog.Default() is used.
//
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithHook(middleware.Logging(logger)...),
//	)
func Logging(logger *slog.Logger) []agentflow.Hook {
	if logger == nil {
		logger = slog.Default()
	}

	return []agentflow.Hook{
		agentflow.HookFunc{
			HookPhase: agentflow.HookPreToolUse,
			Fn: func(_ context.Context, hc *agentflow.HookContext) (*agentflow.HookAction, error) {
				if hc.ToolCall != nil {
					logger.Info("tool execution starting",
						slog.String("tool", hc.ToolCall.Name),
						slog.String("call_id", hc.ToolCall.ID),
						slog.Int("turn", hc.TurnCount),
					)
					hc.Metadata["tool_start_"+hc.ToolCall.ID] = time.Now()
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

				var duration time.Duration
				if start, ok := hc.Metadata["tool_start_"+hc.ToolCall.ID].(time.Time); ok {
					duration = time.Since(start)
					delete(hc.Metadata, "tool_start_"+hc.ToolCall.ID)
				}

				attrs := []any{
					slog.String("tool", hc.ToolCall.Name),
					slog.String("call_id", hc.ToolCall.ID),
					slog.Duration("duration", duration),
					slog.Bool("is_error", hc.ToolResult.IsError),
					slog.Int("turn", hc.TurnCount),
				}

				if hc.ToolResult.IsError {
					logger.Warn("tool execution failed", attrs...)
				} else {
					logger.Info("tool execution completed", attrs...)
				}

				return nil, nil
			},
		},
	}
}
