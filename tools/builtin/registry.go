package builtin

import (
	"context"

	"github.com/CanArslanDev/agentflow"
)

// All returns all built-in tools regardless of locality. Use with ModeLocal
// where the agent runs on the user's machine.
// The AskUser tool requires a callback — register it separately.
func All() []agentflow.Tool {
	return []agentflow.Tool{
		Bash(),
		ReadFile(),
		WriteFile(),
		EditFile(),
		ListDir(),
		Glob(),
		Grep(),
		HTTPRequest(),
		WebSearch(),
		DeepSearch(),
		Sleep(),
	}
}

// Local is an alias for All(). Explicit intent: these tools are for local execution.
func Local() []agentflow.Tool {
	return All()
}

// Remote returns only tools that are safe for server-side execution.
// These tools do not access the local filesystem or run shell commands.
// Use with ModeRemote for server deployments.
func Remote() []agentflow.Tool {
	return []agentflow.Tool{
		HTTPRequest(),
		WebSearch(),
		DeepSearch(),
		Sleep(),
	}
}

// ReadOnly returns tools that perform no mutations.
func ReadOnly() []agentflow.Tool {
	return []agentflow.Tool{
		ReadFile(),
		ListDir(),
		Glob(),
		Grep(),
		HTTPRequest(),
		WebSearch(),
		DeepSearch(),
		Sleep(),
	}
}

// WithAskUser returns All() tools plus an AskUser tool with the given callback.
func WithAskUser(askFn func(ctx context.Context, question string) (string, error)) []agentflow.Tool {
	return append(All(), AskUser(askFn))
}
