package agentflow

import "context"

// PermissionResult is the outcome of a permission check.
type PermissionResult int

const (
	// PermissionAllow permits the tool execution to proceed.
	PermissionAllow PermissionResult = iota

	// PermissionDeny blocks the tool execution. The model receives an error
	// message indicating the tool was denied.
	PermissionDeny

	// PermissionAsk indicates an external decision is needed (e.g., prompting
	// the user). The framework calls the AskFunc configured on the Agent.
	PermissionAsk
)

// PermissionChecker controls whether a tool is allowed to execute for a given
// invocation. Implementations can be static rule-based, interactive, or policy-driven.
type PermissionChecker interface {
	// Check returns the permission decision for a tool invocation. The checker
	// receives both the call details and the tool definition, enabling fine-grained
	// decisions based on the specific input.
	Check(ctx context.Context, call *ToolCall, tool Tool) (PermissionResult, error)
}

// PermissionFunc is an adapter that allows ordinary functions to be used as
// permission checkers, similar to http.HandlerFunc.
type PermissionFunc func(ctx context.Context, call *ToolCall, tool Tool) (PermissionResult, error)

// Check delegates to the wrapped function.
func (f PermissionFunc) Check(ctx context.Context, call *ToolCall, tool Tool) (PermissionResult, error) {
	return f(ctx, call, tool)
}

// AllowAll returns a PermissionChecker that permits every tool invocation.
// Use in trusted environments where all tools are safe to execute.
func AllowAll() PermissionChecker {
	return PermissionFunc(func(_ context.Context, _ *ToolCall, _ Tool) (PermissionResult, error) {
		return PermissionAllow, nil
	})
}

// DenyAll returns a PermissionChecker that blocks every tool invocation.
func DenyAll() PermissionChecker {
	return PermissionFunc(func(_ context.Context, _ *ToolCall, _ Tool) (PermissionResult, error) {
		return PermissionDeny, nil
	})
}

// DenyList returns a PermissionChecker that denies tools whose names appear
// in the provided list. All other tools are allowed.
func DenyList(names ...string) PermissionChecker {
	denied := make(map[string]struct{}, len(names))
	for _, name := range names {
		denied[name] = struct{}{}
	}
	return PermissionFunc(func(_ context.Context, call *ToolCall, _ Tool) (PermissionResult, error) {
		if _, ok := denied[call.Name]; ok {
			return PermissionDeny, nil
		}
		return PermissionAllow, nil
	})
}

// AllowList returns a PermissionChecker that allows only tools whose names
// appear in the provided list. All other tools are denied.
func AllowList(names ...string) PermissionChecker {
	allowed := make(map[string]struct{}, len(names))
	for _, name := range names {
		allowed[name] = struct{}{}
	}
	return PermissionFunc(func(_ context.Context, call *ToolCall, _ Tool) (PermissionResult, error) {
		if _, ok := allowed[call.Name]; ok {
			return PermissionAllow, nil
		}
		return PermissionDeny, nil
	})
}

// ReadOnlyPermission returns a PermissionChecker that allows only tools where
// IsReadOnly returns true for the given input. Write operations are denied.
func ReadOnlyPermission() PermissionChecker {
	return PermissionFunc(func(_ context.Context, call *ToolCall, tool Tool) (PermissionResult, error) {
		if tool.IsReadOnly(call.Input) {
			return PermissionAllow, nil
		}
		return PermissionDeny, nil
	})
}

// ChainPermission evaluates multiple checkers in order. The first non-Allow
// result wins. If all checkers return Allow, the final result is Allow.
// An empty chain allows everything.
func ChainPermission(checkers ...PermissionChecker) PermissionChecker {
	return PermissionFunc(func(ctx context.Context, call *ToolCall, tool Tool) (PermissionResult, error) {
		for _, checker := range checkers {
			result, err := checker.Check(ctx, call, tool)
			if err != nil {
				return PermissionDeny, err
			}
			if result != PermissionAllow {
				return result, nil
			}
		}
		return PermissionAllow, nil
	})
}
