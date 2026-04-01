// Package builtin provides ready-to-use tools for common agent operations.
//
// These tools cover file system access, command execution, HTTP requests,
// and user interaction. Each tool follows agentflow conventions: proper
// concurrency safety declarations, read-only flags, and structured input schemas.
//
// Register all built-in tools at once:
//
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithTools(builtin.All()...),
//	)
//
// Or pick specific tools:
//
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithTools(
//	        builtin.ReadFile(),
//	        builtin.ListDir(),
//	        builtin.Bash(),
//	    ),
//	)
package builtin
