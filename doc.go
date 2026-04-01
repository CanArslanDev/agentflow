// Package agentflow provides a production-grade framework for building agentic AI systems in Go.
//
// An agentic AI system is one where a language model can autonomously decide to invoke tools,
// observe results, and continue reasoning in a loop until the task is complete. This package
// provides the core abstractions and orchestration logic to build such systems with any AI
// provider that supports tool use (function calling).
//
// # Architecture
//
// The framework is built around five core interfaces:
//
//   - [Provider] abstracts an AI model API (Anthropic, OpenAI, OpenRouter, etc.)
//   - [Tool] defines a capability the agent can invoke (web search, file read, API call, etc.)
//   - [Hook] intercepts the execution pipeline at defined lifecycle phases
//   - [PermissionChecker] controls whether a tool is allowed to execute
//   - [Compactor] manages conversation history when context limits are approached
//
// # The Agentic Loop
//
// The core of the framework is a simple but powerful loop:
//
//	for {
//	    response := provider.CreateStream(messages)
//	    toolCalls := extractToolCalls(response)
//	    if len(toolCalls) == 0 {
//	        break // model is done
//	    }
//	    results := executor.Execute(toolCalls)
//	    messages = append(messages, results...)
//	}
//
// The loop streams events through a channel, allowing real-time observation of every step:
// text deltas, tool invocations, progress updates, and completion signals.
//
// # Quick Start
//
//	provider := openrouter.New("sk-or-...", "anthropic/claude-sonnet-4-20250514")
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithTools(myTool),
//	    agentflow.WithSystemPrompt("You are a helpful assistant."),
//	)
//
//	messages := []agentflow.Message{agentflow.NewUserMessage("Hello")}
//	for ev := range agent.Run(ctx, messages) {
//	    // handle events
//	}
package agentflow
