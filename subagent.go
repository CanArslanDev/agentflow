package agentflow

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
)

// SubAgentConfig configures a child agent spawned by a parent. Fields left at
// zero values inherit from the parent agent.
type SubAgentConfig struct {
	// Provider overrides the parent's provider. nil inherits the parent's.
	Provider Provider

	// Tools overrides the parent's tool set. nil inherits the parent's tools.
	// Use an empty slice to spawn a child with no tools.
	Tools []Tool

	// SystemPrompt overrides the parent's system prompt for the child.
	SystemPrompt string

	// MaxTurns limits the child's loop iterations. Zero inherits the parent's.
	MaxTurns int

	// MaxTokens overrides the response token limit. Zero inherits.
	MaxTokens int

	// MaxConcurrency overrides parallel tool execution limit. Zero inherits.
	MaxConcurrency int

	// Hooks are additional hooks for the child. Parent hooks are NOT inherited
	// to keep child execution isolated.
	Hooks []Hook

	// Permission overrides the parent's permission checker. nil inherits.
	Permission PermissionChecker
}

// SpawnChild creates and runs a child agent with the given configuration and task.
// The child runs in a new goroutine and returns an event channel, just like Run().
//
// The child agent inherits unset fields from the parent. The task string becomes
// the initial user message for the child's conversation.
//
// Cancel the context to stop the child and all its in-flight tool executions.
//
//	events := agent.SpawnChild(ctx, agentflow.SubAgentConfig{
//	    SystemPrompt: "You are a research specialist.",
//	    MaxTurns:     5,
//	}, "Find information about Go concurrency patterns")
//
//	for ev := range events {
//	    // handle child events
//	}
func (a *Agent) SpawnChild(ctx context.Context, cfg SubAgentConfig, task string) <-chan Event {
	child := a.buildChild(cfg)
	messages := []Message{NewUserMessage(task)}
	return child.Run(ctx, messages)
}

// SpawnChildren launches multiple child agents in parallel and merges their event
// streams into a single channel. Each child receives its own task string. Events
// are tagged with the child index via EventSubAgentStart/End events.
//
// All children share the same context — canceling it stops all of them.
func (a *Agent) SpawnChildren(ctx context.Context, cfg SubAgentConfig, tasks []string) <-chan Event {
	merged := make(chan Event, DefaultEventBufferSize)

	go func() {
		defer close(merged)

		var wg sync.WaitGroup
		for i, task := range tasks {
			wg.Add(1)
			go func(idx int, t string) {
				defer wg.Done()

				merged <- Event{
					Type: EventSubAgentStart,
					SubAgentStart: &SubAgentStartEvent{
						Index: idx,
						Task:  t,
					},
				}

				child := a.buildChild(cfg)
				var finalMessages []Message
				for ev := range child.Run(ctx, []Message{NewUserMessage(t)}) {
					// Forward events with child index tagging.
					ev.SubAgentIndex = idx
					merged <- ev

					if ev.Type == EventTurnEnd && ev.TurnEnd != nil {
						finalMessages = ev.TurnEnd.Messages
					}
				}

				// Extract the final assistant text as the child's result.
				result := ""
				if len(finalMessages) > 0 {
					last := finalMessages[len(finalMessages)-1]
					if last.Role == RoleAssistant {
						result = last.TextContent()
					}
				}

				merged <- Event{
					Type: EventSubAgentEnd,
					SubAgentEnd: &SubAgentEndEvent{
						Index:  idx,
						Task:   t,
						Result: result,
					},
				}
			}(i, task)
		}
		wg.Wait()
	}()

	return merged
}

// buildChild creates a child Agent from parent settings merged with SubAgentConfig.
func (a *Agent) buildChild(cfg SubAgentConfig) *Agent {
	provider := cfg.Provider
	if provider == nil {
		provider = a.provider
	}

	permission := cfg.Permission
	if permission == nil {
		permission = a.permission
	}

	maxTurns := cfg.MaxTurns
	if maxTurns == 0 {
		maxTurns = a.config.MaxTurns
	}

	maxTokens := cfg.MaxTokens
	if maxTokens == 0 {
		maxTokens = a.config.MaxTokens
	}

	maxConcurrency := cfg.MaxConcurrency
	if maxConcurrency == 0 {
		maxConcurrency = a.config.MaxConcurrency
	}

	systemPrompt := cfg.SystemPrompt
	if systemPrompt == "" {
		systemPrompt = a.config.SystemPrompt
	}

	childTools := make(map[string]Tool)
	if cfg.Tools != nil {
		for _, t := range cfg.Tools {
			childTools[t.Name()] = t
		}
	} else {
		for k, v := range a.tools {
			childTools[k] = v
		}
	}

	return &Agent{
		provider:   provider,
		tools:      childTools,
		hooks:      cfg.Hooks,
		permission: permission,
		compactor:  a.compactor,
		config: Config{
			MaxTurns:        maxTurns,
			MaxConcurrency:  maxConcurrency,
			SystemPrompt:    systemPrompt,
			Temperature:     a.config.Temperature,
			MaxTokens:       maxTokens,
			RetryPolicy:     a.config.RetryPolicy,
			EventBufferSize: a.config.EventBufferSize,
			OnEvent:         a.config.OnEvent,
		},
	}
}

// --- Sub-Agent as a Tool ---

// SubAgentTool creates a Tool that allows the AI model to spawn sub-agents.
// When the model calls this tool, a child agent is created to handle the
// delegated task, and its final response is returned as the tool result.
//
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithTools(
//	        agentflow.SubAgentTool(provider, "You are a researcher.", 5),
//	    ),
//	)
func SubAgentTool(provider Provider, systemPrompt string, maxTurns int) Tool {
	return &subAgentTool{
		provider:     provider,
		systemPrompt: systemPrompt,
		maxTurns:     maxTurns,
	}
}

type subAgentTool struct {
	provider     Provider
	systemPrompt string
	maxTurns     int
}

func (t *subAgentTool) Name() string { return "delegate_task" }

func (t *subAgentTool) Description() string {
	return "Delegate a task to a specialized sub-agent. The sub-agent will work independently and return its findings. Use this for complex sub-tasks that benefit from focused attention."
}

func (t *subAgentTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"task": map[string]any{
				"type":        "string",
				"description": "A clear, detailed description of the task to delegate to the sub-agent.",
			},
		},
		"required": []string{"task"},
	}
}

func (t *subAgentTool) Execute(ctx context.Context, input json.RawMessage, progress ProgressFunc) (*ToolResult, error) {
	var params struct {
		Task string `json:"task"`
	}
	if err := json.Unmarshal(input, &params); err != nil {
		return &ToolResult{Content: "invalid input: " + err.Error(), IsError: true}, nil
	}

	if params.Task == "" {
		return &ToolResult{Content: "task description is required", IsError: true}, nil
	}

	if progress != nil {
		progress(ProgressEvent{Message: "Spawning sub-agent for: " + params.Task})
	}

	child := NewAgent(t.provider,
		WithSystemPrompt(t.systemPrompt),
		WithMaxTurns(t.maxTurns),
	)

	messages, err := child.RunSync(ctx, []Message{NewUserMessage(params.Task)})
	if err != nil {
		return &ToolResult{Content: "sub-agent failed: " + err.Error(), IsError: true}, nil
	}

	// Extract the final assistant response.
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == RoleAssistant {
			text := messages[i].TextContent()
			if text != "" {
				return &ToolResult{Content: text}, nil
			}
		}
	}

	return &ToolResult{Content: "sub-agent completed but produced no text response", IsError: true}, nil
}

func (t *subAgentTool) IsConcurrencySafe(_ json.RawMessage) bool { return true }
func (t *subAgentTool) IsReadOnly(_ json.RawMessage) bool        { return true }

// --- Orchestrator: Parallel Sub-Agent Execution ---

// Orchestrate runs multiple tasks in parallel using sub-agents and collects results.
// This is a high-level utility for common fan-out/fan-in patterns.
//
//	results := agentflow.Orchestrate(ctx, agent, agentflow.SubAgentConfig{
//	    SystemPrompt: "You are a research assistant.",
//	    MaxTurns:     5,
//	}, []string{
//	    "Research Go concurrency patterns",
//	    "Research Go error handling best practices",
//	    "Research Go testing strategies",
//	})
//
//	for _, r := range results {
//	    fmt.Println(r.Task, "→", r.Result)
//	}
func Orchestrate(ctx context.Context, parent *Agent, cfg SubAgentConfig, tasks []string) []OrchestrateResult {
	results := make([]OrchestrateResult, len(tasks))
	var completed int64

	var wg sync.WaitGroup
	for i, task := range tasks {
		wg.Add(1)
		go func(idx int, t string) {
			defer wg.Done()

			child := parent.buildChild(cfg)
			messages, err := child.RunSync(ctx, []Message{NewUserMessage(t)})

			r := OrchestrateResult{
				Index: idx,
				Task:  t,
			}

			if err != nil {
				r.Error = err
			} else {
				for j := len(messages) - 1; j >= 0; j-- {
					if messages[j].Role == RoleAssistant {
						r.Result = messages[j].TextContent()
						break
					}
				}
			}

			results[idx] = r
			done := atomic.AddInt64(&completed, 1)
			_ = done
		}(i, task)
	}

	wg.Wait()
	return results
}

// OrchestrateResult holds the outcome of a single sub-agent task.
type OrchestrateResult struct {
	Index  int
	Task   string
	Result string
	Error  error
}

func (r OrchestrateResult) String() string {
	if r.Error != nil {
		return fmt.Sprintf("[%d] %s → error: %v", r.Index, r.Task, r.Error)
	}
	if len(r.Result) > 100 {
		return fmt.Sprintf("[%d] %s → %s...", r.Index, r.Task, r.Result[:100])
	}
	return fmt.Sprintf("[%d] %s → %s", r.Index, r.Task, r.Result)
}
