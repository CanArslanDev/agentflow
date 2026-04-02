package agentflow

import (
	"context"
	"sync"
	"time"
)

// Trigger defines a scheduled agent execution. Triggers run on a fixed interval
// and execute an agent with a predefined task.
type Trigger struct {
	// ID uniquely identifies this trigger.
	ID string

	// Interval between executions.
	Interval time.Duration

	// Task is the user message sent to the agent on each execution.
	Task string

	// Agent configuration for each run.
	Provider     Provider
	Tools        []Tool
	SystemPrompt string
	MaxTurns     int
	MaxTokens    int

	// OnResult is called with the agent's text response after each execution.
	// Runs in a separate goroutine — must be safe for concurrent use.
	OnResult func(TriggerResult)
}

// TriggerResult holds the outcome of a single trigger execution.
type TriggerResult struct {
	TriggerID string
	Timestamp time.Time
	Response  string
	Error     error
	Duration  time.Duration
}

// TriggerScheduler manages multiple triggers running on intervals.
type TriggerScheduler struct {
	mu       sync.Mutex
	triggers map[string]*runningTrigger
}

type runningTrigger struct {
	trigger Trigger
	cancel  context.CancelFunc
}

// NewTriggerScheduler creates an empty scheduler.
func NewTriggerScheduler() *TriggerScheduler {
	return &TriggerScheduler{
		triggers: make(map[string]*runningTrigger),
	}
}

// Schedule registers and starts a trigger. If a trigger with the same ID
// already exists, it is stopped and replaced.
func (s *TriggerScheduler) Schedule(trigger Trigger) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Stop existing trigger with same ID.
	if existing, ok := s.triggers[trigger.ID]; ok {
		existing.cancel()
	}

	ctx, cancel := context.WithCancel(context.Background())
	rt := &runningTrigger{trigger: trigger, cancel: cancel}
	s.triggers[trigger.ID] = rt

	go s.run(ctx, rt)
}

// Cancel stops a trigger by ID.
func (s *TriggerScheduler) Cancel(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if rt, ok := s.triggers[id]; ok {
		rt.cancel()
		delete(s.triggers, id)
	}
}

// CancelAll stops all triggers.
func (s *TriggerScheduler) CancelAll() {
	s.mu.Lock()
	defer s.mu.Unlock()

	for id, rt := range s.triggers {
		rt.cancel()
		delete(s.triggers, id)
	}
}

// List returns all active trigger IDs.
func (s *TriggerScheduler) List() []string {
	s.mu.Lock()
	defer s.mu.Unlock()

	ids := make([]string, 0, len(s.triggers))
	for id := range s.triggers {
		ids = append(ids, id)
	}
	return ids
}

// run executes the trigger on its interval until cancelled.
func (s *TriggerScheduler) run(ctx context.Context, rt *runningTrigger) {
	trigger := rt.trigger
	ticker := time.NewTicker(trigger.Interval)
	defer ticker.Stop()

	// Execute immediately on first run.
	s.executeTrigger(ctx, trigger)

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.executeTrigger(ctx, trigger)
		}
	}
}

// executeTrigger runs a single trigger execution.
func (s *TriggerScheduler) executeTrigger(ctx context.Context, trigger Trigger) {
	start := time.Now()

	maxTurns := trigger.MaxTurns
	if maxTurns == 0 {
		maxTurns = 5
	}
	maxTokens := trigger.MaxTokens
	if maxTokens == 0 {
		maxTokens = 1024
	}

	agent := NewAgent(trigger.Provider,
		WithTools(trigger.Tools...),
		WithSystemPrompt(trigger.SystemPrompt),
		WithMaxTurns(maxTurns),
		WithMaxTokens(maxTokens),
	)

	execCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()

	messages, err := agent.RunSync(execCtx, []Message{NewUserMessage(trigger.Task)})

	result := TriggerResult{
		TriggerID: trigger.ID,
		Timestamp: time.Now(),
		Duration:  time.Since(start),
		Error:     err,
	}

	if err == nil && len(messages) > 0 {
		for i := len(messages) - 1; i >= 0; i-- {
			if messages[i].Role == RoleAssistant {
				result.Response = messages[i].TextContent()
				break
			}
		}
	}

	if trigger.OnResult != nil {
		trigger.OnResult(result)
	}
}
