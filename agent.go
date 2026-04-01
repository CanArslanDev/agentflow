package agentflow

import (
	"context"
	"errors"
	"io"
	"sync"
	"time"
)

// Agent orchestrates the agentic loop: model calls, tool execution, and message
// management. It is the primary entry point for consumers of the framework.
//
// An Agent is immutable after construction — configuration is set via Option
// functions passed to NewAgent. The same Agent can be used for multiple concurrent
// Run calls, each with its own independent conversation state.
type Agent struct {
	provider     Provider
	tools        map[string]Tool
	hooks        []Hook
	permission   PermissionChecker
	compactor    Compactor
	sessionStore SessionStore
	config       Config
}

// NewAgent creates an Agent with the given provider and options. The provider
// handles communication with the AI model, while options configure tools,
// hooks, permissions, and loop behavior.
//
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithTools(searchTool, calcTool),
//	    agentflow.WithSystemPrompt("You are helpful."),
//	    agentflow.WithMaxTurns(10),
//	)
func NewAgent(provider Provider, opts ...Option) *Agent {
	a := &Agent{
		provider:   provider,
		tools:      make(map[string]Tool),
		permission: AllowAll(),
		config: Config{
			MaxConcurrency:  DefaultMaxConcurrency,
			EventBufferSize: DefaultEventBufferSize,
		},
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// Run executes the agentic loop asynchronously. It returns a channel that delivers
// events as they occur: streaming text, tool invocations, progress updates, and
// completion signals. The channel is closed when the loop terminates.
//
// Cancel the context to stop the loop gracefully. In-flight tool executions will
// receive the cancellation signal via their context.
//
//	for ev := range agent.Run(ctx, messages) {
//	    switch ev.Type {
//	    case agentflow.EventTextDelta:
//	        fmt.Print(ev.TextDelta.Text)
//	    case agentflow.EventTurnEnd:
//	        fmt.Println("Done:", ev.TurnEnd.Reason)
//	    }
//	}
func (a *Agent) Run(ctx context.Context, messages []Message) <-chan Event {
	bufSize := a.config.EventBufferSize
	if bufSize <= 0 {
		bufSize = DefaultEventBufferSize
	}
	events := make(chan Event, bufSize)

	go func() {
		defer close(events)
		a.runLoop(ctx, messages, events)
	}()

	return events
}

// RunSync is a convenience wrapper that collects all events and returns the final
// conversation history. Useful for non-streaming use cases or testing.
//
// Returns the complete message history including all assistant and tool result
// messages generated during the loop. Returns an error if the loop terminated
// due to an unrecoverable error.
func (a *Agent) RunSync(ctx context.Context, messages []Message) ([]Message, error) {
	var finalMessages []Message
	var lastErr error

	for ev := range a.Run(ctx, messages) {
		switch ev.Type {
		case EventTurnEnd:
			if ev.TurnEnd != nil {
				finalMessages = ev.TurnEnd.Messages
				if ev.TurnEnd.Reason == TurnEndError {
					lastErr = ErrProviderUnavailable
				}
			}
		case EventError:
			if ev.Error != nil && !ev.Error.Retrying {
				lastErr = ev.Error.Err
			}
		}
	}

	if finalMessages == nil {
		finalMessages = messages
	}
	return finalMessages, lastErr
}

// RunSession executes the agentic loop with automatic session persistence.
// The session is saved after each turn completes. If the session has an empty
// ID, one is generated automatically.
//
// Use this instead of Run when you need crash recovery and conversation history.
//
//	session := &agentflow.Session{
//	    Metadata: map[string]any{"user": "alice"},
//	}
//	for ev := range agent.RunSession(ctx, session, messages) { ... }
//	// session.ID is now set and can be used with Resume()
func (a *Agent) RunSession(ctx context.Context, session *Session, messages []Message) <-chan Event {
	if session.ID == "" {
		session.ID = GenerateSessionID()
	}
	if session.CreatedAt.IsZero() {
		session.CreatedAt = time.Now().UTC()
	}
	session.ModelID = a.provider.ModelID()

	bufSize := a.config.EventBufferSize
	if bufSize <= 0 {
		bufSize = DefaultEventBufferSize
	}
	events := make(chan Event, bufSize)

	go func() {
		defer close(events)

		// Wrap the event channel to intercept TurnEnd for auto-save.
		innerEvents := make(chan Event, bufSize)
		go func() {
			defer close(innerEvents)
			a.runLoop(ctx, messages, innerEvents)
		}()

		for ev := range innerEvents {
			// Auto-save on every TurnEnd if session store is configured.
			if ev.Type == EventTurnEnd && ev.TurnEnd != nil && a.sessionStore != nil {
				session.Messages = ev.TurnEnd.Messages
				session.TurnCount = ev.TurnEnd.TurnNumber
				session.UpdatedAt = time.Now().UTC()
				a.sessionStore.Save(ctx, session)
			}
			events <- ev
		}
	}()

	return events
}

// Resume loads a previously saved session and continues the conversation.
// The agent picks up where it left off, using the stored message history.
//
// If additionalMessage is non-empty, it is appended as a new user message
// before the loop resumes.
//
//	for ev := range agent.Resume(ctx, "session-id-123", "Continue from where you left off") { ... }
func (a *Agent) Resume(ctx context.Context, sessionID string, additionalMessage string) (<-chan Event, error) {
	if a.sessionStore == nil {
		return nil, errors.New("agentflow: session store not configured, use WithSessionStore")
	}

	session, err := a.sessionStore.Load(ctx, sessionID)
	if err != nil {
		return nil, err
	}

	messages := session.Messages
	if additionalMessage != "" {
		messages = append(messages, NewUserMessage(additionalMessage))
	}

	return a.RunSession(ctx, session, messages), nil
}

// Tools returns the registered tool names.
func (a *Agent) Tools() []string {
	names := make([]string, 0, len(a.tools))
	for name := range a.tools {
		names = append(names, name)
	}
	return names
}

// toolDefinitions builds the list of ToolDefinition values sent to the provider.
// In ModeRemote, only remote-safe tools are included — the model never sees
// local-only tools and therefore cannot attempt to call them.
func (a *Agent) toolDefinitions() []ToolDefinition {
	defs := make([]ToolDefinition, 0, len(a.tools))
	for _, t := range a.tools {
		if IsToolAllowed(t, a.config.ExecutionMode) {
			defs = append(defs, toolDefinitionFrom(t))
		}
	}
	return defs
}

// emit sends an event to both the callback (if set) and the channel.
func (a *Agent) emit(events chan<- Event, ev Event) {
	if a.config.OnEvent != nil {
		a.config.OnEvent(ev)
	}
	events <- ev
}

// runLoop is the core agentic loop. It runs in a dedicated goroutine launched by Run.
func (a *Agent) runLoop(ctx context.Context, messages []Message, events chan<- Event) {
	state := &loopState{
		messages: make([]Message, len(messages)),
		metadata: make(map[string]any),
	}
	copy(state.messages, messages)

	var lastUsage *Usage
	bt := newBudgetTracker(a.config.TokenBudget)

	for {
		// Check context cancellation.
		if ctx.Err() != nil {
			a.emit(events, Event{
				Type:    EventTurnEnd,
				TurnEnd: &TurnEndEvent{TurnNumber: state.turnCount, Reason: TurnEndAborted, Messages: state.messages},
			})
			return
		}

		// Check turn limit.
		state.turnCount++
		if a.config.MaxTurns > 0 && state.turnCount > a.config.MaxTurns {
			a.emit(events, Event{
				Type:    EventTurnEnd,
				TurnEnd: &TurnEndEvent{TurnNumber: state.turnCount - 1, Reason: TurnEndMaxTurns, Messages: state.messages},
			})
			return
		}

		a.emit(events, Event{
			Type:      EventTurnStart,
			TurnStart: &TurnStartEvent{TurnNumber: state.turnCount},
		})

		// Context compaction.
		if a.compactor != nil && a.compactor.ShouldCompact(state.messages, lastUsage) {
			compacted, err := a.compactor.Compact(ctx, state.messages)
			if err == nil {
				state.messages = compacted
			}
		}

		// Pre-model-call hooks.
		if blocked := a.runHooks(ctx, events, HookPreModelCall, &HookContext{
			Phase:     HookPreModelCall,
			Messages:  state.messages,
			TurnCount: state.turnCount,
			Metadata:  state.metadata,
		}, state); blocked {
			a.emit(events, Event{
				Type:    EventTurnEnd,
				TurnEnd: &TurnEndEvent{TurnNumber: state.turnCount, Reason: TurnEndHookBlock, Messages: state.messages},
			})
			return
		}

		// Build and send the model request.
		req := &Request{
			Messages:      state.messages,
			SystemPrompt:  a.config.SystemPrompt,
			Tools:         a.toolDefinitions(),
			MaxTokens:     a.config.MaxTokens,
			Temperature:   a.config.Temperature,
			StopSequences: nil,
		}

		stream, err := a.createStreamWithRetry(ctx, req)
		if err != nil {
			a.emit(events, Event{
				Type:  EventError,
				Error: &ErrorEvent{Err: err, Retrying: false, TurnCount: state.turnCount},
			})
			a.emit(events, Event{
				Type:    EventTurnEnd,
				TurnEnd: &TurnEndEvent{TurnNumber: state.turnCount, Reason: TurnEndError, Messages: state.messages},
			})
			return
		}

		// Consume the stream: emit text deltas, collect tool calls.
		assistantMsg, toolCalls, err := a.consumeStream(ctx, stream, events)
		stream.Close()
		if usage := stream.Usage(); usage != nil {
			lastUsage = usage
			a.emit(events, Event{
				Type:  EventUsage,
				Usage: &UsageEvent{Usage: *usage, TurnCount: state.turnCount},
			})

			// Token budget enforcement.
			if bt != nil {
				exhausted := bt.record(usage)
				if bt.shouldWarn() {
					a.emit(events, Event{
						Type: EventBudgetWarning,
						BudgetWarning: &BudgetWarningEvent{
							ConsumedTokens: bt.totalConsumed(),
							MaxTokens:      bt.budget.MaxTokens,
							Percentage:     float64(bt.totalConsumed()) / float64(bt.budget.MaxTokens),
						},
					})
				}
				if exhausted {
					// Append the assistant message before terminating so it's not lost.
					if len(assistantMsg.Content) > 0 {
						state.messages = append(state.messages, assistantMsg)
					}
					a.emit(events, Event{
						Type:    EventTurnEnd,
						TurnEnd: &TurnEndEvent{TurnNumber: state.turnCount, Reason: TurnEndBudgetExhausted, Messages: state.messages},
					})
					return
				}
			}
		}

		if err != nil {
			a.emit(events, Event{
				Type:  EventError,
				Error: &ErrorEvent{Err: err, Retrying: false, TurnCount: state.turnCount},
			})
			a.emit(events, Event{
				Type:    EventTurnEnd,
				TurnEnd: &TurnEndEvent{TurnNumber: state.turnCount, Reason: TurnEndError, Messages: state.messages},
			})
			return
		}

		// Append the assistant message to history.
		state.messages = append(state.messages, assistantMsg)
		a.emit(events, Event{Type: EventMessage, Message: &assistantMsg})

		// Post-model-call hooks.
		a.runHooks(ctx, events, HookPostModelCall, &HookContext{
			Phase:     HookPostModelCall,
			Messages:  state.messages,
			TurnCount: state.turnCount,
			Metadata:  state.metadata,
		}, state)

		// Check if tool calls are needed.
		if len(toolCalls) == 0 {
			// No tool calls — check turn-end hooks for possible continuation.
			shouldContinue := false
			for _, hook := range a.hooksForPhase(HookOnTurnEnd) {
				action, err := hook.Execute(ctx, &HookContext{
					Phase:     HookOnTurnEnd,
					Messages:  state.messages,
					TurnCount: state.turnCount,
					Metadata:  state.metadata,
				})
				if err != nil {
					continue
				}
				if action != nil && len(action.InjectMessages) > 0 {
					state.messages = append(state.messages, action.InjectMessages...)
					shouldContinue = true
					break
				}
			}
			if !shouldContinue {
				a.emit(events, Event{
					Type:    EventTurnEnd,
					TurnEnd: &TurnEndEvent{TurnNumber: state.turnCount, Reason: TurnEndComplete, Messages: state.messages},
				})
				return
			}
			continue
		}

		// Execute tools.
		results := a.executeTools(ctx, toolCalls, state, events)

		// Build and append the tool results message.
		toolResultMsg := newToolResultMessage(results)
		state.messages = append(state.messages, toolResultMsg)
		a.emit(events, Event{Type: EventMessage, Message: &toolResultMsg})
	}
}

// consumeStream reads all events from the provider stream, emitting text deltas
// to the event channel and collecting tool calls. Returns the complete assistant
// message and extracted tool calls.
func (a *Agent) consumeStream(ctx context.Context, stream Stream, events chan<- Event) (Message, []ToolCall, error) {
	var textParts []string
	var toolCalls []ToolCall
	var blocks []ContentBlock

	for {
		if ctx.Err() != nil {
			return Message{}, nil, ctx.Err()
		}

		ev, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return Message{}, nil, err
		}

		switch ev.Type {
		case StreamEventDelta:
			if ev.Delta != nil {
				a.emit(events, Event{
					Type:      EventTextDelta,
					TextDelta: &TextDeltaEvent{Text: ev.Delta.Text},
				})
				textParts = append(textParts, ev.Delta.Text)
			}

		case StreamEventThinkingDelta:
			if ev.ThinkingDelta != nil {
				a.emit(events, Event{
					Type:       EventThinkingDelta,
					ThinkDelta: &TextDeltaEvent{Text: ev.ThinkingDelta.Text},
				})
			}

		case StreamEventToolCall:
			if ev.ToolCall != nil {
				toolCalls = append(toolCalls, *ev.ToolCall)
				blocks = append(blocks, ContentBlock{
					Type:     ContentToolCall,
					ToolCall: ev.ToolCall,
				})
			}

		case StreamEventError:
			return Message{}, nil, ev.Error
		}
	}

	// Build the complete text block if any text was accumulated.
	if len(textParts) > 0 {
		fullText := ""
		for _, part := range textParts {
			fullText += part
		}
		textBlock := ContentBlock{Type: ContentText, Text: fullText}
		blocks = append([]ContentBlock{textBlock}, blocks...)
	}

	msg := Message{Role: RoleAssistant, Content: blocks}
	return msg, toolCalls, nil
}

// executeTools runs all tool calls through the execution pipeline. Tools marked
// as concurrency-safe are batched and run in parallel; others run sequentially.
func (a *Agent) executeTools(ctx context.Context, calls []ToolCall, state *loopState, events chan<- Event) []toolExecResult {
	batches := partitionToolCalls(calls, a.tools)
	var allResults []toolExecResult

	for _, batch := range batches {
		if batch.concurrent && len(batch.calls) > 1 {
			results := a.runConcurrentBatch(ctx, batch.calls, state, events)
			allResults = append(allResults, results...)
		} else {
			for _, call := range batch.calls {
				result := a.executeSingleTool(ctx, call, state, events)
				allResults = append(allResults, result)
			}
		}
	}

	return allResults
}

// runConcurrentBatch executes concurrency-safe tools in parallel with a semaphore.
func (a *Agent) runConcurrentBatch(ctx context.Context, calls []ToolCall, state *loopState, events chan<- Event) []toolExecResult {
	batchCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	results := make([]toolExecResult, len(calls))
	sem := make(chan struct{}, a.config.MaxConcurrency)
	var wg sync.WaitGroup

	for i, call := range calls {
		wg.Add(1)
		go func(idx int, tc ToolCall) {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-batchCtx.Done():
				results[idx] = toolExecResult{
					callID: tc.ID,
					result: &ToolResult{Content: "execution cancelled: sibling tool error", IsError: true},
				}
				return
			}
			results[idx] = a.executeSingleTool(batchCtx, tc, state, events)
		}(i, call)
	}

	wg.Wait()
	return results
}

// executeSingleTool runs a single tool through the full pipeline:
// validate → pre-hooks → permission → execute → post-hooks.
func (a *Agent) executeSingleTool(ctx context.Context, call ToolCall, state *loopState, events chan<- Event) toolExecResult {
	// emitEarlyReturn is a helper that emits ToolStart + ToolEnd for cases where
	// the tool never actually executes (unknown, blocked, denied).
	emitEarlyReturn := func(result *ToolResult) toolExecResult {
		a.emit(events, Event{
			Type:      EventToolStart,
			ToolStart: &ToolStartEvent{ToolCall: call},
		})
		a.emit(events, Event{
			Type: EventToolEnd,
			ToolEnd: &ToolEndEvent{
				ToolCall: call,
				Result:   *result,
				Duration: 0,
			},
		})
		return toolExecResult{callID: call.ID, result: result}
	}

	tool, ok := a.tools[call.Name]
	if !ok {
		return emitEarlyReturn(&ToolResult{
			Content: "Error: unknown tool \"" + call.Name + "\"",
			IsError: true,
		})
	}

	// Execution mode guard — block local-only tools in remote mode.
	if !IsToolAllowed(tool, a.config.ExecutionMode) {
		return emitEarlyReturn(&ToolResult{
			Content: "Error: tool \"" + call.Name + "\" is not available in remote execution mode",
			IsError: true,
		})
	}

	// Pre-tool hooks.
	currentCall := call
	for _, hook := range a.hooksForPhase(HookPreToolUse) {
		action, err := hook.Execute(ctx, &HookContext{
			Phase:     HookPreToolUse,
			ToolCall:  &currentCall,
			Messages:  state.messages,
			TurnCount: state.turnCount,
			Metadata:  state.metadata,
		})
		if err != nil {
			return emitEarlyReturn(&ToolResult{Content: "hook error: " + err.Error(), IsError: true})
		}
		if action != nil {
			if action.Block {
				reason := action.BlockReason
				if reason == "" {
					reason = "blocked by hook"
				}
				return emitEarlyReturn(&ToolResult{Content: reason, IsError: true})
			}
			if action.ModifiedInput != nil {
				currentCall.Input = action.ModifiedInput
			}
		}
	}

	// Permission check.
	perm, err := a.permission.Check(ctx, &currentCall, tool)
	if err != nil {
		return emitEarlyReturn(&ToolResult{Content: "permission check error: " + err.Error(), IsError: true})
	}
	if perm == PermissionDeny {
		return emitEarlyReturn(&ToolResult{Content: "Permission denied for tool \"" + call.Name + "\"", IsError: true})
	}

	// Execute.
	a.emit(events, Event{
		Type:      EventToolStart,
		ToolStart: &ToolStartEvent{ToolCall: call},
	})
	start := time.Now()

	progressFn := func(pe ProgressEvent) {
		pe.ToolCallID = call.ID
		a.emit(events, Event{Type: EventToolProgress, ToolProgress: &pe})
	}

	result, err := a.callToolWithRecovery(ctx, tool, currentCall.Input, progressFn)
	duration := time.Since(start)

	if err != nil {
		result = &ToolResult{Content: err.Error(), IsError: true}
	}

	// Apply result size limiting.
	result = a.limitResult(result)

	a.emit(events, Event{
		Type: EventToolEnd,
		ToolEnd: &ToolEndEvent{
			ToolCall: call,
			Result:   *result,
			Duration: duration,
		},
	})

	// Post-tool hooks.
	for _, hook := range a.hooksForPhase(HookPostToolUse) {
		hook.Execute(ctx, &HookContext{
			Phase:      HookPostToolUse,
			ToolCall:   &call,
			ToolResult: result,
			Messages:   state.messages,
			TurnCount:  state.turnCount,
			Metadata:   state.metadata,
		})
	}

	return toolExecResult{callID: call.ID, result: result}
}

// callToolWithRecovery calls tool.Execute with panic recovery.
func (a *Agent) callToolWithRecovery(ctx context.Context, tool Tool, input []byte, progress ProgressFunc) (result *ToolResult, err error) {
	defer func() {
		if r := recover(); r != nil {
			result = &ToolResult{
				Content: "internal error: tool panicked",
				IsError: true,
			}
			err = nil
		}
	}()
	return tool.Execute(ctx, input, progress)
}

// createStreamWithRetry wraps provider.CreateStream with retry logic.
func (a *Agent) createStreamWithRetry(ctx context.Context, req *Request) (Stream, error) {
	policy := a.config.RetryPolicy
	if policy == nil || policy.MaxRetries <= 0 {
		return a.provider.CreateStream(ctx, req)
	}

	var lastErr error
	for attempt := 0; attempt <= policy.MaxRetries; attempt++ {
		if attempt > 0 {
			delay := policy.BaseDelay * time.Duration(1<<(attempt-1))
			if delay > policy.MaxDelay {
				delay = policy.MaxDelay
			}
			select {
			case <-time.After(delay):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		stream, err := a.provider.CreateStream(ctx, req)
		if err == nil {
			return stream, nil
		}
		lastErr = err
		if !IsRetryableError(err) {
			return nil, err
		}
	}
	return nil, lastErr
}

// runHooks executes all hooks for a given phase. Returns true if any hook blocked.
func (a *Agent) runHooks(ctx context.Context, events chan<- Event, phase HookPhase, hc *HookContext, state *loopState) bool {
	for _, hook := range a.hooksForPhase(phase) {
		action, err := hook.Execute(ctx, hc)
		if err != nil {
			continue
		}
		if action != nil {
			if action.Block {
				return true
			}
			if len(action.InjectMessages) > 0 {
				state.messages = append(state.messages, action.InjectMessages...)
			}
		}
	}
	return false
}

// hooksForPhase returns all registered hooks that fire at the given phase.
func (a *Agent) hooksForPhase(phase HookPhase) []Hook {
	var matched []Hook
	for _, h := range a.hooks {
		if h.Phase() == phase {
			matched = append(matched, h)
		}
	}
	return matched
}

// limitResult applies the configured ResultLimiter to a tool result.
func (a *Agent) limitResult(result *ToolResult) *ToolResult {
	if result == nil || result.IsError {
		return result
	}

	limiter := a.config.ResultLimiter
	if limiter == nil {
		limiter = TruncateLimiter{}
	}

	maxChars := a.config.MaxResultChars
	if maxChars <= 0 {
		maxChars = DefaultMaxResultChars
	}

	return limiter.Limit(result, maxChars)
}

// loopState carries mutable state across agentic loop iterations.
// It is never shared across goroutines.
type loopState struct {
	messages  []Message
	turnCount int
	metadata  map[string]any
}

// toolBatch groups tool calls by their concurrency safety.
type toolBatch struct {
	concurrent bool
	calls      []ToolCall
}

// partitionToolCalls groups consecutive concurrency-safe calls into concurrent
// batches and isolates non-safe calls into serial batches.
func partitionToolCalls(calls []ToolCall, tools map[string]Tool) []toolBatch {
	if len(calls) == 0 {
		return nil
	}

	var batches []toolBatch
	for _, call := range calls {
		tool, ok := tools[call.Name]
		safe := ok && tool.IsConcurrencySafe(call.Input)

		if safe && len(batches) > 0 && batches[len(batches)-1].concurrent {
			batches[len(batches)-1].calls = append(batches[len(batches)-1].calls, call)
		} else {
			batches = append(batches, toolBatch{concurrent: safe, calls: []ToolCall{call}})
		}
	}
	return batches
}
