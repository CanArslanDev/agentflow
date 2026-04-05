package agentflow

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/CanArslanDev/agentflow/internal/jsonschema"
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

// Clone creates a copy of this agent with the same configuration, then applies
// the given options on top. Useful for deriving specialized agents from a
// common base without SubAgentConfig.
//
//	base := agentflow.NewAgent(provider, agentflow.WithMaxTurns(10))
//	researcher := base.Clone(agentflow.WithSystemPrompt("You are a researcher."))
//	writer := base.Clone(agentflow.WithSystemPrompt("You are a writer."))
func (a *Agent) Clone(opts ...Option) *Agent {
	// Copy tools map.
	toolsCopy := make(map[string]Tool, len(a.tools))
	for k, v := range a.tools {
		toolsCopy[k] = v
	}

	// Copy hooks slice.
	hooksCopy := make([]Hook, len(a.hooks))
	copy(hooksCopy, a.hooks)

	clone := &Agent{
		provider:     a.provider,
		tools:        toolsCopy,
		hooks:        hooksCopy,
		permission:   a.permission,
		compactor:    a.compactor,
		sessionStore: a.sessionStore,
		config:       a.config,
	}

	for _, opt := range opts {
		opt(clone)
	}
	return clone
}

// AddHook appends a lifecycle hook to the agent. This is useful when hooks
// need to be added after initial construction (e.g., by extension packages).
func (a *Agent) AddHook(h Hook) {
	a.hooks = append(a.hooks, h)
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
// Recovers from panic if the channel is already closed (e.g., budget exhaustion
// terminates the loop while a streaming tool executor goroutine is still running).
func (a *Agent) emit(events chan<- Event, ev Event) {
	if a.config.OnEvent != nil {
		a.config.OnEvent(ev)
	}
	// Recover from send-on-closed-channel if the loop has already terminated
	// while a streaming executor goroutine is still finishing.
	defer func() { recover() }()
	events <- ev
}

// runLoop is the core agentic loop. It runs in a dedicated goroutine launched by Run.
func (a *Agent) runLoop(ctx context.Context, messages []Message, events chan<- Event) {
	state := &loopState{
		messages: make([]Message, len(messages)),
		metadata: make(map[string]any),
		done:     make(chan struct{}),
	}
	copy(state.messages, messages)

	defer func() {
		close(state.done)   // Signal streaming executor goroutines to stop.
		state.bgWork.Wait() // Wait for them to finish before returning (caller closes events).
	}()

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
			beforeCount := len(state.messages)
			compacted, err := a.compactor.Compact(ctx, state.messages)
			if err == nil {
				state.messages = compacted
				a.logInfo("context compacted",
					slog.Int("turn", state.turnCount),
					slog.Int("before", beforeCount),
					slog.Int("after", len(compacted)),
				)
				a.emit(events, Event{
					Type: EventCompaction,
					Compaction: &CompactionEvent{
						BeforeCount: beforeCount,
						AfterCount:  len(compacted),
						TurnCount:   state.turnCount,
					},
				})
			} else {
				a.logWarn("compaction failed", slog.Int("turn", state.turnCount), slog.String("error", err.Error()))
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
		// Tools are NEVER sent in Request.Tools. All tool execution is
		// text-based: tool descriptions are injected into the system prompt
		// and the model writes [TOOL_CALL: tool_name("args")] in its response.
		sysPrompt := a.config.SystemPrompt
		toolDefs := a.toolDefinitions()
		if len(toolDefs) > 0 {
			sysPrompt += a.buildToolInstruction()
		}

		req := &Request{
			Messages:      state.messages,
			SystemPrompt:  sysPrompt,
			Tools:         nil, // Never send tools to provider.
			MaxTokens:     a.config.MaxTokens,
			Temperature:   a.config.Temperature,
			StopSequences: nil,
			Metadata:      extractRequestMetadata(state.metadata),
		}

		// Agentic thinking: on the thinking turn, inject thinking prompt.
		// Tools remain available so the model can use them while thinking
		// (e.g., web search when the user asks to research something).
		if a.config.ThinkingPrompt != "" && !state.thinkingCompleted && !state.nativeThinkingSeen {
			state.isThinkingTurn = true
			if req.SystemPrompt != "" {
				req.SystemPrompt += "\n\n" + a.config.ThinkingPrompt
			} else {
				req.SystemPrompt = a.config.ThinkingPrompt
			}
		}

		modelStart := time.Now()
		a.logInfo("model call starting",
			slog.Int("turn", state.turnCount),
			slog.Int("messages", len(state.messages)),
			slog.Int("tools", len(toolDefs)),
		)

		stream, err := a.createStreamWithRetry(ctx, req, events)
		if err != nil {
			a.logError("model call failed", slog.Int("turn", state.turnCount), slog.String("error", err.Error()))
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

		// Consume the stream: emit text deltas, start tool execution immediately.
		stExec := newStreamingToolExecutor(a, state, events)
		assistantMsg, toolCalls, err := a.consumeStreamWithExecutor(ctx, stream, events, stExec)
		stream.Close()
		a.logInfo("model call completed",
			slog.Int("turn", state.turnCount),
			slog.Duration("duration", time.Since(modelStart)),
			slog.Int("tool_calls", len(toolCalls)),
		)
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
					pct := float64(bt.totalConsumed()) / float64(bt.budget.MaxTokens)
					a.logWarn("token budget warning",
						slog.Int("consumed", bt.totalConsumed()),
						slog.Int("max", bt.budget.MaxTokens),
						slog.Float64("percentage", pct),
					)
					a.emit(events, Event{
						Type: EventBudgetWarning,
						BudgetWarning: &BudgetWarningEvent{
							ConsumedTokens: bt.totalConsumed(),
							MaxTokens:      bt.budget.MaxTokens,
							Percentage:     pct,
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

		// Text-based tool calling: detect [TOOL_CALL: ...] in assistant text and thinking content.
		// Models may write tool calls inside <think> blocks (native thinking) or during
		// agentic thinking turns. We check thinking content first (since it may contain
		// tool calls not present in the text content), then fall back to text content.
		if len(a.tools) > 0 && len(toolCalls) == 0 {
			var textToolCalls []ToolCall
			// First check thinking content (native thinking or agentic thinking turn).
			if state.lastThinkContent != "" {
				textToolCalls = a.parseTextToolCalls(state.lastThinkContent)
			}
			// If no tool calls found in thinking, check text content.
			if len(textToolCalls) == 0 {
				textToolCalls = a.parseTextToolCalls(assistantMsg.TextContent())
			}
			if len(textToolCalls) > 0 {
				results := a.executeTools(ctx, textToolCalls, state, events)
				// Build tool result as a system message (model doesn't expect tool_result format).
				var resultParts []string
				for _, r := range results {
					prefix := "Tool result"
					if r.result.IsError {
						prefix = "Tool error"
					}
					resultParts = append(resultParts, fmt.Sprintf("[%s from %s]: %s", prefix, r.callID, r.result.Content))
				}
				resultMsg := Message{
					Role:    RoleSystem,
					Content: []ContentBlock{{Type: ContentText, Text: strings.Join(resultParts, "\n\n")}},
				}
				state.messages = append(state.messages, resultMsg)
				a.emit(events, Event{Type: EventMessage, Message: &resultMsg})
				continue // Loop back so model can use the results.
			}
		}

		// Check if tool calls are needed.
		if len(toolCalls) == 0 {
			// Agentic thinking: thinking turn completed, inject answer prompt.
			if state.isThinkingTurn && !state.nativeThinkingSeen {
				state.isThinkingTurn = false
				state.thinkingCompleted = true
				state.messages = append(state.messages, Message{
					Role:    RoleUser,
					Content: []ContentBlock{{Type: ContentText, Text: a.config.AnswerPrompt}},
				})
				a.emit(events, Event{
					Type:    EventTurnEnd,
					TurnEnd: &TurnEndEvent{TurnNumber: state.turnCount, Reason: TurnEndComplete, Messages: state.messages},
				})
				continue // Proceed to answer turn.
			}

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

		// Collect results from streaming executor (tools already started during stream).
		// If streaming executor has pending results, use those. Otherwise fall back
		// to sequential execution for any tools not yet started.
		var results []toolExecResult
		if stExec.hasPending() {
			results = stExec.collect(ctx)
		} else {
			results = a.executeTools(ctx, toolCalls, state, events)
		}

		// Build and append the tool results message.
		toolResultMsg := newToolResultMessage(results)
		state.messages = append(state.messages, toolResultMsg)
		a.emit(events, Event{Type: EventMessage, Message: &toolResultMsg})
	}
}

// consumeStreamWithExecutor reads events from the provider stream and immediately
// submits tool calls to the streaming executor as they arrive, rather than waiting
// for the stream to finish. This overlaps model generation with tool execution.
func (a *Agent) consumeStreamWithExecutor(ctx context.Context, stream Stream, events chan<- Event, executor *streamingToolExecutor) (Message, []ToolCall, error) {
	msg, calls, err := a.consumeStream(ctx, stream, events, executor.state)
	if err != nil {
		return msg, calls, err
	}

	// Submit all tool calls to the streaming executor for parallel execution.
	for _, call := range calls {
		tool, ok := a.tools[call.Name]
		if ok && IsToolAllowed(tool, a.config.ExecutionMode) {
			executor.submit(call)
		}
	}

	return msg, calls, nil
}

// consumeStream reads all events from the provider stream, emitting text deltas
// to the event channel and collecting tool calls. Returns the complete assistant
// message and extracted tool calls. The state parameter is used for agentic
// thinking: when state.isThinkingTurn is true, text deltas are emitted as
// EventThinkingDelta instead of EventTextDelta.
func (a *Agent) consumeStream(ctx context.Context, stream Stream, events chan<- Event, state *loopState) (Message, []ToolCall, error) {
	var textParts    []string
	var thinkParts   []string
	var toolCalls    []ToolCall
	var blocks       []ContentBlock

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
				if state != nil && state.isThinkingTurn {
					// Agentic thinking: emit text as thinking delta.
					a.emit(events, Event{
						Type:       EventThinkingDelta,
						ThinkDelta: &TextDeltaEvent{Text: ev.Delta.Text},
					})
					thinkParts = append(thinkParts, ev.Delta.Text)
				} else {
					a.emit(events, Event{
						Type:      EventTextDelta,
						TextDelta: &TextDeltaEvent{Text: ev.Delta.Text},
					})
				}
				textParts = append(textParts, ev.Delta.Text)
			}

		case StreamEventThinkingDelta:
			if ev.ThinkingDelta != nil {
				// Native thinking detected — disable agentic thinking.
				if state != nil {
					state.nativeThinkingSeen = true
					state.isThinkingTurn = false
				}
				a.emit(events, Event{
					Type:       EventThinkingDelta,
					ThinkDelta: &TextDeltaEvent{Text: ev.ThinkingDelta.Text},
				})
				thinkParts = append(thinkParts, ev.ThinkingDelta.Text)
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

	// Store thinking content in state for text tool call detection.
	if state != nil && len(thinkParts) > 0 {
		var sb strings.Builder
		for _, p := range thinkParts {
			sb.WriteString(p)
		}
		state.lastThinkContent = sb.String()
	} else if state != nil {
		state.lastThinkContent = ""
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
	// Take a snapshot of messages for hook context to avoid races with the
	// main loop appending to state.messages while streaming executor runs.
	msgSnapshot := make([]Message, len(state.messages))
	copy(msgSnapshot, state.messages)

	for _, hook := range a.hooksForPhase(HookPreToolUse) {
		action, err := hook.Execute(ctx, &HookContext{
			Phase:     HookPreToolUse,
			ToolCall:  &currentCall,
			Messages:  msgSnapshot,
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
				a.emit(events, Event{
					Type: EventHookBlocked,
					HookBlocked: &HookBlockedEvent{
						Phase:     HookPreToolUse,
						ToolCall:  &call,
						Reason:    reason,
						TurnCount: state.turnCount,
					},
				})
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
		a.emit(events, Event{
			Type:             EventPermissionDenied,
			PermissionDenied: &PermissionDeniedEvent{ToolCall: call},
		})
		return emitEarlyReturn(&ToolResult{Content: "Permission denied for tool \"" + call.Name + "\"", IsError: true})
	}

	// Input validation against tool's JSON Schema.
	if !a.config.DisableInputValidation {
		if schema := tool.InputSchema(); len(schema) > 0 {
			if err := jsonschema.Validate(schema, currentCall.Input); err != nil {
				a.logWarn("tool input validation failed",
					slog.String("tool", call.Name),
					slog.String("error", err.Error()),
				)
				return emitEarlyReturn(&ToolResult{
					Content: "input validation error for tool \"" + call.Name + "\": " + err.Error(),
					IsError: true,
				})
			}
		}
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

	maxAttempts := 1 + a.config.ToolRetries
	var result *ToolResult

	for attempt := 0; attempt < maxAttempts; attempt++ {
		toolCtx := ctx
		var cancelTimeout context.CancelFunc

		// Determine timeout: tool-specific (TimeoutAware) overrides global.
		timeout := a.config.ToolTimeout
		if ta, ok := tool.(TimeoutAware); ok {
			if toolTimeout := ta.Timeout(); toolTimeout > 0 {
				timeout = toolTimeout
			}
		}
		if timeout > 0 {
			toolCtx, cancelTimeout = context.WithTimeout(ctx, timeout)
		}

		var toolErr error
		result, toolErr = a.callToolWithRecovery(toolCtx, tool, currentCall.Input, progressFn)

		if cancelTimeout != nil {
			cancelTimeout()
		}

		if toolErr != nil {
			result = &ToolResult{Content: toolErr.Error(), IsError: true}
		}

		// Only retry if result is an error and we have attempts left.
		if result != nil && !result.IsError {
			break
		}
		if attempt < maxAttempts-1 {
			a.logWarn("retrying tool execution",
				slog.String("tool", call.Name),
				slog.Int("attempt", attempt+1),
				slog.Int("max_attempts", maxAttempts),
			)
		}
	}
	duration := time.Since(start)

	// Apply error strategy if the tool returned an error.
	if result != nil && result.IsError && a.config.ErrorStrategy != nil {
		newResult, action := a.config.ErrorStrategy.OnToolError(&call, result)
		if newResult != nil {
			result = newResult
		}
		if action == ErrorActionAbort {
			a.emit(events, Event{
				Type: EventToolEnd,
				ToolEnd: &ToolEndEvent{ToolCall: call, Result: *result, Duration: duration},
			})
			return toolExecResult{callID: call.ID, result: &ToolResult{Content: "aborted by error strategy", IsError: true}}
		}
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

	// Post-tool hooks (use snapshot to avoid race with main loop).
	postMsgSnapshot := make([]Message, len(state.messages))
	copy(postMsgSnapshot, state.messages)

	for _, hook := range a.hooksForPhase(HookPostToolUse) {
		hook.Execute(ctx, &HookContext{
			Phase:      HookPostToolUse,
			ToolCall:   &call,
			ToolResult: result,
			Messages:   postMsgSnapshot,
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

// createStreamWithRetry wraps provider.CreateStream with rate limiting and retry logic.
func (a *Agent) createStreamWithRetry(ctx context.Context, req *Request, events chan<- Event) (Stream, error) {
	policy := a.config.RetryPolicy
	if policy == nil || policy.MaxRetries <= 0 {
		if err := a.waitRateLimit(ctx); err != nil {
			return nil, err
		}
		return a.provider.CreateStream(ctx, req)
	}

	var lastErr error
	for attempt := 0; attempt <= policy.MaxRetries; attempt++ {
		if attempt > 0 {
			delay := policy.BaseDelay * time.Duration(1<<(attempt-1))
			if delay > policy.MaxDelay {
				delay = policy.MaxDelay
			}
			a.logWarn("retrying provider call",
				slog.Int("attempt", attempt),
				slog.Duration("delay", delay),
				slog.String("error", lastErr.Error()),
			)
			a.emit(events, Event{
				Type: EventRetry,
				Retry: &RetryEvent{
					Attempt: attempt,
					Delay:   delay,
					Err:     lastErr,
				},
			})
			select {
			case <-time.After(delay):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		if err := a.waitRateLimit(ctx); err != nil {
			return nil, err
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

// waitRateLimit blocks until the rate limiter allows the request, if configured.
func (a *Agent) waitRateLimit(ctx context.Context) error {
	if a.config.RateLimiter == nil {
		return nil
	}
	return a.config.RateLimiter.Wait(ctx)
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
				reason := action.BlockReason
				if reason == "" {
					reason = "blocked by hook"
				}
				a.emit(events, Event{
					Type: EventHookBlocked,
					HookBlocked: &HookBlockedEvent{
						Phase:     phase,
						Reason:    reason,
						TurnCount: hc.TurnCount,
					},
				})
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
// Supports both single-phase Hook and MultiPhaseHook interfaces.
func (a *Agent) hooksForPhase(phase HookPhase) []Hook {
	var matched []Hook
	for _, h := range a.hooks {
		if h.Phase() == phase {
			matched = append(matched, h)
			continue
		}
		// Check if this hook implements MultiPhaseHook.
		if mph, ok := h.(MultiPhaseHook); ok {
			for _, p := range mph.Phases() {
				if p == phase {
					matched = append(matched, h)
					break
				}
			}
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

// requestMetadataPrefix is the prefix for hook metadata keys that should be
// propagated to provider HTTP requests as headers.
const requestMetadataPrefix = "request:"

// extractRequestMetadata copies entries from hook metadata that have the
// "request:" prefix into a map suitable for Request.Metadata. Hooks (like
// the Tracer) write "request:traceparent" = "00-..." which becomes the
// "traceparent" HTTP header on the provider request.
func extractRequestMetadata(hookMetadata map[string]any) map[string]string {
	var result map[string]string
	for k, v := range hookMetadata {
		if len(k) > len(requestMetadataPrefix) && k[:len(requestMetadataPrefix)] == requestMetadataPrefix {
			if str, ok := v.(string); ok {
				if result == nil {
					result = make(map[string]string)
				}
				result[k[len(requestMetadataPrefix):]] = str
			}
		}
	}
	return result
}

// logInfo logs at INFO level if the agent has a logger configured.
func (a *Agent) logInfo(msg string, args ...any) {
	if a.config.Logger != nil {
		a.config.Logger.Info(msg, args...)
	}
}

// logWarn logs at WARN level if the agent has a logger configured.
func (a *Agent) logWarn(msg string, args ...any) {
	if a.config.Logger != nil {
		a.config.Logger.Warn(msg, args...)
	}
}

// logError logs at ERROR level if the agent has a logger configured.
func (a *Agent) logError(msg string, args ...any) {
	if a.config.Logger != nil {
		a.config.Logger.Error(msg, args...)
	}
}

// --- Text-based tool calling ---

// textToolCallRegex matches [TOOL_CALL: tool_name("arguments")] or [TOOL_CALL: tool_name(arguments)].
var textToolCallRegex = regexp.MustCompile(`\[TOOL_CALL:\s*(\w+)\(([^)]*)\)\]`)

// buildToolInstruction generates a system prompt fragment describing available
// tools and the expected [TOOL_CALL: ...] format for text-based tool calling.
func (a *Agent) buildToolInstruction() string {
	tools := a.toolDefinitions()
	if len(tools) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("\n\nYou have access to the following tools:\n")
	for _, t := range tools {
		sb.WriteString(fmt.Sprintf("- %s: %s\n", t.Name, t.Description))
	}
	sb.WriteString("\nWhen you need to use a tool, write EXACTLY this format in your response:\n")
	sb.WriteString("[TOOL_CALL: tool_name(\"arguments\")]\n\n")
	sb.WriteString("Examples:\n")
	sb.WriteString("[TOOL_CALL: web_search(\"current weather in Istanbul\")]\n")
	sb.WriteString("[TOOL_CALL: deep_search(\"April 5 2026 special events\")]\n\n")
	sb.WriteString("Wait for the tool result before providing your final answer.\n")
	sb.WriteString("Do NOT make up information - use the appropriate tool for current or factual data.\n")
	return sb.String()
}

// parseTextToolCalls extracts [TOOL_CALL: tool_name("args")] patterns from text.
// Returns ToolCall objects ready for execution through the standard tool pipeline.
func (a *Agent) parseTextToolCalls(text string) []ToolCall {
	matches := textToolCallRegex.FindAllStringSubmatch(text, -1)
	if len(matches) == 0 {
		return nil
	}

	var calls []ToolCall
	for i, match := range matches {
		if len(match) < 3 {
			continue
		}
		toolName := match[1]
		rawArgs := match[2]

		// Check if tool is registered.
		if _, ok := a.tools[toolName]; !ok {
			continue
		}

		// Clean up arguments: strip surrounding quotes if present.
		rawArgs = strings.TrimSpace(rawArgs)
		if len(rawArgs) >= 2 && rawArgs[0] == '"' && rawArgs[len(rawArgs)-1] == '"' {
			rawArgs = rawArgs[1 : len(rawArgs)-1]
		}

		// Build JSON input. For simple string arguments, wrap as {"query": "..."}.
		// For JSON-like input, pass through.
		var inputJSON json.RawMessage
		if len(rawArgs) > 0 && rawArgs[0] == '{' {
			inputJSON = json.RawMessage(rawArgs)
		} else {
			inputJSON, _ = json.Marshal(map[string]string{"query": rawArgs})
		}

		calls = append(calls, ToolCall{
			ID:    fmt.Sprintf("text-call-%d", i),
			Name:  toolName,
			Input: inputJSON,
		})
	}

	return calls
}

// loopState carries mutable state across agentic loop iterations.
type loopState struct {
	messages  []Message
	turnCount int
	metadata  map[string]any
	done      chan struct{} // closed when the loop terminates, guards emit
	bgWork    sync.WaitGroup // tracks background goroutines (streaming executor)

	// Agentic thinking state (for non-native thinking models).
	isThinkingTurn     bool // true when current turn should emit ThinkingDelta
	thinkingCompleted  bool // true after the thinking turn has finished
	nativeThinkingSeen bool // true if provider emitted StreamEventThinkingDelta

	// Last turn's thinking content for text tool call detection.
	// When [TOOL_CALL: ...] appears inside thinking, we need to parse it.
	lastThinkContent string
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
