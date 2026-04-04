package agentflow

import (
	"context"
	"sync"
	"time"
)

// streamingToolExecutor starts executing tools as they arrive from the model
// stream, rather than waiting for the entire stream to finish. This reduces
// perceived latency by overlapping model generation with tool execution.
type streamingToolExecutor struct {
	agent *Agent
	state *loopState

	mu      sync.Mutex
	pending []trackedToolExec
	events  chan<- Event
}

type trackedToolExec struct {
	call   ToolCall
	done   chan struct{}
	result toolExecResult
}

// newStreamingToolExecutor creates an executor that will process tool calls
// as they arrive during streaming.
func newStreamingToolExecutor(agent *Agent, state *loopState, events chan<- Event) *streamingToolExecutor {
	return &streamingToolExecutor{
		agent:  agent,
		state:  state,
		events: events,
	}
}

// submit queues a tool call for immediate execution in a goroutine.
// Safe to call from the stream consumer goroutine.
func (e *streamingToolExecutor) submit(call ToolCall) {
	tracked := trackedToolExec{
		call: call,
		done: make(chan struct{}),
	}

	e.mu.Lock()
	e.pending = append(e.pending, tracked)
	idx := len(e.pending) - 1
	e.mu.Unlock()

	e.state.bgWork.Add(1)
	go func() {
		defer e.state.bgWork.Done()
		defer close(tracked.done)
		// Use a context derived from the loop's done channel so that
		// tool execution is cancelled when the loop terminates.
		ctx, cancel := context.WithCancel(context.Background())
		go func() {
			select {
			case <-e.state.done:
				cancel()
			case <-ctx.Done():
			}
		}()
		result := e.agent.executeSingleTool(ctx, call, e.state, e.events)
		cancel()
		e.mu.Lock()
		e.pending[idx].result = result
		e.mu.Unlock()
	}()
}

// collect waits for all submitted tool calls to complete and returns results
// in submission order. Call this after the stream finishes.
func (e *streamingToolExecutor) collect(ctx context.Context) []toolExecResult {
	e.mu.Lock()
	pending := make([]trackedToolExec, len(e.pending))
	copy(pending, e.pending)
	e.mu.Unlock()

	results := make([]toolExecResult, len(pending))
	for i, tracked := range pending {
		select {
		case <-tracked.done:
			e.mu.Lock()
			results[i] = e.pending[i].result
			e.mu.Unlock()
		case <-ctx.Done():
			results[i] = toolExecResult{
				callID: tracked.call.ID,
				result: &ToolResult{Content: "execution cancelled", IsError: true},
			}
		}
	}
	return results
}

// hasPending returns true if there are submitted but not yet collected tool calls.
func (e *streamingToolExecutor) hasPending() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.pending) > 0
}

// collectTimings returns execution timing info for observability.
func (e *streamingToolExecutor) collectTimings() map[string]time.Duration {
	e.mu.Lock()
	defer e.mu.Unlock()

	timings := make(map[string]time.Duration)
	for _, tracked := range e.pending {
		if tracked.result.result != nil {
			if md := tracked.result.result.Metadata; md != nil {
				if d, ok := md["duration"].(time.Duration); ok {
					timings[tracked.call.Name] = d
				}
			}
		}
	}
	return timings
}
