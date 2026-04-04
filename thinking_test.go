package agentflow_test

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	groqprovider "github.com/CanArslanDev/agentflow/provider/groq"
	"github.com/CanArslanDev/agentflow/provider/mock"
	"github.com/CanArslanDev/agentflow/tools"
)

func TestAgenticThinking_Basic(t *testing.T) {
	provider := mock.New(
		// Turn 1: thinking response
		mock.WithResponse(
			mock.TextDelta("Let me think... 2+2 = 4."),
		),
		// Turn 2: answer response
		mock.WithResponse(
			mock.TextDelta("4"),
		),
	)

	agent := agentflow.NewAgent(provider,
		agentflow.WithThinkingPrompt(
			"Think step by step about this question.",
			"Now give your final answer.",
		),
		agentflow.WithMaxTurns(5),
	)

	var thinkingContent, messageContent string
	var turnEnds []agentflow.TurnEndReason

	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("What is 2+2?"),
	}) {
		switch ev.Type {
		case agentflow.EventThinkingDelta:
			thinkingContent += ev.ThinkDelta.Text
		case agentflow.EventTextDelta:
			messageContent += ev.TextDelta.Text
		case agentflow.EventTurnEnd:
			turnEnds = append(turnEnds, ev.TurnEnd.Reason)
		}
	}

	if thinkingContent != "Let me think... 2+2 = 4." {
		t.Errorf("thinking content: %q", thinkingContent)
	}
	if messageContent != "4" {
		t.Errorf("message content: %q", messageContent)
	}
	if len(turnEnds) != 2 {
		t.Errorf("expected 2 turn ends, got %d", len(turnEnds))
	}

	// Verify provider was called twice (thinking + answer).
	if provider.CallCount() != 2 {
		t.Errorf("expected 2 provider calls, got %d", provider.CallCount())
	}
}

func TestAgenticThinking_NativeBypass(t *testing.T) {
	// Provider that sends native thinking deltas.
	provider := mock.New(
		mock.WithResponse(
			mock.ThinkingDelta("Native thinking..."),
			mock.TextDelta("The answer is 4"),
		),
	)

	agent := agentflow.NewAgent(provider,
		agentflow.WithThinkingPrompt(
			"Think step by step.",
			"Now answer.",
		),
		agentflow.WithMaxTurns(5),
	)

	var thinkingContent, messageContent string
	var turnCount int

	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("What is 2+2?"),
	}) {
		switch ev.Type {
		case agentflow.EventThinkingDelta:
			thinkingContent += ev.ThinkDelta.Text
		case agentflow.EventTextDelta:
			messageContent += ev.TextDelta.Text
		case agentflow.EventTurnEnd:
			turnCount++
		}
	}

	if thinkingContent != "Native thinking..." {
		t.Errorf("thinking: %q", thinkingContent)
	}
	if messageContent != "The answer is 4" {
		t.Errorf("message: %q", messageContent)
	}
	// Should complete in 1 turn (no answer prompt injection).
	if turnCount != 1 {
		t.Errorf("expected 1 turn, got %d", turnCount)
	}
	// Provider called only once.
	if provider.CallCount() != 1 {
		t.Errorf("expected 1 provider call, got %d", provider.CallCount())
	}
}

func TestAgenticThinking_Disabled(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(mock.TextDelta("Hello")),
	)

	// No WithThinkingPrompt -- normal behavior.
	agent := agentflow.NewAgent(provider, agentflow.WithMaxTurns(1))

	var thinkingContent, messageContent string
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("Hi"),
	}) {
		switch ev.Type {
		case agentflow.EventThinkingDelta:
			thinkingContent += ev.ThinkDelta.Text
		case agentflow.EventTextDelta:
			messageContent += ev.TextDelta.Text
		}
	}

	if thinkingContent != "" {
		t.Errorf("unexpected thinking content: %q", thinkingContent)
	}
	if messageContent != "Hello" {
		t.Errorf("message: %q", messageContent)
	}
}

func TestAgenticThinking_WithToolUse(t *testing.T) {
	provider := mock.New(
		// Turn 1: thinking (no tool calls since tools are hidden)
		mock.WithResponse(
			mock.TextDelta("I need to check the weather."),
		),
		// Turn 2: answer with tool call
		mock.WithResponse(
			mock.ToolCallEvent("tc_1", "get_weather", `{"city":"Istanbul"}`),
		),
		// Turn 3: final answer after tool result
		mock.WithResponse(
			mock.TextDelta("The weather in Istanbul is sunny."),
		),
	)

	weatherTool := tools.New("get_weather", "Get weather for a city").
		WithSchema(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string"},
			},
			"required": []string{"city"},
		}).
		ConcurrencySafe(true).
		ReadOnly(true).
		WithExecute(func(ctx context.Context, input json.RawMessage, p agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{Content: "Sunny, 25C"}, nil
		}).Build()

	agent := agentflow.NewAgent(provider,
		agentflow.WithThinkingPrompt("Think first.", "Now answer using tools if needed."),
		agentflow.WithTool(weatherTool),
		agentflow.WithMaxTurns(10),
	)

	var thinkingContent, messageContent string
	var toolCalls []string
	var turnCount int

	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("What is the weather in Istanbul?"),
	}) {
		switch ev.Type {
		case agentflow.EventThinkingDelta:
			thinkingContent += ev.ThinkDelta.Text
		case agentflow.EventTextDelta:
			messageContent += ev.TextDelta.Text
		case agentflow.EventToolStart:
			toolCalls = append(toolCalls, ev.ToolStart.ToolCall.Name)
		case agentflow.EventTurnEnd:
			turnCount++
		}
	}

	// Turn 1: thinking output
	if thinkingContent != "I need to check the weather." {
		t.Errorf("thinking: %q", thinkingContent)
	}
	// Turn 2: tool call (answer turn)
	if len(toolCalls) != 1 || toolCalls[0] != "get_weather" {
		t.Errorf("tool calls: %v", toolCalls)
	}
	// Turn 3: final answer
	if messageContent != "The weather in Istanbul is sunny." {
		t.Errorf("message: %q", messageContent)
	}
	// 2 TurnEnd events: thinking turn + final answer turn.
	// The tool call turn does not emit TurnEnd (loop continues to process tool results).
	if turnCount != 2 {
		t.Errorf("expected 2 turn ends, got %d", turnCount)
	}
	// 3 provider calls: thinking + answer(tool call) + final answer.
	if provider.CallCount() != 3 {
		t.Errorf("expected 3 provider calls, got %d", provider.CallCount())
	}
}

func TestAgenticThinking_TurnEndMessages(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(mock.TextDelta("thinking...")),
		mock.WithResponse(mock.TextDelta("answer")),
	)

	agent := agentflow.NewAgent(provider,
		agentflow.WithThinkingPrompt("Think.", "Answer."),
		agentflow.WithMaxTurns(5),
	)

	var lastMessages []agentflow.Message
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("test"),
	}) {
		if ev.Type == agentflow.EventTurnEnd && ev.TurnEnd != nil {
			lastMessages = ev.TurnEnd.Messages
		}
	}

	// Final messages should include: user("test"), assistant("thinking..."),
	// user(answerPrompt), assistant("answer")
	if len(lastMessages) < 4 {
		t.Fatalf("expected at least 4 messages, got %d", len(lastMessages))
	}

	// Check the injected answer prompt is in the messages.
	answerMsg := lastMessages[2]
	if answerMsg.Role != agentflow.RoleUser || answerMsg.TextContent() != "Answer." {
		t.Errorf("expected answer prompt message, got role=%s text=%q", answerMsg.Role, answerMsg.TextContent())
	}
}

// --- TimeoutAware tests ---

func TestTimeoutAware_ToolSpecificTimeout(t *testing.T) {
	// Tool with a very short custom timeout.
	slowTool := tools.New("slow_tool", "A slow tool").
		WithSchema(map[string]any{"type": "object", "properties": map[string]any{}}).
		ConcurrencySafe(true).
		ReadOnly(true).
		WithTimeout(50 * time.Millisecond). // 50ms tool-specific timeout
		WithExecute(func(ctx context.Context, input json.RawMessage, p agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(5 * time.Second):
				return &agentflow.ToolResult{Content: "done"}, nil
			}
		}).Build()

	provider := mock.New(
		mock.WithResponse(
			mock.ToolCallEvent("tc_1", "slow_tool", `{}`),
		),
		mock.WithResponse(
			mock.TextDelta("ok"),
		),
	)

	agent := agentflow.NewAgent(provider,
		agentflow.WithTool(slowTool),
		agentflow.WithToolTimeout(30*time.Second), // Global: 30s (should be overridden)
		agentflow.WithMaxTurns(3),
	)

	start := time.Now()
	var toolError bool
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("run slow tool"),
	}) {
		if ev.Type == agentflow.EventToolEnd && ev.ToolEnd != nil {
			toolError = ev.ToolEnd.Result.IsError
		}
	}
	elapsed := time.Since(start)

	// Should timeout at ~50ms, not 30s.
	if elapsed > 2*time.Second {
		t.Errorf("took too long (%v), tool-specific timeout not applied", elapsed)
	}
	if !toolError {
		t.Error("expected tool to timeout with error")
	}
}

func TestTimeoutAware_FallbackToGlobal(t *testing.T) {
	// Tool WITHOUT TimeoutAware -- should use global timeout.
	normalTool := tools.New("normal_tool", "A normal tool").
		WithSchema(map[string]any{"type": "object", "properties": map[string]any{}}).
		ConcurrencySafe(true).
		ReadOnly(true).
		// No WithTimeout -- uses global
		WithExecute(func(ctx context.Context, input json.RawMessage, p agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(5 * time.Second):
				return &agentflow.ToolResult{Content: "done"}, nil
			}
		}).Build()

	provider := mock.New(
		mock.WithResponse(
			mock.ToolCallEvent("tc_1", "normal_tool", `{}`),
		),
		mock.WithResponse(
			mock.TextDelta("ok"),
		),
	)

	agent := agentflow.NewAgent(provider,
		agentflow.WithTool(normalTool),
		agentflow.WithToolTimeout(50*time.Millisecond), // Global: 50ms
		agentflow.WithMaxTurns(3),
	)

	start := time.Now()
	var toolError bool
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("run normal tool"),
	}) {
		if ev.Type == agentflow.EventToolEnd && ev.ToolEnd != nil {
			toolError = ev.ToolEnd.Result.IsError
		}
	}
	elapsed := time.Since(start)

	if elapsed > 2*time.Second {
		t.Errorf("took too long (%v), global timeout not applied", elapsed)
	}
	if !toolError {
		t.Error("expected tool to timeout with error")
	}
}

func TestBuilderWithTimeout(t *testing.T) {
	tool := tools.New("test", "test tool").
		WithTimeout(120 * time.Second).
		WithExecute(func(ctx context.Context, input json.RawMessage, p agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{Content: "ok"}, nil
		}).Build()

	ta, ok := tool.(agentflow.TimeoutAware)
	if !ok {
		t.Fatal("built tool should implement TimeoutAware")
	}
	if ta.Timeout() != 120*time.Second {
		t.Errorf("timeout: %v", ta.Timeout())
	}
}

// --- Integration tests ---

func TestIntegration_AgenticThinking(t *testing.T) {
	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		t.Skip("GROQ_API_KEY not set")
	}

	provider := groqprovider.New(key, "llama-3.3-70b-versatile")

	agent := agentflow.NewAgent(provider,
		agentflow.WithThinkingPrompt(
			"Think step by step about this question. Share your reasoning process. Do not give a final answer yet.",
			"Based on your thinking above, now provide only the final answer in one short sentence.",
		),
		agentflow.WithMaxTurns(5),
		agentflow.WithMaxTokens(200),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var thinkingContent, messageContent string
	var turnCount int

	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("What is 15 * 28?"),
	}) {
		switch ev.Type {
		case agentflow.EventThinkingDelta:
			thinkingContent += ev.ThinkDelta.Text
		case agentflow.EventTextDelta:
			messageContent += ev.TextDelta.Text
		case agentflow.EventTurnEnd:
			turnCount++
		}
	}

	t.Logf("Thinking: %q", thinkingContent)
	t.Logf("Answer: %q", messageContent)
	t.Logf("Turns: %d", turnCount)

	if thinkingContent == "" {
		t.Error("expected thinking content from turn 1")
	}
	if messageContent == "" {
		t.Error("expected answer content from turn 2")
	}
	if turnCount != 2 {
		t.Errorf("expected 2 turns (thinking + answer), got %d", turnCount)
	}
}

func TestBuilderWithTimeout_Zero(t *testing.T) {
	// No WithTimeout call -- should return zero (use global).
	tool := tools.New("test", "test tool").
		WithExecute(func(ctx context.Context, input json.RawMessage, p agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{Content: "ok"}, nil
		}).Build()

	ta, ok := tool.(agentflow.TimeoutAware)
	if !ok {
		t.Fatal("built tool should implement TimeoutAware")
	}
	if ta.Timeout() != 0 {
		t.Errorf("expected zero timeout, got %v", ta.Timeout())
	}
}
