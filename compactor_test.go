package agentflow_test

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/compactor"
	"github.com/CanArslanDev/agentflow/provider/groq"
	"github.com/CanArslanDev/agentflow/provider/mock"
	"github.com/CanArslanDev/agentflow/tools"
)

// --- SlidingWindowCompactor tests ---

func TestSlidingWindow_NoCompactionNeeded(t *testing.T) {
	c := compactor.NewSlidingWindow(10, 0) // trigger at 20

	messages := make([]agentflow.Message, 5)
	for i := range messages {
		messages[i] = agentflow.NewUserMessage("msg")
	}

	if c.ShouldCompact(messages, nil) {
		t.Error("should not compact with only 5 messages")
	}
}

func TestSlidingWindow_CompactsWhenTriggered(t *testing.T) {
	c := compactor.NewSlidingWindow(4, 8) // keep last 4, trigger at 8

	messages := make([]agentflow.Message, 12)
	messages[0] = agentflow.NewUserMessage("Initial task")
	for i := 1; i < 12; i++ {
		if i%2 == 1 {
			messages[i] = agentflow.NewUserMessage("user msg " + string(rune('A'+i)))
		} else {
			messages[i] = agentflow.NewAssistantMessage("assistant msg " + string(rune('A'+i)))
		}
	}

	if !c.ShouldCompact(messages, nil) {
		t.Fatal("should compact with 12 messages (trigger=8)")
	}

	compacted, err := c.Compact(context.Background(), messages)
	if err != nil {
		t.Fatalf("compact: %v", err)
	}

	// Should have: first + notice + last 4 = 6 messages.
	if len(compacted) != 6 {
		t.Errorf("expected 6 messages (1+1+4), got %d", len(compacted))
	}

	// First message preserved.
	if compacted[0].TextContent() != "Initial task" {
		t.Errorf("first message not preserved: %q", compacted[0].TextContent())
	}

	// Second should be system compaction notice.
	if compacted[1].Role != agentflow.RoleSystem {
		t.Errorf("expected system notice, got role %s", compacted[1].Role)
	}
	notice := compacted[1].TextContent()
	if notice == "" {
		t.Error("empty compaction notice")
	}
	t.Logf("Compaction notice: %s", notice)

	// Last message preserved.
	last := compacted[len(compacted)-1]
	lastOriginal := messages[len(messages)-1]
	if last.TextContent() != lastOriginal.TextContent() {
		t.Errorf("last message mismatch: %q vs %q", last.TextContent(), lastOriginal.TextContent())
	}
}

func TestSlidingWindow_PreservesFirstAndRecent(t *testing.T) {
	c := compactor.NewSlidingWindow(2, 5)

	messages := []agentflow.Message{
		agentflow.NewUserMessage("task"),           // 0 - preserved
		agentflow.NewAssistantMessage("response1"), // 1 - discarded
		agentflow.NewUserMessage("follow up 1"),    // 2 - discarded
		agentflow.NewAssistantMessage("response2"), // 3 - discarded
		agentflow.NewUserMessage("follow up 2"),    // 4 - discarded
		agentflow.NewAssistantMessage("response3"), // 5 - kept
		agentflow.NewUserMessage("follow up 3"),    // 6 - kept
	}

	compacted, err := c.Compact(context.Background(), messages)
	if err != nil {
		t.Fatalf("compact: %v", err)
	}

	// first + notice + 2 recent = 4
	if len(compacted) != 4 {
		t.Fatalf("expected 4, got %d", len(compacted))
	}

	if compacted[0].TextContent() != "task" {
		t.Error("first message lost")
	}
	if compacted[2].TextContent() != "response3" {
		t.Errorf("expected response3, got %q", compacted[2].TextContent())
	}
	if compacted[3].TextContent() != "follow up 3" {
		t.Errorf("expected follow up 3, got %q", compacted[3].TextContent())
	}
}

// --- TokenWindowCompactor tests ---

func TestTokenWindow_TriggersOnEstimate(t *testing.T) {
	c := compactor.NewTokenWindow(100, 4) // 100 tokens = 400 chars

	// Each message ~200 chars -> ~50 tokens. 3 messages = ~150 tokens > 100.
	messages := []agentflow.Message{
		agentflow.NewUserMessage(string(make([]byte, 200))),
		agentflow.NewAssistantMessage(string(make([]byte, 200))),
		agentflow.NewUserMessage(string(make([]byte, 200))),
	}

	if !c.ShouldCompact(messages, nil) {
		t.Error("should compact based on estimated tokens")
	}
}

func TestTokenWindow_UsesRealUsage(t *testing.T) {
	c := compactor.NewTokenWindow(1000, 4)

	messages := []agentflow.Message{agentflow.NewUserMessage("short")}
	usage := &agentflow.Usage{PromptTokens: 1500}

	if !c.ShouldCompact(messages, usage) {
		t.Error("should compact when real usage exceeds threshold")
	}
}

// --- Agent integration with compactor ---

func TestAgentWithSlidingWindow(t *testing.T) {
	// Provider returns tool calls to force many turns, then final response.
	provider := mock.New(
		mock.WithResponse(mock.ToolCallEvent("tc1", "echo", `{}`)),
		mock.WithResponse(mock.ToolCallEvent("tc2", "echo", `{}`)),
		mock.WithResponse(mock.ToolCallEvent("tc3", "echo", `{}`)),
		mock.WithResponse(mock.ToolCallEvent("tc4", "echo", `{}`)),
		mock.WithResponse(mock.ToolCallEvent("tc5", "echo", `{}`)),
		mock.WithResponse(mock.TextDelta("Done after compaction.")),
	)

	echo := tools.New("echo", "Echo").
		WithSchema(map[string]any{"type": "object"}).
		ConcurrencySafe(true).ReadOnly(true).
		WithExecute(func(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{Content: "ok"}, nil
		}).Build()

	// Keep last 4 messages, trigger at 6. After 3 tool rounds (user+assistant+toolresult each),
	// we'll have ~7+ messages which triggers compaction.
	c := compactor.NewSlidingWindow(4, 6)

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(echo),
		agentflow.WithCompactor(c),
		agentflow.WithMaxTurns(10),
	)

	var turnEnd *agentflow.TurnEndEvent
	for ev := range agent.Run(context.Background(), []agentflow.Message{
		agentflow.NewUserMessage("go"),
	}) {
		if ev.Type == agentflow.EventTurnEnd {
			turnEnd = ev.TurnEnd
		}
	}

	if turnEnd == nil {
		t.Fatal("expected TurnEnd")
	}
	t.Logf("Completed at turn %d, reason: %s, final messages: %d",
		turnEnd.TurnNumber, turnEnd.Reason, len(turnEnd.Messages))

	// After compaction, final message count should be less than total turns * 2.
	// The compactor should have kicked in at least once.
	if len(turnEnd.Messages) > 15 {
		t.Errorf("expected compaction to reduce message count, got %d", len(turnEnd.Messages))
	}
}

// --- Integration test: SummaryCompactor with real Groq API ---

func TestIntegration_SummaryCompactor(t *testing.T) {
	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		t.Skip("GROQ_API_KEY not set")
	}

	provider := groq.New(key, "llama-3.3-70b-versatile")

	// Build a long conversation manually, then compact it.
	messages := []agentflow.Message{
		agentflow.NewUserMessage("I'm planning a trip to Japan next month."),
		agentflow.NewAssistantMessage("That sounds exciting! Japan is wonderful. When exactly are you going and what cities are you planning to visit?"),
		agentflow.NewUserMessage("I'll be there from May 10 to May 20. I want to visit Tokyo, Kyoto, and Osaka."),
		agentflow.NewAssistantMessage("Great choices! 10 days is perfect. I'd suggest 4 days in Tokyo, 3 in Kyoto, and 3 in Osaka. Would you like hotel recommendations?"),
		agentflow.NewUserMessage("Yes, budget hotels please. Max $80 per night."),
		agentflow.NewAssistantMessage("For budget stays: Tokyo - Khaosan World Asakusa ($45/night), Kyoto - Piece Hostel Sanjo ($55/night), Osaka - Cross Hotel Osaka ($75/night). All well-reviewed."),
		agentflow.NewUserMessage("Perfect. I also need to know about the JR Pass."),
		agentflow.NewAssistantMessage("The 7-day JR Pass costs about $200 and covers bullet trains between all three cities. Activate it on May 13 when you leave Tokyo for Kyoto."),
		agentflow.NewUserMessage("What about food? I'm vegetarian."),
		agentflow.NewAssistantMessage("Vegetarian options: Shojin ryori (Buddhist cuisine) in Kyoto temples, Ain Soph chain in Tokyo, and many curry houses in Osaka. Download the HappyCow app."),
	}

	c := compactor.NewSummary(provider, 4, 6)

	if !c.ShouldCompact(messages, nil) {
		t.Fatal("should trigger compaction for 10 messages")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	compacted, err := c.Compact(ctx, messages)
	if err != nil {
		t.Fatalf("compact: %v", err)
	}

	t.Logf("Original: %d messages -> Compacted: %d messages", len(messages), len(compacted))

	if len(compacted) >= len(messages) {
		t.Errorf("compaction didn't reduce: %d -> %d", len(messages), len(compacted))
	}

	// First message preserved.
	if compacted[0].TextContent() != messages[0].TextContent() {
		t.Error("first message not preserved")
	}

	// Summary should mention key facts.
	summary := compacted[1].TextContent()
	t.Logf("Summary: %s", summary)
	if summary == "" {
		t.Error("empty summary")
	}

	// Last 4 messages preserved.
	lastCompacted := compacted[len(compacted)-1]
	lastOriginal := messages[len(messages)-1]
	if lastCompacted.TextContent() != lastOriginal.TextContent() {
		t.Error("last message not preserved")
	}
}
