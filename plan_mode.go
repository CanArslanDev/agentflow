package agentflow

import (
	"context"
	"io"
	"strings"
)

// PlanResult holds the output of a planning phase.
type PlanResult struct {
	Plan     string    // The structured plan text.
	Messages []Message // Full conversation from planning.
}

// Plan runs the agent in planning-only mode: the model creates a structured plan
// but does not execute any tools. Returns the plan text that can be reviewed
// before execution.
//
//	plan, err := agentflow.Plan(ctx, provider, "Build a REST API in Go")
//	fmt.Println(plan.Plan)
//	// Review, then execute:
//	for ev := range agent.Run(ctx, append(plan.Messages, agentflow.NewUserMessage("Execute the plan"))) { ... }
func Plan(ctx context.Context, provider Provider, task string) (*PlanResult, error) {
	agent := NewAgent(provider,
		WithSystemPrompt(planSystemPrompt),
		WithMaxTurns(1),
		WithMaxTokens(2048),
	)

	var planText strings.Builder
	var messages []Message

	for ev := range agent.Run(ctx, []Message{NewUserMessage(task)}) {
		switch ev.Type {
		case EventTextDelta:
			if ev.TextDelta != nil {
				planText.WriteString(ev.TextDelta.Text)
			}
		case EventTurnEnd:
			if ev.TurnEnd != nil {
				messages = ev.TurnEnd.Messages
			}
		}
	}

	if planText.Len() == 0 {
		return nil, ErrProviderUnavailable
	}

	return &PlanResult{
		Plan:     planText.String(),
		Messages: messages,
	}, nil
}

// PlanAndExecute first creates a plan, then executes it in a second agent run
// with full tool access. Events from both phases are streamed through the channel.
func PlanAndExecute(ctx context.Context, provider Provider, task string, tools []Tool, opts ...Option) <-chan Event {
	events := make(chan Event, DefaultEventBufferSize)

	go func() {
		defer close(events)

		// Phase 1: Plan.
		events <- Event{
			Type:      EventTurnStart,
			TurnStart: &TurnStartEvent{TurnNumber: 0}, // Turn 0 = planning phase.
		}

		planAgent := NewAgent(provider,
			WithSystemPrompt(planSystemPrompt),
			WithMaxTurns(1),
			WithMaxTokens(2048),
		)

		var planText strings.Builder
		var planMessages []Message

		for ev := range planAgent.Run(ctx, []Message{NewUserMessage(task)}) {
			switch ev.Type {
			case EventTextDelta:
				events <- ev
				if ev.TextDelta != nil {
					planText.WriteString(ev.TextDelta.Text)
				}
			case EventTurnEnd:
				if ev.TurnEnd != nil {
					planMessages = ev.TurnEnd.Messages
				}
			}
		}

		if planText.Len() == 0 || ctx.Err() != nil {
			events <- Event{
				Type:    EventTurnEnd,
				TurnEnd: &TurnEndEvent{TurnNumber: 0, Reason: TurnEndError},
			}
			return
		}

		events <- Event{
			Type:    EventTurnEnd,
			TurnEnd: &TurnEndEvent{TurnNumber: 0, Reason: TurnEndComplete, Messages: planMessages},
		}

		// Phase 2: Execute with tools.
		execOpts := append([]Option{
			WithTools(tools...),
			WithSystemPrompt("Execute the following plan step by step. Use the available tools.\n\nPlan:\n" + planText.String()),
		}, opts...)

		execAgent := NewAgent(provider, execOpts...)

		for ev := range execAgent.Run(ctx, []Message{NewUserMessage("Execute the plan now.")}) {
			events <- ev
		}
	}()

	return events
}

const planSystemPrompt = `You are a planning specialist. When given a task:

1. Analyze the task requirements
2. Break it down into clear, numbered steps
3. Identify what tools or resources would be needed for each step
4. Estimate complexity for each step (simple/medium/complex)
5. Note any dependencies between steps

Output a structured plan. Do NOT execute anything — only plan.

Format:
## Plan: [Task Title]

### Steps:
1. [Step description] — [complexity] — [tools needed]
2. [Step description] — [complexity] — [tools needed]
...

### Dependencies:
- Step X depends on Step Y
...

### Notes:
- [Any important observations]`

// --- Session Memory Extraction ---

// ExtractMemories analyzes a conversation and extracts key facts worth remembering.
// Returns a list of memory entries that can be stored for future context.
func ExtractMemories(ctx context.Context, provider Provider, messages []Message) ([]string, error) {
	// Build conversation text for analysis.
	var conversation strings.Builder
	for _, msg := range messages {
		text := msg.TextContent()
		if text == "" {
			continue
		}
		conversation.WriteString(string(msg.Role))
		conversation.WriteString(": ")
		conversation.WriteString(text)
		conversation.WriteString("\n")
	}

	agent := NewAgent(provider,
		WithSystemPrompt(memoryExtractionPrompt),
		WithMaxTurns(1),
		WithMaxTokens(500),
	)

	var result strings.Builder
	for ev := range agent.Run(ctx, []Message{NewUserMessage(conversation.String())}) {
		if ev.Type == EventTextDelta && ev.TextDelta != nil {
			result.WriteString(ev.TextDelta.Text)
		}
	}

	if result.Len() == 0 {
		return nil, nil
	}

	// Parse bullet points.
	var memories []string
	for _, line := range strings.Split(result.String(), "\n") {
		line = strings.TrimSpace(line)
		line = strings.TrimPrefix(line, "- ")
		line = strings.TrimPrefix(line, "* ")
		if line != "" && len(line) > 5 {
			memories = append(memories, line)
		}
	}

	return memories, nil
}

const memoryExtractionPrompt = `Analyze the following conversation and extract key facts worth remembering for future conversations. Focus on:

- User preferences and personal information
- Important decisions made
- Key facts or data points discussed
- Tool results that produced useful information
- Action items or commitments

Output as a bullet list. Each bullet should be a self-contained fact.
Only include information that would be useful in future conversations.
If there are no memorable facts, output nothing.`

// --- Ensure io.EOF usage ---
var _ = io.EOF
