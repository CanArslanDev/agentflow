// Package plan provides a two-phase planning workflow for agents. The Plan
// function creates a structured plan without executing tools, and PlanAndExecute
// chains planning with tool-powered execution.
package plan

import (
	"context"
	"strings"

	"github.com/CanArslanDev/agentflow"
)

// Result holds the output of a planning phase.
type Result struct {
	Plan     string             // The structured plan text.
	Messages []agentflow.Message // Full conversation from planning.
}

// Plan runs the agent in planning-only mode: the model creates a structured plan
// but does not execute any tools. Returns the plan text that can be reviewed
// before execution.
//
//	result, err := plan.Plan(ctx, provider, "Build a REST API in Go")
//	fmt.Println(result.Plan)
func Plan(ctx context.Context, provider agentflow.Provider, task string) (*Result, error) {
	agent := agentflow.NewAgent(provider,
		agentflow.WithSystemPrompt(systemPrompt),
		agentflow.WithMaxTurns(1),
		agentflow.WithMaxTokens(2048),
	)

	var planText strings.Builder
	var messages []agentflow.Message

	for ev := range agent.Run(ctx, []agentflow.Message{agentflow.NewUserMessage(task)}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			if ev.TextDelta != nil {
				planText.WriteString(ev.TextDelta.Text)
			}
		case agentflow.EventTurnEnd:
			if ev.TurnEnd != nil {
				messages = ev.TurnEnd.Messages
			}
		}
	}

	if planText.Len() == 0 {
		return nil, agentflow.ErrProviderUnavailable
	}

	return &Result{
		Plan:     planText.String(),
		Messages: messages,
	}, nil
}

// PlanAndExecute first creates a plan, then executes it in a second agent run
// with full tool access. Events from both phases are streamed through the channel.
func PlanAndExecute(ctx context.Context, provider agentflow.Provider, task string, tools []agentflow.Tool, opts ...agentflow.Option) <-chan agentflow.Event {
	events := make(chan agentflow.Event, agentflow.DefaultEventBufferSize)

	go func() {
		defer close(events)

		// Phase 1: Plan.
		events <- agentflow.Event{
			Type:      agentflow.EventTurnStart,
			TurnStart: &agentflow.TurnStartEvent{TurnNumber: 0}, // Turn 0 = planning phase.
		}

		planAgent := agentflow.NewAgent(provider,
			agentflow.WithSystemPrompt(systemPrompt),
			agentflow.WithMaxTurns(1),
			agentflow.WithMaxTokens(2048),
		)

		var planText strings.Builder
		var planMessages []agentflow.Message

		for ev := range planAgent.Run(ctx, []agentflow.Message{agentflow.NewUserMessage(task)}) {
			switch ev.Type {
			case agentflow.EventTextDelta:
				events <- ev
				if ev.TextDelta != nil {
					planText.WriteString(ev.TextDelta.Text)
				}
			case agentflow.EventTurnEnd:
				if ev.TurnEnd != nil {
					planMessages = ev.TurnEnd.Messages
				}
			}
		}

		if planText.Len() == 0 || ctx.Err() != nil {
			events <- agentflow.Event{
				Type:    agentflow.EventTurnEnd,
				TurnEnd: &agentflow.TurnEndEvent{TurnNumber: 0, Reason: agentflow.TurnEndError},
			}
			return
		}

		events <- agentflow.Event{
			Type:    agentflow.EventTurnEnd,
			TurnEnd: &agentflow.TurnEndEvent{TurnNumber: 0, Reason: agentflow.TurnEndComplete, Messages: planMessages},
		}

		// Phase 2: Execute with tools.
		execOpts := append([]agentflow.Option{
			agentflow.WithTools(tools...),
			agentflow.WithSystemPrompt("Execute the following plan step by step. Use the available tools.\n\nPlan:\n" + planText.String()),
		}, opts...)

		execAgent := agentflow.NewAgent(provider, execOpts...)

		for ev := range execAgent.Run(ctx, []agentflow.Message{agentflow.NewUserMessage("Execute the plan now.")}) {
			events <- ev
		}
	}()

	return events
}

const systemPrompt = `You are a planning specialist. When given a task:

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

// ExtractMemories analyzes a conversation and extracts key facts worth remembering.
// Returns a list of memory entries that can be stored for future context.
func ExtractMemories(ctx context.Context, provider agentflow.Provider, messages []agentflow.Message) ([]string, error) {
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

	agent := agentflow.NewAgent(provider,
		agentflow.WithSystemPrompt(memoryExtractionPrompt),
		agentflow.WithMaxTurns(1),
		agentflow.WithMaxTokens(500),
	)

	var result strings.Builder
	for ev := range agent.Run(ctx, []agentflow.Message{agentflow.NewUserMessage(conversation.String())}) {
		if ev.Type == agentflow.EventTextDelta && ev.TextDelta != nil {
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
