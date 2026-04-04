package skill_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/groq"
	"github.com/CanArslanDev/agentflow/skill"
	"github.com/CanArslanDev/agentflow/tools/builtin"
)

func groqProvider(t *testing.T) *groq.Provider {
	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		t.Skip("GROQ_API_KEY not set")
	}
	return groq.New(key, "llama-3.3-70b-versatile")
}

func TestIntegration_SkillSummarize(t *testing.T) {
	provider := groqProvider(t)
	registry := skill.NewRegistry()

	registry.Register(&skill.Skill{
		Name:         "summarize",
		Description:  "Summarize text into a concise paragraph",
		SystemPrompt: "You are a summarization expert. Provide a concise summary in 2-3 sentences maximum. Be direct and factual.",
		MaxTurns:     1,
		MaxTokens:    200,
	})

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(builtin.SkillTools(registry, provider)...),
		agentflow.WithSystemPrompt("You have access to skills. Use run_skill to execute them. Use list_skills to see available skills."),
		agentflow.WithMaxTurns(5),
		agentflow.WithMaxTokens(800),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	longText := `Go is a statically typed, compiled high-level programming language designed at Google
by Robert Griesemer, Rob Pike, and Ken Thompson. It is syntactically similar to C, but also has
memory safety, garbage collection, structural typing, and CSP-style concurrency. It was designed
to improve programming productivity in an era of multicore processors, networked machines, and
large codebases. Go was publicly announced in November 2009, and version 1.0 was released in
March 2012. It is used extensively at Google and in many open-source projects.`

	var skillRun bool
	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("Use the summarize skill to summarize this text:\n\n" + longText),
	}) {
		switch ev.Type {
		case agentflow.EventToolStart:
			t.Logf("Tool: %s", ev.ToolStart.ToolCall.Name)
			if ev.ToolStart.ToolCall.Name == "run_skill" {
				skillRun = true
			}
		case agentflow.EventToolEnd:
			t.Logf("Skill result (%d chars): %s...", len(ev.ToolEnd.Result.Content),
				ev.ToolEnd.Result.Content[:min(len(ev.ToolEnd.Result.Content), 150)])
		case agentflow.EventTurnEnd:
			t.Logf("Turn %d: %s", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		}
	}

	if !skillRun {
		t.Error("expected run_skill to be called")
	}
}

func TestIntegration_SkillTranslate(t *testing.T) {
	provider := groqProvider(t)
	registry := skill.NewRegistry()

	registry.Register(&skill.Skill{
		Name:         "translate",
		Description:  "Translate text to a target language",
		SystemPrompt: "You are a translator. Translate the given text accurately. Only output the translation, nothing else.",
		MaxTurns:     1,
		MaxTokens:    300,
	})
	registry.Register(&skill.Skill{
		Name:         "summarize",
		Description:  "Summarize text concisely",
		SystemPrompt: "Summarize in 1-2 sentences.",
		MaxTurns:     1,
		MaxTokens:    200,
	})

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(builtin.SkillTools(registry, provider)...),
		agentflow.WithSystemPrompt("You have skills available. Use list_skills to see them, run_skill to execute them."),
		agentflow.WithMaxTurns(5),
		agentflow.WithMaxTokens(800),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var toolNames []string
	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("Translate 'Hello, how are you today?' to Turkish using the translate skill."),
	}) {
		switch ev.Type {
		case agentflow.EventToolStart:
			toolNames = append(toolNames, ev.ToolStart.ToolCall.Name)
			t.Logf("Tool: %s", ev.ToolStart.ToolCall.Name)
		case agentflow.EventToolEnd:
			t.Logf("Result: %s", ev.ToolEnd.Result.Content)
		case agentflow.EventTurnEnd:
			t.Logf("Turn %d: %s", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		}
	}

	hasRunSkill := false
	for _, n := range toolNames {
		if n == "run_skill" {
			hasRunSkill = true
		}
	}
	if !hasRunSkill {
		t.Error("expected run_skill call")
	}
}

func TestIntegration_SkillListAndRun(t *testing.T) {
	provider := groqProvider(t)
	registry := skill.NewRegistry()

	registry.Register(&skill.Skill{
		Name:         "count_words",
		Description:  "Count the number of words in text",
		SystemPrompt: "Count the words in the given text. Reply with only the number.",
		MaxTurns:     1,
		MaxTokens:    50,
	})

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(builtin.SkillTools(registry, provider)...),
		agentflow.WithSystemPrompt("Use list_skills to discover skills, then use run_skill to execute them."),
		agentflow.WithMaxTurns(5),
		agentflow.WithMaxTokens(500),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var toolNames []string
	for ev := range agent.Run(ctx, []agentflow.Message{
		agentflow.NewUserMessage("List available skills, then use count_words to count words in: 'The quick brown fox jumps over the lazy dog'"),
	}) {
		if ev.Type == agentflow.EventToolStart {
			toolNames = append(toolNames, ev.ToolStart.ToolCall.Name)
			t.Logf("Tool: %s", ev.ToolStart.ToolCall.Name)
		}
		if ev.Type == agentflow.EventToolEnd {
			t.Logf("Result: %s", ev.ToolEnd.Result.Content[:min(len(ev.ToolEnd.Result.Content), 200)])
		}
		if ev.Type == agentflow.EventTurnEnd {
			t.Logf("Turn %d: %s", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		}
	}

	hasListSkills := false
	hasRunSkill := false
	for _, n := range toolNames {
		if n == "list_skills" {
			hasListSkills = true
		}
		if n == "run_skill" {
			hasRunSkill = true
		}
	}
	if !hasListSkills {
		t.Error("expected list_skills call")
	}
	if !hasRunSkill {
		t.Error("expected run_skill call")
	}
}
