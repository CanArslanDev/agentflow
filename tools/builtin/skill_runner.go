package builtin

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/skill"
)

// SkillTools returns tools for discovering and executing skills from a registry.
//
//	registry := skill.NewRegistry()
//	registry.Register(&skill.Skill{
//	    Name:         "summarize",
//	    Description:  "Summarize text concisely",
//	    SystemPrompt: "You are a summarization expert.",
//	})
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithTools(builtin.SkillTools(registry, provider)...),
//	)
func SkillTools(registry *skill.Registry, provider agentflow.Provider) []agentflow.Tool {
	return []agentflow.Tool{
		&runSkillTool{registry: registry, provider: provider},
		&listSkillsTool{registry: registry},
	}
}

// --- run_skill ---

type runSkillTool struct {
	registry *skill.Registry
	provider agentflow.Provider
}

func (t *runSkillTool) Name() string { return "run_skill" }
func (t *runSkillTool) Description() string {
	return "Execute a registered skill by name with the given input. Skills are specialized workflows (summarize, translate, analyze, etc.). Use list_skills first to see available skills."
}
func (t *runSkillTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"skill_name": map[string]any{"type": "string", "description": "Name of the skill to execute"},
			"input":      map[string]any{"type": "string", "description": "Input text or instructions for the skill"},
		},
		"required": []string{"skill_name", "input"},
	}
}
func (t *runSkillTool) Execute(ctx context.Context, input json.RawMessage, progress agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var p struct {
		SkillName string `json:"skill_name"`
		Input     string `json:"input"`
	}
	if err := json.Unmarshal(input, &p); err != nil {
		return &agentflow.ToolResult{Content: "invalid input: " + err.Error(), IsError: true}, nil
	}

	sk := t.registry.Get(p.SkillName)
	if sk == nil {
		available := t.registry.List()
		names := make([]string, len(available))
		for i, s := range available {
			names[i] = s.Name
		}
		return &agentflow.ToolResult{
			Content: fmt.Sprintf("skill %q not found. Available: %s", p.SkillName, strings.Join(names, ", ")),
			IsError: true,
		}, nil
	}

	if progress != nil {
		progress(agentflow.ProgressEvent{Message: "Running skill: " + sk.Name})
	}

	result, err := skill.Execute(ctx, t.provider, sk, p.Input)
	if err != nil {
		return &agentflow.ToolResult{Content: "skill failed: " + err.Error(), IsError: true}, nil
	}

	return &agentflow.ToolResult{
		Content:  result,
		Metadata: map[string]any{"skill": sk.Name},
	}, nil
}
func (t *runSkillTool) IsConcurrencySafe(_ json.RawMessage) bool { return true }
func (t *runSkillTool) IsReadOnly(_ json.RawMessage) bool        { return true }
func (t *runSkillTool) Locality() agentflow.ToolLocality          { return agentflow.ToolRemoteSafe }

// --- list_skills ---

type listSkillsTool struct {
	registry *skill.Registry
}

func (t *listSkillsTool) Name() string { return "list_skills" }
func (t *listSkillsTool) Description() string {
	return "List all available skills that can be executed with run_skill."
}
func (t *listSkillsTool) InputSchema() map[string]any {
	return map[string]any{"type": "object", "properties": map[string]any{}}
}
func (t *listSkillsTool) Execute(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	skills := t.registry.List()
	if len(skills) == 0 {
		return &agentflow.ToolResult{Content: "No skills registered."}, nil
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Available skills (%d):\n\n", len(skills)))
	for _, s := range skills {
		sb.WriteString(fmt.Sprintf("- %s: %s\n", s.Name, s.Description))
	}

	return &agentflow.ToolResult{Content: sb.String()}, nil
}
func (t *listSkillsTool) IsConcurrencySafe(_ json.RawMessage) bool { return true }
func (t *listSkillsTool) IsReadOnly(_ json.RawMessage) bool        { return true }
func (t *listSkillsTool) Locality() agentflow.ToolLocality          { return agentflow.ToolAny }
