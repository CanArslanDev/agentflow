package agentflow

import (
	"context"
	"fmt"
	"io"
	"strings"
	"sync"
)

// Skill is a reusable workflow template that an agent can invoke by name.
// Skills define a system prompt and instructions that guide the agent through
// a specific task pattern (summarize, translate, analyze, etc.).
type Skill struct {
	// Name is the unique identifier used to invoke the skill.
	Name string `json:"name"`

	// Description explains what the skill does (shown to the model).
	Description string `json:"description"`

	// SystemPrompt is the specialized system prompt for skill execution.
	SystemPrompt string `json:"system_prompt"`

	// MaxTurns limits the skill execution (0 = inherit from agent).
	MaxTurns int `json:"max_turns,omitempty"`

	// MaxTokens limits the response length (0 = inherit from agent).
	MaxTokens int `json:"max_tokens,omitempty"`
}

// SkillRegistry manages available skills. Thread-safe for concurrent access.
type SkillRegistry struct {
	mu     sync.RWMutex
	skills map[string]*Skill
}

// NewSkillRegistry creates an empty registry.
func NewSkillRegistry() *SkillRegistry {
	return &SkillRegistry{
		skills: make(map[string]*Skill),
	}
}

// Register adds a skill to the registry. Overwrites if name already exists.
func (r *SkillRegistry) Register(skill *Skill) {
	r.mu.Lock()
	r.skills[skill.Name] = skill
	r.mu.Unlock()
}

// Get returns a skill by name. Returns nil if not found.
func (r *SkillRegistry) Get(name string) *Skill {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.skills[name]
}

// List returns all registered skill names and descriptions.
func (r *SkillRegistry) List() []Skill {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make([]Skill, 0, len(r.skills))
	for _, s := range r.skills {
		result = append(result, *s)
	}
	return result
}

// ExecuteSkill runs a skill as a sub-agent call. The skill's system prompt
// overrides the parent agent's prompt. Returns the skill's text response.
func ExecuteSkill(ctx context.Context, provider Provider, skill *Skill, input string) (string, error) {
	maxTurns := skill.MaxTurns
	if maxTurns <= 0 {
		maxTurns = 3
	}
	maxTokens := skill.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 1024
	}

	agent := NewAgent(provider,
		WithSystemPrompt(skill.SystemPrompt),
		WithMaxTurns(maxTurns),
		WithMaxTokens(maxTokens),
	)

	var result strings.Builder
	for ev := range agent.Run(ctx, []Message{NewUserMessage(input)}) {
		if ev.Type == EventTextDelta && ev.TextDelta != nil {
			result.WriteString(ev.TextDelta.Text)
		}
	}

	if result.Len() == 0 {
		return "", fmt.Errorf("skill %q produced no output", skill.Name)
	}
	return result.String(), nil
}

// ParseSkill parses a skill definition from a simple text format:
//
//	---
//	name: summarize
//	description: Summarize text concisely
//	max_turns: 1
//	max_tokens: 500
//	---
//	You are a summarization expert. Provide clear, concise summaries.
func ParseSkill(content string) (*Skill, error) {
	content = strings.TrimSpace(content)
	if !strings.HasPrefix(content, "---") {
		return nil, fmt.Errorf("skill must start with --- frontmatter delimiter")
	}

	// Split frontmatter and body.
	parts := strings.SplitN(content[3:], "---", 2)
	if len(parts) < 2 {
		return nil, fmt.Errorf("missing closing --- frontmatter delimiter")
	}

	frontmatter := strings.TrimSpace(parts[0])
	body := strings.TrimSpace(parts[1])

	skill := &Skill{
		SystemPrompt: body,
	}

	// Parse frontmatter key-value pairs.
	for _, line := range strings.Split(frontmatter, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		kv := strings.SplitN(line, ":", 2)
		if len(kv) != 2 {
			continue
		}
		key := strings.TrimSpace(kv[0])
		val := strings.TrimSpace(kv[1])

		switch key {
		case "name":
			skill.Name = val
		case "description":
			skill.Description = val
		case "max_turns":
			fmt.Sscanf(val, "%d", &skill.MaxTurns)
		case "max_tokens":
			fmt.Sscanf(val, "%d", &skill.MaxTokens)
		}
	}

	if skill.Name == "" {
		return nil, fmt.Errorf("skill name is required in frontmatter")
	}
	if skill.SystemPrompt == "" {
		return nil, fmt.Errorf("skill system prompt (body) is required")
	}

	return skill, nil
}

// Ensure Stream implements io.EOF convention.
var _ error = io.EOF
