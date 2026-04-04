// Package skill provides reusable workflow templates that agents can invoke
// by name. Skills define a system prompt and instructions that guide the agent
// through a specific task pattern (summarize, translate, analyze, etc.).
package skill

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/CanArslanDev/agentflow"
)

// Skill is a reusable workflow template that an agent can invoke by name.
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

// Registry manages available skills. Thread-safe for concurrent access.
type Registry struct {
	mu     sync.RWMutex
	skills map[string]*Skill
}

// NewRegistry creates an empty registry.
func NewRegistry() *Registry {
	return &Registry{
		skills: make(map[string]*Skill),
	}
}

// Register adds a skill to the registry. Overwrites if name already exists.
func (r *Registry) Register(skill *Skill) {
	r.mu.Lock()
	r.skills[skill.Name] = skill
	r.mu.Unlock()
}

// Get returns a skill by name. Returns nil if not found.
func (r *Registry) Get(name string) *Skill {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.skills[name]
}

// List returns all registered skill names and descriptions.
func (r *Registry) List() []Skill {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make([]Skill, 0, len(r.skills))
	for _, s := range r.skills {
		result = append(result, *s)
	}
	return result
}

// Execute runs a skill as a sub-agent call. The skill's system prompt
// overrides the parent agent's prompt. Returns the skill's text response.
func Execute(ctx context.Context, provider agentflow.Provider, s *Skill, input string) (string, error) {
	maxTurns := s.MaxTurns
	if maxTurns <= 0 {
		maxTurns = 3
	}
	maxTokens := s.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 1024
	}

	agent := agentflow.NewAgent(provider,
		agentflow.WithSystemPrompt(s.SystemPrompt),
		agentflow.WithMaxTurns(maxTurns),
		agentflow.WithMaxTokens(maxTokens),
	)

	var result strings.Builder
	for ev := range agent.Run(ctx, []agentflow.Message{agentflow.NewUserMessage(input)}) {
		if ev.Type == agentflow.EventTextDelta && ev.TextDelta != nil {
			result.WriteString(ev.TextDelta.Text)
		}
	}

	if result.Len() == 0 {
		return "", fmt.Errorf("skill %q produced no output", s.Name)
	}
	return result.String(), nil
}

// Parse parses a skill definition from a simple text format:
//
//	---
//	name: summarize
//	description: Summarize text concisely
//	max_turns: 1
//	max_tokens: 500
//	---
//	You are a summarization expert. Provide clear, concise summaries.
func Parse(content string) (*Skill, error) {
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

	s := &Skill{
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
			s.Name = val
		case "description":
			s.Description = val
		case "max_turns":
			fmt.Sscanf(val, "%d", &s.MaxTurns)
		case "max_tokens":
			fmt.Sscanf(val, "%d", &s.MaxTokens)
		}
	}

	if s.Name == "" {
		return nil, fmt.Errorf("skill name is required in frontmatter")
	}
	if s.SystemPrompt == "" {
		return nil, fmt.Errorf("skill system prompt (body) is required")
	}

	return s, nil
}
