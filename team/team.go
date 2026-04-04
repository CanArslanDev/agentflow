// Package team provides multi-agent coordination with shared communication
// channels and memory. Team members run as independent agents that can
// exchange messages through a mailbox and share state through shared memory.
package team

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/CanArslanDev/agentflow"
)

// Member is an agent within a team, identified by a role name.
type Member struct {
	// Role is the unique identifier for this member (e.g., "researcher", "writer").
	Role string

	// SystemPrompt defines the member's specialization.
	SystemPrompt string

	// Tools available to this member. nil inherits from the team's shared tools.
	Tools []agentflow.Tool

	// MaxTurns for this member's individual runs.
	MaxTurns int

	// MaxTokens for this member's responses.
	MaxTokens int
}

// Mailbox enables inter-agent communication within a team. Messages are
// keyed by recipient role and delivered in FIFO order.
type Mailbox struct {
	mu       sync.Mutex
	messages map[string][]MailMessage
	notify   map[string]chan struct{}
}

// MailMessage is a message sent between team members.
type MailMessage struct {
	From    string `json:"from"`
	To      string `json:"to"`
	Content string `json:"content"`
}

// NewMailbox creates an empty mailbox.
func NewMailbox() *Mailbox {
	return &Mailbox{
		messages: make(map[string][]MailMessage),
		notify:   make(map[string]chan struct{}),
	}
}

// Send delivers a message to the recipient's inbox.
func (m *Mailbox) Send(msg MailMessage) {
	m.mu.Lock()
	m.messages[msg.To] = append(m.messages[msg.To], msg)
	ch, ok := m.notify[msg.To]
	m.mu.Unlock()
	if ok {
		select {
		case ch <- struct{}{}:
		default:
		}
	}
}

// Receive returns all unread messages for the given role and clears them.
func (m *Mailbox) Receive(role string) []MailMessage {
	m.mu.Lock()
	defer m.mu.Unlock()
	msgs := m.messages[role]
	m.messages[role] = nil
	return msgs
}

// HasMessages returns true if there are unread messages for the role.
func (m *Mailbox) HasMessages(role string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.messages[role]) > 0
}

// SharedMemory is a thread-safe key-value store accessible to all team members.
type SharedMemory struct {
	mu   sync.RWMutex
	data map[string]string
}

// NewSharedMemory creates an empty shared memory.
func NewSharedMemory() *SharedMemory {
	return &SharedMemory{data: make(map[string]string)}
}

// Set stores a value.
func (m *SharedMemory) Set(key, value string) {
	m.mu.Lock()
	m.data[key] = value
	m.mu.Unlock()
}

// Get retrieves a value. Returns empty string if not found.
func (m *SharedMemory) Get(key string) string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.data[key]
}

// All returns a copy of all stored data.
func (m *SharedMemory) All() map[string]string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	cp := make(map[string]string, len(m.data))
	for k, v := range m.data {
		cp[k] = v
	}
	return cp
}

// Summary returns a formatted string of all shared memory entries.
func (m *SharedMemory) Summary() string {
	data := m.All()
	if len(data) == 0 {
		return "(empty)"
	}
	var result string
	for k, v := range data {
		result += fmt.Sprintf("%s: %s\n", k, v)
	}
	return result
}

// Team coordinates multiple agents working together on a shared goal.
// Members can communicate via a shared mailbox and access shared memory.
type Team struct {
	provider     agentflow.Provider
	members      map[string]*Member
	mailbox      *Mailbox
	sharedMemory *SharedMemory
	hooks        []agentflow.Hook
	permission   agentflow.PermissionChecker
}

// New creates a team with the given provider and members.
func New(provider agentflow.Provider, members []Member, opts ...Option) *Team {
	t := &Team{
		provider:     provider,
		members:      make(map[string]*Member, len(members)),
		mailbox:      NewMailbox(),
		sharedMemory: NewSharedMemory(),
		permission:   agentflow.AllowAll(),
	}
	for i := range members {
		t.members[members[i].Role] = &members[i]
	}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

// Option configures a Team.
type Option func(*Team)

// WithHooks sets hooks shared by all team members.
func WithHooks(hooks ...agentflow.Hook) Option {
	return func(t *Team) { t.hooks = hooks }
}

// WithPermission sets the permission checker for all members.
func WithPermission(p agentflow.PermissionChecker) Option {
	return func(t *Team) { t.permission = p }
}

// RunMember executes a single team member's agent with access to the mailbox
// and shared memory. Returns the event channel.
func (t *Team) RunMember(ctx context.Context, role string, task string) (<-chan agentflow.Event, error) {
	member, ok := t.members[role]
	if !ok {
		return nil, fmt.Errorf("team member %q not found", role)
	}

	// Build system prompt with team context.
	prompt := member.SystemPrompt
	prompt += fmt.Sprintf("\n\nYour role: %s\nTeam members: %s",
		role, t.memberRoles())

	// Include any unread messages.
	msgs := t.mailbox.Receive(role)
	if len(msgs) > 0 {
		prompt += "\n\nMessages for you:"
		for _, m := range msgs {
			prompt += fmt.Sprintf("\n- From %s: %s", m.From, m.Content)
		}
	}

	// Include shared memory context.
	memory := t.sharedMemory.Summary()
	if memory != "(empty)" {
		prompt += "\n\nShared team knowledge:\n" + memory
	}

	maxTurns := member.MaxTurns
	if maxTurns == 0 {
		maxTurns = 5
	}
	maxTokens := member.MaxTokens
	if maxTokens == 0 {
		maxTokens = 1024
	}

	var tools []agentflow.Tool
	if member.Tools != nil {
		tools = member.Tools
	}

	// Add team communication tools.
	tools = append(tools,
		&sendMessageTool{mailbox: t.mailbox, senderRole: role},
		&readMessagesTool{mailbox: t.mailbox, role: role},
		&setMemoryTool{memory: t.sharedMemory, role: role},
		&getMemoryTool{memory: t.sharedMemory},
	)

	agent := agentflow.NewAgent(t.provider,
		agentflow.WithTools(tools...),
		agentflow.WithSystemPrompt(prompt),
		agentflow.WithMaxTurns(maxTurns),
		agentflow.WithMaxTokens(maxTokens),
		agentflow.WithPermission(t.permission),
	)
	for _, h := range t.hooks {
		agent.AddHook(h)
	}

	return agent.Run(ctx, []agentflow.Message{agentflow.NewUserMessage(task)}), nil
}

// RunAll executes all team members in parallel with their individual tasks.
// Returns a map of role -> final text response.
func (t *Team) RunAll(ctx context.Context, tasks map[string]string) map[string]Result {
	results := make(map[string]Result, len(tasks))
	var mu sync.Mutex
	var wg sync.WaitGroup

	for role, task := range tasks {
		wg.Add(1)
		go func(r, tsk string) {
			defer wg.Done()

			events, err := t.RunMember(ctx, r, tsk)
			if err != nil {
				mu.Lock()
				results[r] = Result{Role: r, Error: err}
				mu.Unlock()
				return
			}

			var text string
			for ev := range events {
				if ev.Type == agentflow.EventTextDelta && ev.TextDelta != nil {
					text += ev.TextDelta.Text
				}
			}

			mu.Lock()
			results[r] = Result{Role: r, Response: text}
			mu.Unlock()
		}(role, task)
	}

	wg.Wait()
	return results
}

// Result holds the outcome of a team member's execution.
type Result struct {
	Role     string
	Response string
	Error    error
}

func (t *Team) memberRoles() string {
	roles := make([]string, 0, len(t.members))
	for role := range t.members {
		roles = append(roles, role)
	}
	result := ""
	for i, r := range roles {
		if i > 0 {
			result += ", "
		}
		result += r
	}
	return result
}

// --- Team Communication Tools ---

type sendMessageTool struct {
	mailbox    *Mailbox
	senderRole string
}

func (t *sendMessageTool) Name() string { return "send_message" }
func (t *sendMessageTool) Description() string {
	return "Send a message to another team member. They will receive it on their next turn."
}
func (t *sendMessageTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"to":      map[string]any{"type": "string", "description": "Role name of the recipient"},
			"message": map[string]any{"type": "string", "description": "Message content"},
		},
		"required": []string{"to", "message"},
	}
}
func (t *sendMessageTool) Execute(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var p struct {
		To      string `json:"to"`
		Message string `json:"message"`
	}
	if err := json.Unmarshal(input, &p); err != nil {
		return &agentflow.ToolResult{Content: err.Error(), IsError: true}, nil
	}
	t.mailbox.Send(MailMessage{From: t.senderRole, To: p.To, Content: p.Message})
	return &agentflow.ToolResult{Content: fmt.Sprintf("Message sent to %s", p.To)}, nil
}
func (t *sendMessageTool) IsConcurrencySafe(_ json.RawMessage) bool { return false }
func (t *sendMessageTool) IsReadOnly(_ json.RawMessage) bool        { return false }
func (t *sendMessageTool) Locality() agentflow.ToolLocality          { return agentflow.ToolAny }

type readMessagesTool struct {
	mailbox *Mailbox
	role    string
}

func (t *readMessagesTool) Name() string        { return "read_messages" }
func (t *readMessagesTool) Description() string { return "Read unread messages from other team members." }
func (t *readMessagesTool) InputSchema() map[string]any {
	return map[string]any{"type": "object", "properties": map[string]any{}}
}
func (t *readMessagesTool) Execute(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	msgs := t.mailbox.Receive(t.role)
	if len(msgs) == 0 {
		return &agentflow.ToolResult{Content: "No new messages."}, nil
	}
	var result string
	for _, m := range msgs {
		result += fmt.Sprintf("From %s: %s\n", m.From, m.Content)
	}
	return &agentflow.ToolResult{Content: result}, nil
}
func (t *readMessagesTool) IsConcurrencySafe(_ json.RawMessage) bool { return true }
func (t *readMessagesTool) IsReadOnly(_ json.RawMessage) bool        { return true }
func (t *readMessagesTool) Locality() agentflow.ToolLocality          { return agentflow.ToolAny }

type setMemoryTool struct {
	memory *SharedMemory
	role   string
}

func (t *setMemoryTool) Name() string { return "set_shared_memory" }
func (t *setMemoryTool) Description() string {
	return "Store information in shared team memory. Other team members can access this."
}
func (t *setMemoryTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"key":   map[string]any{"type": "string", "description": "Memory key"},
			"value": map[string]any{"type": "string", "description": "Value to store"},
		},
		"required": []string{"key", "value"},
	}
}
func (t *setMemoryTool) Execute(_ context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var p struct {
		Key   string `json:"key"`
		Value string `json:"value"`
	}
	if err := json.Unmarshal(input, &p); err != nil {
		return &agentflow.ToolResult{Content: err.Error(), IsError: true}, nil
	}
	t.memory.Set(p.Key, p.Value)
	return &agentflow.ToolResult{Content: fmt.Sprintf("Stored: %s = %s", p.Key, p.Value)}, nil
}
func (t *setMemoryTool) IsConcurrencySafe(_ json.RawMessage) bool { return false }
func (t *setMemoryTool) IsReadOnly(_ json.RawMessage) bool        { return false }
func (t *setMemoryTool) Locality() agentflow.ToolLocality          { return agentflow.ToolAny }

type getMemoryTool struct {
	memory *SharedMemory
}

func (t *getMemoryTool) Name() string        { return "get_shared_memory" }
func (t *getMemoryTool) Description() string { return "Read all shared team memory entries." }
func (t *getMemoryTool) InputSchema() map[string]any {
	return map[string]any{"type": "object", "properties": map[string]any{}}
}
func (t *getMemoryTool) Execute(_ context.Context, _ json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	return &agentflow.ToolResult{Content: t.memory.Summary()}, nil
}
func (t *getMemoryTool) IsConcurrencySafe(_ json.RawMessage) bool { return true }
func (t *getMemoryTool) IsReadOnly(_ json.RawMessage) bool        { return true }
func (t *getMemoryTool) Locality() agentflow.ToolLocality          { return agentflow.ToolAny }
