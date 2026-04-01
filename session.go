package agentflow

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"time"
)

// SessionStore persists conversation state so agents can survive restarts,
// resume interrupted work, and maintain history across multiple invocations.
//
// Implementations must be safe for concurrent use from multiple goroutines.
type SessionStore interface {
	// Save persists the current conversation state. If a session with the given
	// ID already exists, it is overwritten.
	Save(ctx context.Context, session *Session) error

	// Load retrieves a previously saved session by ID. Returns ErrSessionNotFound
	// if the session does not exist.
	Load(ctx context.Context, id string) (*Session, error)

	// List returns metadata for all stored sessions, ordered by last update time
	// (most recent first).
	List(ctx context.Context) ([]SessionInfo, error)

	// Delete removes a session by ID. Returns nil if the session does not exist.
	Delete(ctx context.Context, id string) error
}

// Session holds the complete state needed to resume an agent conversation.
type Session struct {
	// ID uniquely identifies this session. Generated automatically if empty.
	ID string `json:"id"`

	// Messages is the full conversation history at the time of save.
	Messages []Message `json:"messages"`

	// Metadata is an arbitrary key-value bag for application-specific data
	// (user ID, task description, tags, etc.).
	Metadata map[string]any `json:"metadata,omitempty"`

	// CreatedAt is when the session was first created.
	CreatedAt time.Time `json:"created_at"`

	// UpdatedAt is when the session was last saved.
	UpdatedAt time.Time `json:"updated_at"`

	// ModelID records which model was used, for reference when resuming.
	ModelID string `json:"model_id,omitempty"`

	// TurnCount records how many turns were completed before saving.
	TurnCount int `json:"turn_count"`
}

// SessionInfo is a lightweight summary returned by SessionStore.List.
type SessionInfo struct {
	ID        string         `json:"id"`
	CreatedAt time.Time      `json:"created_at"`
	UpdatedAt time.Time      `json:"updated_at"`
	ModelID   string         `json:"model_id,omitempty"`
	TurnCount int            `json:"turn_count"`
	Metadata  map[string]any `json:"metadata,omitempty"`

	// Preview is the first user message text, truncated for display.
	Preview string `json:"preview,omitempty"`
}

// GenerateSessionID creates a cryptographically random session identifier.
func GenerateSessionID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return hex.EncodeToString(b)
}

// SessionPreview extracts a preview string from the first user message.
func SessionPreview(messages []Message, maxLen int) string {
	for _, msg := range messages {
		if msg.Role == RoleUser {
			text := msg.TextContent()
			if len(text) > maxLen {
				return text[:maxLen] + "..."
			}
			return text
		}
	}
	return ""
}
