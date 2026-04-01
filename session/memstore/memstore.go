// Package memstore provides an in-memory SessionStore implementation.
// Useful for testing and short-lived applications where persistence across
// process restarts is not needed.
package memstore

import (
	"context"
	"sort"
	"sync"
	"time"

	"github.com/canarslan/agentflow"
)

// Store is a thread-safe in-memory session store.
type Store struct {
	mu       sync.RWMutex
	sessions map[string]*agentflow.Session
}

// New creates an empty in-memory store.
func New() *Store {
	return &Store{
		sessions: make(map[string]*agentflow.Session),
	}
}

// Save stores or overwrites a session in memory.
func (s *Store) Save(_ context.Context, session *agentflow.Session) error {
	if session.ID == "" {
		session.ID = agentflow.GenerateSessionID()
	}
	now := time.Now().UTC()
	if session.CreatedAt.IsZero() {
		session.CreatedAt = now
	}
	session.UpdatedAt = now

	// Deep copy to prevent external mutation.
	cp := *session
	cp.Messages = make([]agentflow.Message, len(session.Messages))
	copy(cp.Messages, session.Messages)
	if session.Metadata != nil {
		cp.Metadata = make(map[string]any, len(session.Metadata))
		for k, v := range session.Metadata {
			cp.Metadata[k] = v
		}
	}

	s.mu.Lock()
	s.sessions[cp.ID] = &cp
	s.mu.Unlock()
	return nil
}

// Load retrieves a session by ID. Returns ErrSessionNotFound if not found.
func (s *Store) Load(_ context.Context, id string) (*agentflow.Session, error) {
	s.mu.RLock()
	session, ok := s.sessions[id]
	s.mu.RUnlock()

	if !ok {
		return nil, agentflow.ErrSessionNotFound
	}

	// Return a copy.
	cp := *session
	cp.Messages = make([]agentflow.Message, len(session.Messages))
	copy(cp.Messages, session.Messages)
	return &cp, nil
}

// List returns all sessions ordered by UpdatedAt descending.
func (s *Store) List(_ context.Context) ([]agentflow.SessionInfo, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	infos := make([]agentflow.SessionInfo, 0, len(s.sessions))
	for _, sess := range s.sessions {
		infos = append(infos, agentflow.SessionInfo{
			ID:        sess.ID,
			CreatedAt: sess.CreatedAt,
			UpdatedAt: sess.UpdatedAt,
			ModelID:   sess.ModelID,
			TurnCount: sess.TurnCount,
			Metadata:  sess.Metadata,
			Preview:   agentflow.SessionPreview(sess.Messages, 80),
		})
	}

	sort.Slice(infos, func(i, j int) bool {
		return infos[i].UpdatedAt.After(infos[j].UpdatedAt)
	})
	return infos, nil
}

// Delete removes a session by ID. Returns nil if not found.
func (s *Store) Delete(_ context.Context, id string) error {
	s.mu.Lock()
	delete(s.sessions, id)
	s.mu.Unlock()
	return nil
}

// Count returns the number of stored sessions.
func (s *Store) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.sessions)
}
