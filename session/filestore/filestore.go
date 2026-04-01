// Package filestore provides a file-system-based SessionStore implementation.
// Each session is stored as a JSON file in a configurable directory.
// Suitable for single-process agents and local development.
package filestore

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/CanArslanDev/agentflow"
)

const fileExtension = ".session.json"

// Store persists sessions as individual JSON files in a directory.
type Store struct {
	dir string
}

// New creates a file-based session store. The directory is created if it
// does not exist. Returns an error if the directory cannot be created.
func New(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}
	return &Store{dir: dir}, nil
}

// Save writes a session to disk as a JSON file.
func (s *Store) Save(_ context.Context, session *agentflow.Session) error {
	if session.ID == "" {
		session.ID = agentflow.GenerateSessionID()
	}
	now := time.Now().UTC()
	if session.CreatedAt.IsZero() {
		session.CreatedAt = now
	}
	session.UpdatedAt = now

	data, err := json.MarshalIndent(session, "", "  ")
	if err != nil {
		return err
	}

	path := s.filePath(session.ID)

	// Atomic write: write to temp file, then rename.
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// Load reads a session from disk by ID. Returns ErrSessionNotFound if the
// file does not exist.
func (s *Store) Load(_ context.Context, id string) (*agentflow.Session, error) {
	data, err := os.ReadFile(s.filePath(id))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, agentflow.ErrSessionNotFound
		}
		return nil, err
	}

	var session agentflow.Session
	if err := json.Unmarshal(data, &session); err != nil {
		return nil, err
	}
	return &session, nil
}

// List scans the directory for session files and returns their metadata,
// ordered by most recently updated first.
func (s *Store) List(_ context.Context) ([]agentflow.SessionInfo, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return nil, err
	}

	var infos []agentflow.SessionInfo
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), fileExtension) {
			continue
		}

		path := filepath.Join(s.dir, entry.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}

		var session agentflow.Session
		if err := json.Unmarshal(data, &session); err != nil {
			continue
		}

		infos = append(infos, agentflow.SessionInfo{
			ID:        session.ID,
			CreatedAt: session.CreatedAt,
			UpdatedAt: session.UpdatedAt,
			ModelID:   session.ModelID,
			TurnCount: session.TurnCount,
			Metadata:  session.Metadata,
			Preview:   agentflow.SessionPreview(session.Messages, 80),
		})
	}

	sort.Slice(infos, func(i, j int) bool {
		return infos[i].UpdatedAt.After(infos[j].UpdatedAt)
	})
	return infos, nil
}

// Delete removes a session file. Returns nil if the file does not exist.
func (s *Store) Delete(_ context.Context, id string) error {
	err := os.Remove(s.filePath(id))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	return err
}

func (s *Store) filePath(id string) string {
	return filepath.Join(s.dir, id+fileExtension)
}
