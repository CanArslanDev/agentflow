// Package task provides a thread-safe task store for agents to create, track,
// and manage units of work during execution.
package task

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// Status represents the lifecycle state of a task.
type Status string

const (
	Pending    Status = "pending"
	InProgress Status = "in_progress"
	Completed  Status = "completed"
	Failed     Status = "failed"
)

// Task represents a unit of work that an agent creates, tracks, and completes.
type Task struct {
	ID          int            `json:"id"`
	Title       string         `json:"title"`
	Description string         `json:"description,omitempty"`
	Status      Status         `json:"status"`
	CreatedAt   time.Time      `json:"created_at"`
	UpdatedAt   time.Time      `json:"updated_at"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}

// Store manages tasks for an agent run. It is safe for concurrent use.
type Store struct {
	mu     sync.RWMutex
	tasks  map[int]*Task
	nextID atomic.Int64
}

// NewStore creates an empty task store.
func NewStore() *Store {
	return &Store{
		tasks: make(map[int]*Task),
	}
}

// Create adds a new task and returns its assigned ID.
func (s *Store) Create(title, description string) *Task {
	id := int(s.nextID.Add(1))
	now := time.Now().UTC()

	t := &Task{
		ID:          id,
		Title:       title,
		Description: description,
		Status:      Pending,
		CreatedAt:   now,
		UpdatedAt:   now,
	}

	s.mu.Lock()
	s.tasks[id] = t
	s.mu.Unlock()

	return t
}

// Get returns a task by ID. Returns nil if not found.
func (s *Store) Get(id int) *Task {
	s.mu.RLock()
	defer s.mu.RUnlock()

	t, ok := s.tasks[id]
	if !ok {
		return nil
	}
	cp := *t
	return &cp
}

// Update modifies a task's status and optional fields. Returns error if not found.
func (s *Store) Update(id int, status Status, description string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	t, ok := s.tasks[id]
	if !ok {
		return fmt.Errorf("task %d not found", id)
	}

	t.Status = status
	t.UpdatedAt = time.Now().UTC()
	if description != "" {
		t.Description = description
	}
	return nil
}

// List returns all tasks ordered by ID.
func (s *Store) List() []*Task {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make([]*Task, 0, len(s.tasks))
	for i := 1; i <= int(s.nextID.Load()); i++ {
		if t, ok := s.tasks[i]; ok {
			cp := *t
			result = append(result, &cp)
		}
	}
	return result
}

// Count returns the total number of tasks.
func (s *Store) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.tasks)
}

// Summary returns a formatted string of all tasks.
func (s *Store) Summary() string {
	tasks := s.List()
	if len(tasks) == 0 {
		return "No tasks."
	}

	var result string
	for _, t := range tasks {
		status := string(t.Status)
		switch t.Status {
		case Pending:
			status = "[ ] pending"
		case InProgress:
			status = "[~] in progress"
		case Completed:
			status = "[x] completed"
		case Failed:
			status = "[!] failed"
		}
		result += fmt.Sprintf("#%d %s — %s", t.ID, status, t.Title)
		if t.Description != "" {
			result += " (" + t.Description + ")"
		}
		result += "\n"
	}
	return result
}
