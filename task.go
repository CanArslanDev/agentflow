package agentflow

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// TaskStatus represents the lifecycle state of a task.
type TaskStatus string

const (
	TaskPending    TaskStatus = "pending"
	TaskInProgress TaskStatus = "in_progress"
	TaskCompleted  TaskStatus = "completed"
	TaskFailed     TaskStatus = "failed"
)

// Task represents a unit of work that an agent creates, tracks, and completes.
type Task struct {
	ID          int            `json:"id"`
	Title       string         `json:"title"`
	Description string         `json:"description,omitempty"`
	Status      TaskStatus     `json:"status"`
	CreatedAt   time.Time      `json:"created_at"`
	UpdatedAt   time.Time      `json:"updated_at"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}

// TaskStore manages tasks for an agent run. It is safe for concurrent use.
type TaskStore struct {
	mu     sync.RWMutex
	tasks  map[int]*Task
	nextID atomic.Int64
}

// NewTaskStore creates an empty task store.
func NewTaskStore() *TaskStore {
	return &TaskStore{
		tasks: make(map[int]*Task),
	}
}

// Create adds a new task and returns its assigned ID.
func (s *TaskStore) Create(title, description string) *Task {
	id := int(s.nextID.Add(1))
	now := time.Now().UTC()

	task := &Task{
		ID:          id,
		Title:       title,
		Description: description,
		Status:      TaskPending,
		CreatedAt:   now,
		UpdatedAt:   now,
	}

	s.mu.Lock()
	s.tasks[id] = task
	s.mu.Unlock()

	return task
}

// Get returns a task by ID. Returns nil if not found.
func (s *TaskStore) Get(id int) *Task {
	s.mu.RLock()
	defer s.mu.RUnlock()

	task, ok := s.tasks[id]
	if !ok {
		return nil
	}
	cp := *task
	return &cp
}

// Update modifies a task's status and optional fields. Returns error if not found.
func (s *TaskStore) Update(id int, status TaskStatus, description string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	task, ok := s.tasks[id]
	if !ok {
		return fmt.Errorf("task %d not found", id)
	}

	task.Status = status
	task.UpdatedAt = time.Now().UTC()
	if description != "" {
		task.Description = description
	}
	return nil
}

// List returns all tasks ordered by ID.
func (s *TaskStore) List() []*Task {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make([]*Task, 0, len(s.tasks))
	for i := 1; i <= int(s.nextID.Load()); i++ {
		if task, ok := s.tasks[i]; ok {
			cp := *task
			result = append(result, &cp)
		}
	}
	return result
}

// Count returns the total number of tasks.
func (s *TaskStore) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.tasks)
}

// Summary returns a formatted string of all tasks.
func (s *TaskStore) Summary() string {
	tasks := s.List()
	if len(tasks) == 0 {
		return "No tasks."
	}

	var result string
	for _, t := range tasks {
		status := string(t.Status)
		switch t.Status {
		case TaskPending:
			status = "[ ] pending"
		case TaskInProgress:
			status = "[~] in progress"
		case TaskCompleted:
			status = "[x] completed"
		case TaskFailed:
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
