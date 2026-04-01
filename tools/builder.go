// Package tools provides utilities for building agentflow tools with a fluent API.
//
// The ToolBuilder eliminates boilerplate when defining tools:
//
//	searchTool := tools.New("web_search", "Search the web for information").
//	    WithSchema(map[string]any{
//	        "type": "object",
//	        "properties": map[string]any{
//	            "query": map[string]any{"type": "string", "description": "Search query"},
//	        },
//	        "required": []string{"query"},
//	    }).
//	    ReadOnly(true).
//	    ConcurrencySafe(true).
//	    WithExecute(func(ctx context.Context, input json.RawMessage, progress agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
//	        // implementation
//	    }).
//	    Build()
package tools

import (
	"context"
	"encoding/json"

	"github.com/CanArslanDev/agentflow"
)

// ExecuteFunc is the function signature for tool execution.
type ExecuteFunc func(ctx context.Context, input json.RawMessage, progress agentflow.ProgressFunc) (*agentflow.ToolResult, error)

// Builder provides a fluent API for constructing tools.
type Builder struct {
	name            string
	description     string
	schema          map[string]any
	executeFn       ExecuteFunc
	concurrencySafe bool
	readOnly        bool
	locality        agentflow.ToolLocality
}

// New creates a Builder with the given name and description.
func New(name, description string) *Builder {
	return &Builder{
		name:        name,
		description: description,
		schema: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
	}
}

// WithSchema sets the JSON Schema for the tool's input parameters.
func (b *Builder) WithSchema(schema map[string]any) *Builder {
	b.schema = schema
	return b
}

// ReadOnly marks the tool as read-only.
func (b *Builder) ReadOnly(v bool) *Builder {
	b.readOnly = v
	return b
}

// ConcurrencySafe marks the tool as safe for parallel execution.
func (b *Builder) ConcurrencySafe(v bool) *Builder {
	b.concurrencySafe = v
	return b
}

// WithLocality sets the tool's execution environment compatibility.
func (b *Builder) WithLocality(l agentflow.ToolLocality) *Builder {
	b.locality = l
	return b
}

// RemoteSafe is a shorthand for WithLocality(agentflow.ToolRemoteSafe).
func (b *Builder) RemoteSafe() *Builder {
	b.locality = agentflow.ToolRemoteSafe
	return b
}

// WithExecute sets the execution function.
func (b *Builder) WithExecute(fn ExecuteFunc) *Builder {
	b.executeFn = fn
	return b
}

// Build returns the completed Tool. Panics if no execute function is set.
func (b *Builder) Build() agentflow.Tool {
	if b.executeFn == nil {
		panic("agentflow/tools: Build() called without WithExecute()")
	}
	return &builtTool{
		name:            b.name,
		description:     b.description,
		schema:          b.schema,
		executeFn:       b.executeFn,
		concurrencySafe: b.concurrencySafe,
		readOnly:        b.readOnly,
		locality:        b.locality,
	}
}

// builtTool implements agentflow.Tool and agentflow.LocalityAware.
type builtTool struct {
	name            string
	description     string
	schema          map[string]any
	executeFn       ExecuteFunc
	concurrencySafe bool
	readOnly        bool
	locality        agentflow.ToolLocality
}

func (t *builtTool) Name() string                     { return t.name }
func (t *builtTool) Description() string               { return t.description }
func (t *builtTool) InputSchema() map[string]any       { return t.schema }
func (t *builtTool) IsConcurrencySafe(_ json.RawMessage) bool { return t.concurrencySafe }
func (t *builtTool) IsReadOnly(_ json.RawMessage) bool        { return t.readOnly }

func (t *builtTool) Locality() agentflow.ToolLocality { return t.locality }

func (t *builtTool) Execute(ctx context.Context, input json.RawMessage, progress agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	return t.executeFn(ctx, input, progress)
}
