package tools_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/tools"
)

type SearchInput struct {
	Query      string `json:"query" description:"Search query"`
	MaxResults int    `json:"max_results,omitempty" description:"Max results"`
}

func TestNewTyped_SchemaGeneration(t *testing.T) {
	tool := tools.NewTyped[SearchInput](
		"search", "Search tool",
		[]string{"query"},
		func(_ context.Context, input SearchInput, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{Content: input.Query}, nil
		},
	)

	if tool.Name() != "search" {
		t.Errorf("name: %s", tool.Name())
	}

	schema := tool.InputSchema()
	if schema["type"] != "object" {
		t.Errorf("schema type: %v", schema["type"])
	}

	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("no properties in schema")
	}

	queryProp, ok := props["query"].(map[string]any)
	if !ok {
		t.Fatal("no query property")
	}
	if queryProp["type"] != "string" {
		t.Errorf("query type: %v", queryProp["type"])
	}
	if queryProp["description"] != "Search query" {
		t.Errorf("query description: %v", queryProp["description"])
	}

	maxProp, ok := props["max_results"].(map[string]any)
	if !ok {
		t.Fatal("no max_results property")
	}
	if maxProp["type"] != "integer" {
		t.Errorf("max_results type: %v", maxProp["type"])
	}

	required, ok := schema["required"].([]string)
	if !ok || len(required) != 1 || required[0] != "query" {
		t.Errorf("required: %v", schema["required"])
	}
}

func TestNewTyped_Execution(t *testing.T) {
	tool := tools.NewTyped[SearchInput](
		"search", "Search tool",
		[]string{"query"},
		func(_ context.Context, input SearchInput, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{Content: "results for: " + input.Query}, nil
		},
	)

	result, err := tool.Execute(context.Background(), json.RawMessage(`{"query":"golang"}`), nil)
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	if result.Content != "results for: golang" {
		t.Errorf("content: %s", result.Content)
	}
}

func TestNewTyped_InvalidInput(t *testing.T) {
	tool := tools.NewTyped[SearchInput](
		"search", "Search tool",
		nil,
		func(_ context.Context, input SearchInput, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{Content: "ok"}, nil
		},
	)

	result, err := tool.Execute(context.Background(), json.RawMessage(`{invalid`), nil)
	if err != nil {
		t.Fatalf("should not return error, got: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError for invalid JSON")
	}
	if !strings.Contains(result.Content, "invalid input") {
		t.Errorf("error message: %s", result.Content)
	}
}

type MathInput struct {
	A  float64 `json:"a" description:"First number"`
	B  float64 `json:"b" description:"Second number"`
	Op string  `json:"op" description:"Operator" enum:"+,-,*,/"`
}

func TestNewTyped_EnumTag(t *testing.T) {
	tool := tools.NewTyped[MathInput](
		"calc", "Calculator",
		[]string{"a", "b", "op"},
		func(_ context.Context, input MathInput, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			return &agentflow.ToolResult{Content: "42"}, nil
		},
	)

	schema := tool.InputSchema()
	props := schema["properties"].(map[string]any)
	opProp := props["op"].(map[string]any)

	enumVals, ok := opProp["enum"].([]any)
	if !ok {
		t.Fatal("no enum in op property")
	}
	if len(enumVals) != 4 {
		t.Errorf("expected 4 enum values, got %d", len(enumVals))
	}
	t.Logf("Enum values: %v", enumVals)
}

func TestNewTyped_BoolField(t *testing.T) {
	type Input struct {
		Verbose bool `json:"verbose" description:"Enable verbose output"`
	}

	tool := tools.NewTyped[Input](
		"test", "Test",
		nil,
		func(_ context.Context, input Input, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
			if input.Verbose {
				return &agentflow.ToolResult{Content: "verbose"}, nil
			}
			return &agentflow.ToolResult{Content: "quiet"}, nil
		},
	)

	schema := tool.InputSchema()
	props := schema["properties"].(map[string]any)
	verboseProp := props["verbose"].(map[string]any)
	if verboseProp["type"] != "boolean" {
		t.Errorf("verbose type: %v", verboseProp["type"])
	}

	result, _ := tool.Execute(context.Background(), json.RawMessage(`{"verbose":true}`), nil)
	if result.Content != "verbose" {
		t.Errorf("expected verbose, got: %s", result.Content)
	}
}
