package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/CanArslanDev/agentflow"
)

// TypedExecuteFunc is the execution function for a TypedTool with concrete input/output types.
type TypedExecuteFunc[I any] func(ctx context.Context, input I, progress agentflow.ProgressFunc) (*agentflow.ToolResult, error)

// TypedTool creates a type-safe tool that automatically:
//   - Generates JSON Schema from the input struct's fields and json tags
//   - Unmarshals json.RawMessage into the concrete input type
//   - Returns clear validation errors if unmarshalling fails
//
// The input type I must be a struct with json tags. Fields tagged with
// `json:",required"` or listed in the required parameter are marked required.
//
//	type SearchInput struct {
//	    Query      string `json:"query" description:"Search query"`
//	    MaxResults int    `json:"max_results,omitempty" description:"Max results to return"`
//	}
//
//	tool := tools.NewTyped[SearchInput](
//	    "web_search", "Search the web",
//	    []string{"query"},
//	    func(ctx context.Context, input SearchInput, p agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
//	        // input.Query is already typed
//	        return &agentflow.ToolResult{Content: "results for: " + input.Query}, nil
//	    },
//	)
func NewTyped[I any](name, description string, required []string, fn TypedExecuteFunc[I]) agentflow.Tool {
	schema := schemaFromType[I](required)
	return &typedTool[I]{
		name:        name,
		description: description,
		schema:      schema,
		executeFn:   fn,
	}
}

type typedTool[I any] struct {
	name        string
	description string
	schema      map[string]any
	executeFn   TypedExecuteFunc[I]
}

func (t *typedTool[I]) Name() string               { return t.name }
func (t *typedTool[I]) Description() string         { return t.description }
func (t *typedTool[I]) InputSchema() map[string]any { return t.schema }
func (t *typedTool[I]) IsConcurrencySafe(_ json.RawMessage) bool { return true }
func (t *typedTool[I]) IsReadOnly(_ json.RawMessage) bool       { return true }
func (t *typedTool[I]) Locality() agentflow.ToolLocality         { return agentflow.ToolAny }

func (t *typedTool[I]) Execute(ctx context.Context, input json.RawMessage, progress agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var typed I
	if err := json.Unmarshal(input, &typed); err != nil {
		return &agentflow.ToolResult{
			Content: fmt.Sprintf("invalid input for %s: %v", t.name, err),
			IsError: true,
		}, nil
	}
	return t.executeFn(ctx, typed, progress)
}

// schemaFromType generates a JSON Schema object from a struct type using
// reflection. It reads json tags for field names and "description" tags
// for field descriptions.
func schemaFromType[I any](required []string) map[string]any {
	var zero I
	t := reflect.TypeOf(zero)

	// Handle pointer types.
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	if t.Kind() != reflect.Struct {
		return map[string]any{"type": "object"}
	}

	properties := make(map[string]any)

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if !field.IsExported() {
			continue
		}

		jsonTag := field.Tag.Get("json")
		if jsonTag == "-" {
			continue
		}

		name := field.Name
		if jsonTag != "" {
			parts := strings.Split(jsonTag, ",")
			if parts[0] != "" {
				name = parts[0]
			}
		}

		prop := map[string]any{
			"type": goTypeToJSONSchemaType(field.Type),
		}

		if desc := field.Tag.Get("description"); desc != "" {
			prop["description"] = desc
		}

		if enumTag := field.Tag.Get("enum"); enumTag != "" {
			values := strings.Split(enumTag, ",")
			enumAny := make([]any, len(values))
			for j, v := range values {
				enumAny[j] = v
			}
			prop["enum"] = enumAny
		}

		properties[name] = prop
	}

	schema := map[string]any{
		"type":       "object",
		"properties": properties,
	}

	if len(required) > 0 {
		schema["required"] = required
	}

	return schema
}

// goTypeToJSONSchemaType maps Go types to JSON Schema type strings.
func goTypeToJSONSchemaType(t reflect.Type) string {
	switch t.Kind() {
	case reflect.String:
		return "string"
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return "integer"
	case reflect.Float32, reflect.Float64:
		return "number"
	case reflect.Bool:
		return "boolean"
	case reflect.Slice, reflect.Array:
		return "array"
	case reflect.Map, reflect.Struct:
		return "object"
	default:
		return "string"
	}
}
