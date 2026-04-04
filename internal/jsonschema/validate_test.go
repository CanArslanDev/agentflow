package jsonschema

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestValidate_NilSchema(t *testing.T) {
	if err := Validate(nil, json.RawMessage(`{"any":"thing"}`)); err != nil {
		t.Errorf("nil schema should pass: %v", err)
	}
}

func TestValidate_EmptySchema(t *testing.T) {
	if err := Validate(map[string]any{}, json.RawMessage(`{"any":"thing"}`)); err != nil {
		t.Errorf("empty schema should pass: %v", err)
	}
}

func TestValidate_EmptyInput(t *testing.T) {
	schema := map[string]any{"type": "object"}
	if err := Validate(schema, nil); err != nil {
		t.Errorf("nil input should default to empty object: %v", err)
	}
	if err := Validate(schema, json.RawMessage(``)); err != nil {
		t.Errorf("empty input should default to empty object: %v", err)
	}
}

func TestValidate_InvalidJSON(t *testing.T) {
	schema := map[string]any{"type": "object"}
	err := Validate(schema, json.RawMessage(`{broken`))
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
	if !strings.Contains(err.Error(), "invalid JSON") {
		t.Errorf("expected 'invalid JSON' in error, got: %v", err)
	}
}

func TestValidate_TypeObject(t *testing.T) {
	schema := map[string]any{"type": "object"}

	if err := Validate(schema, json.RawMessage(`{"key":"val"}`)); err != nil {
		t.Errorf("valid object: %v", err)
	}
	if err := Validate(schema, json.RawMessage(`"string"`)); err == nil {
		t.Error("string should fail object type check")
	}
}

func TestValidate_TypeString(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
		},
	}

	if err := Validate(schema, json.RawMessage(`{"name":"Go"}`)); err != nil {
		t.Errorf("valid string: %v", err)
	}
	err := Validate(schema, json.RawMessage(`{"name":123}`))
	if err == nil {
		t.Fatal("number should fail string type check")
	}
	if !strings.Contains(err.Error(), "expected string") {
		t.Errorf("expected 'expected string', got: %v", err)
	}
}

func TestValidate_TypeInteger(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"count": map[string]any{"type": "integer"},
		},
	}

	if err := Validate(schema, json.RawMessage(`{"count":42}`)); err != nil {
		t.Errorf("valid integer: %v", err)
	}
	err := Validate(schema, json.RawMessage(`{"count":3.14}`))
	if err == nil {
		t.Fatal("float should fail integer type check")
	}
	if !strings.Contains(err.Error(), "expected integer") {
		t.Errorf("expected 'expected integer', got: %v", err)
	}
}

func TestValidate_TypeNumber(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"score": map[string]any{"type": "number"},
		},
	}

	if err := Validate(schema, json.RawMessage(`{"score":3.14}`)); err != nil {
		t.Errorf("valid number: %v", err)
	}
	if err := Validate(schema, json.RawMessage(`{"score":42}`)); err != nil {
		t.Errorf("integer is also valid number: %v", err)
	}
	err := Validate(schema, json.RawMessage(`{"score":"high"}`))
	if err == nil {
		t.Fatal("string should fail number type check")
	}
}

func TestValidate_TypeBoolean(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"active": map[string]any{"type": "boolean"},
		},
	}

	if err := Validate(schema, json.RawMessage(`{"active":true}`)); err != nil {
		t.Errorf("valid boolean: %v", err)
	}
	err := Validate(schema, json.RawMessage(`{"active":"yes"}`))
	if err == nil {
		t.Fatal("string should fail boolean type check")
	}
}

func TestValidate_TypeArray(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"items": map[string]any{"type": "array"},
		},
	}

	if err := Validate(schema, json.RawMessage(`{"items":[1,2,3]}`)); err != nil {
		t.Errorf("valid array: %v", err)
	}
	err := Validate(schema, json.RawMessage(`{"items":"not-array"}`))
	if err == nil {
		t.Fatal("string should fail array type check")
	}
}

func TestValidate_Required(t *testing.T) {
	schema := map[string]any{
		"type":     "object",
		"required": []string{"command"},
		"properties": map[string]any{
			"command": map[string]any{"type": "string"},
			"timeout": map[string]any{"type": "integer"},
		},
	}

	if err := Validate(schema, json.RawMessage(`{"command":"ls"}`)); err != nil {
		t.Errorf("valid with required: %v", err)
	}
	if err := Validate(schema, json.RawMessage(`{"command":"ls","timeout":30}`)); err != nil {
		t.Errorf("valid with optional: %v", err)
	}

	err := Validate(schema, json.RawMessage(`{"timeout":30}`))
	if err == nil {
		t.Fatal("missing required field should fail")
	}
	if !strings.Contains(err.Error(), "command") {
		t.Errorf("error should mention missing field 'command': %v", err)
	}
}

func TestValidate_RequiredMultiple(t *testing.T) {
	schema := map[string]any{
		"type":     "object",
		"required": []string{"a", "b"},
	}

	err := Validate(schema, json.RawMessage(`{}`))
	if err == nil {
		t.Fatal("missing both required fields should fail")
	}
	if !strings.Contains(err.Error(), "a") || !strings.Contains(err.Error(), "b") {
		t.Errorf("error should mention both missing fields: %v", err)
	}
}

func TestValidate_Enum(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"method": map[string]any{
				"type": "string",
				"enum": []any{"GET", "POST", "PUT", "DELETE"},
			},
		},
	}

	if err := Validate(schema, json.RawMessage(`{"method":"GET"}`)); err != nil {
		t.Errorf("valid enum: %v", err)
	}

	err := Validate(schema, json.RawMessage(`{"method":"PATCH"}`))
	if err == nil {
		t.Fatal("invalid enum value should fail")
	}
	if !strings.Contains(err.Error(), "PATCH") && !strings.Contains(err.Error(), "not one of") {
		t.Errorf("error should describe invalid enum value: %v", err)
	}
}

func TestValidate_EnumStringSlice(t *testing.T) {
	// Some tools use []string instead of []any for enum.
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"status": map[string]any{
				"type": "string",
				"enum": []string{"pending", "completed"},
			},
		},
	}

	if err := Validate(schema, json.RawMessage(`{"status":"pending"}`)); err != nil {
		t.Errorf("valid enum with []string: %v", err)
	}
	err := Validate(schema, json.RawMessage(`{"status":"unknown"}`))
	if err == nil {
		t.Fatal("invalid enum should fail")
	}
}

func TestValidate_NestedObject(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"config": map[string]any{
				"type":     "object",
				"required": []string{"name"},
				"properties": map[string]any{
					"name":  map[string]any{"type": "string"},
					"count": map[string]any{"type": "integer"},
				},
			},
		},
	}

	if err := Validate(schema, json.RawMessage(`{"config":{"name":"test","count":5}}`)); err != nil {
		t.Errorf("valid nested: %v", err)
	}

	err := Validate(schema, json.RawMessage(`{"config":{"count":5}}`))
	if err == nil {
		t.Fatal("missing nested required should fail")
	}
	if !strings.Contains(err.Error(), "config") && !strings.Contains(err.Error(), "name") {
		t.Errorf("error should reference nested path: %v", err)
	}
}

func TestValidate_AdditionalPropertiesFalse(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
		},
		"additionalProperties": false,
	}

	if err := Validate(schema, json.RawMessage(`{"name":"Go"}`)); err != nil {
		t.Errorf("valid without extra props: %v", err)
	}

	err := Validate(schema, json.RawMessage(`{"name":"Go","extra":"field"}`))
	if err == nil {
		t.Fatal("additional property should fail when additionalProperties=false")
	}
	if !strings.Contains(err.Error(), "extra") {
		t.Errorf("error should mention unknown property: %v", err)
	}
}

func TestValidate_AdditionalPropertiesSchema(t *testing.T) {
	schema := map[string]any{
		"type":                 "object",
		"additionalProperties": map[string]any{"type": "string"},
	}

	if err := Validate(schema, json.RawMessage(`{"a":"hello","b":"world"}`)); err != nil {
		t.Errorf("valid additional string props: %v", err)
	}

	err := Validate(schema, json.RawMessage(`{"a":"hello","b":42}`))
	if err == nil {
		t.Fatal("non-string additional property should fail")
	}
}

func TestValidate_ArrayItems(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"tags": map[string]any{
				"type":  "array",
				"items": map[string]any{"type": "string"},
			},
		},
	}

	if err := Validate(schema, json.RawMessage(`{"tags":["go","rust"]}`)); err != nil {
		t.Errorf("valid array items: %v", err)
	}

	err := Validate(schema, json.RawMessage(`{"tags":["go",42]}`))
	if err == nil {
		t.Fatal("non-string array item should fail")
	}
	if !strings.Contains(err.Error(), "[1]") {
		t.Errorf("error should reference array index: %v", err)
	}
}

func TestValidate_OptionalFieldSkipped(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name":    map[string]any{"type": "string"},
			"timeout": map[string]any{"type": "integer"},
		},
		"required": []string{"name"},
	}

	// Omitting optional "timeout" is fine.
	if err := Validate(schema, json.RawMessage(`{"name":"test"}`)); err != nil {
		t.Errorf("optional field omitted should pass: %v", err)
	}
}

func TestValidate_RealToolSchema_Bash(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"command": map[string]any{"type": "string", "description": "Shell command"},
			"timeout": map[string]any{"type": "integer", "description": "Timeout in seconds"},
		},
		"required": []string{"command"},
	}

	if err := Validate(schema, json.RawMessage(`{"command":"ls -la"}`)); err != nil {
		t.Errorf("bash valid: %v", err)
	}
	if err := Validate(schema, json.RawMessage(`{"command":"ls","timeout":30}`)); err != nil {
		t.Errorf("bash with timeout: %v", err)
	}
	err := Validate(schema, json.RawMessage(`{"timeout":30}`))
	if err == nil {
		t.Fatal("bash without command should fail")
	}
}

func TestValidate_RealToolSchema_HTTP(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"url": map[string]any{"type": "string"},
			"method": map[string]any{
				"type": "string",
				"enum": []any{"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"},
			},
			"headers": map[string]any{
				"type":                 "object",
				"additionalProperties": map[string]any{"type": "string"},
			},
		},
		"required": []string{"url"},
	}

	if err := Validate(schema, json.RawMessage(`{"url":"https://example.com","method":"GET"}`)); err != nil {
		t.Errorf("http valid: %v", err)
	}
	if err := Validate(schema, json.RawMessage(`{"url":"https://example.com","headers":{"Content-Type":"application/json"}}`)); err != nil {
		t.Errorf("http with headers: %v", err)
	}

	err := Validate(schema, json.RawMessage(`{"url":"https://example.com","method":"INVALID"}`))
	if err == nil {
		t.Fatal("invalid HTTP method should fail")
	}
}
