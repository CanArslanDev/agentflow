// Package jsonschema provides lightweight JSON Schema validation for tool inputs.
// It supports the subset of JSON Schema used by agentflow tools: type checking,
// required fields, enum constraints, and nested object validation.
package jsonschema

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Validate checks that input conforms to the given JSON Schema.
// Returns nil if valid, or an error describing the first violation found.
// An empty or nil schema always passes validation.
func Validate(schema map[string]any, input json.RawMessage) error {
	if len(schema) == 0 {
		return nil
	}
	if len(input) == 0 || string(input) == "null" {
		input = []byte("{}")
	}

	var value any
	if err := json.Unmarshal(input, &value); err != nil {
		return fmt.Errorf("invalid JSON input: %v", err)
	}

	return validateValue(schema, value, "")
}

// validateValue recursively validates a value against a schema at the given path.
func validateValue(schema map[string]any, value any, path string) error {
	// Type check.
	if schemaType, ok := schema["type"].(string); ok {
		if err := checkType(schemaType, value, path); err != nil {
			return err
		}
	}

	// Enum check.
	if enumRaw, ok := schema["enum"]; ok {
		if err := checkEnum(enumRaw, value, path); err != nil {
			return err
		}
	}

	// Object-specific checks.
	if obj, ok := value.(map[string]any); ok {
		// Required fields.
		if required, ok := schema["required"]; ok {
			if err := checkRequired(required, obj, path); err != nil {
				return err
			}
		}

		// Property validation.
		if properties, ok := schema["properties"].(map[string]any); ok {
			for propName, propSchema := range properties {
				propVal, exists := obj[propName]
				if !exists {
					continue // Missing optional fields are fine.
				}
				ps, ok := propSchema.(map[string]any)
				if !ok {
					continue
				}
				childPath := propName
				if path != "" {
					childPath = path + "." + propName
				}
				if err := validateValue(ps, propVal, childPath); err != nil {
					return err
				}
			}
		}

		// additionalProperties check.
		if additionalProps, ok := schema["additionalProperties"]; ok {
			if err := checkAdditionalProperties(additionalProps, schema, obj, path); err != nil {
				return err
			}
		}
	}

	// Array items validation.
	if arr, ok := value.([]any); ok {
		if itemSchema, ok := schema["items"].(map[string]any); ok {
			for i, item := range arr {
				childPath := fmt.Sprintf("%s[%d]", path, i)
				if path == "" {
					childPath = fmt.Sprintf("[%d]", i)
				}
				if err := validateValue(itemSchema, item, childPath); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

// checkType verifies that value matches the expected JSON Schema type.
func checkType(schemaType string, value any, path string) error {
	prefix := fieldPrefix(path)

	switch schemaType {
	case "object":
		if _, ok := value.(map[string]any); !ok {
			return fmt.Errorf("%sexpected object, got %s", prefix, jsonType(value))
		}
	case "array":
		if _, ok := value.([]any); !ok {
			return fmt.Errorf("%sexpected array, got %s", prefix, jsonType(value))
		}
	case "string":
		if _, ok := value.(string); !ok {
			return fmt.Errorf("%sexpected string, got %s", prefix, jsonType(value))
		}
	case "integer":
		switch v := value.(type) {
		case float64:
			if v != float64(int64(v)) {
				return fmt.Errorf("%sexpected integer, got float", prefix)
			}
		default:
			return fmt.Errorf("%sexpected integer, got %s", prefix, jsonType(value))
		}
	case "number":
		if _, ok := value.(float64); !ok {
			return fmt.Errorf("%sexpected number, got %s", prefix, jsonType(value))
		}
	case "boolean":
		if _, ok := value.(bool); !ok {
			return fmt.Errorf("%sexpected boolean, got %s", prefix, jsonType(value))
		}
	}
	return nil
}

// checkRequired verifies all required fields are present in the object.
func checkRequired(required any, obj map[string]any, path string) error {
	reqList, ok := toStringSlice(required)
	if !ok {
		return nil
	}

	var missing []string
	for _, field := range reqList {
		if _, exists := obj[field]; !exists {
			missing = append(missing, field)
		}
	}

	if len(missing) > 0 {
		prefix := fieldPrefix(path)
		return fmt.Errorf("%smissing required field(s): %s", prefix, strings.Join(missing, ", "))
	}
	return nil
}

// checkEnum verifies the value is one of the allowed enum values.
func checkEnum(enumRaw any, value any, path string) error {
	enumSlice, ok := enumRaw.([]any)
	if !ok {
		// Try []string.
		if strSlice, ok := enumRaw.([]string); ok {
			enumSlice = make([]any, len(strSlice))
			for i, s := range strSlice {
				enumSlice[i] = s
			}
		} else {
			return nil
		}
	}

	for _, allowed := range enumSlice {
		if fmt.Sprintf("%v", value) == fmt.Sprintf("%v", allowed) {
			return nil
		}
	}

	// Build allowed values list for error message.
	allowed := make([]string, len(enumSlice))
	for i, v := range enumSlice {
		allowed[i] = fmt.Sprintf("%v", v)
	}
	prefix := fieldPrefix(path)
	return fmt.Errorf("%svalue %v is not one of allowed values: [%s]", prefix, value, strings.Join(allowed, ", "))
}

// checkAdditionalProperties validates extra properties in an object.
func checkAdditionalProperties(additionalProps any, schema map[string]any, obj map[string]any, path string) error {
	properties, _ := schema["properties"].(map[string]any)

	switch ap := additionalProps.(type) {
	case bool:
		if !ap {
			// No additional properties allowed.
			for key := range obj {
				if properties != nil {
					if _, defined := properties[key]; defined {
						continue
					}
				}
				prefix := fieldPrefix(path)
				return fmt.Errorf("%sunknown property %q", prefix, key)
			}
		}
	case map[string]any:
		// Validate each additional property against the schema.
		for key, val := range obj {
			if properties != nil {
				if _, defined := properties[key]; defined {
					continue
				}
			}
			childPath := key
			if path != "" {
				childPath = path + "." + key
			}
			if err := validateValue(ap, val, childPath); err != nil {
				return err
			}
		}
	}
	return nil
}

// jsonType returns a human-readable type name for a JSON value.
func jsonType(v any) string {
	switch v.(type) {
	case nil:
		return "null"
	case bool:
		return "boolean"
	case float64:
		return "number"
	case string:
		return "string"
	case []any:
		return "array"
	case map[string]any:
		return "object"
	default:
		return fmt.Sprintf("%T", v)
	}
}

// fieldPrefix returns "field_name: " or "" for the root.
func fieldPrefix(path string) string {
	if path == "" {
		return ""
	}
	return path + ": "
}

// toStringSlice converts []any or []string to []string.
func toStringSlice(v any) ([]string, bool) {
	switch s := v.(type) {
	case []string:
		return s, true
	case []any:
		result := make([]string, len(s))
		for i, item := range s {
			str, ok := item.(string)
			if !ok {
				return nil, false
			}
			result[i] = str
		}
		return result, true
	}
	return nil, false
}
