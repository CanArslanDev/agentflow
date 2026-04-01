package builtin

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/canarslan/agentflow"
)

// HTTPRequest returns a tool that makes HTTP requests.
func HTTPRequest() agentflow.Tool { return &httpTool{} }

type httpTool struct{}

func (t *httpTool) Name() string { return "http_request" }
func (t *httpTool) Description() string {
	return "Make an HTTP request to a URL. Returns the response status, headers, and body. Useful for calling APIs, fetching web pages, or checking endpoints."
}
func (t *httpTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"url":    map[string]any{"type": "string", "description": "The URL to request"},
			"method": map[string]any{"type": "string", "description": "HTTP method (default: GET)", "enum": []string{"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}},
			"body":   map[string]any{"type": "string", "description": "Request body (for POST/PUT/PATCH)"},
			"headers": map[string]any{
				"type":        "object",
				"description": "Request headers as key-value pairs",
				"additionalProperties": map[string]any{"type": "string"},
			},
		},
		"required": []string{"url"},
	}
}

func (t *httpTool) Execute(ctx context.Context, input json.RawMessage, progress agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var params struct {
		URL     string            `json:"url"`
		Method  string            `json:"method"`
		Body    string            `json:"body"`
		Headers map[string]string `json:"headers"`
	}
	if err := json.Unmarshal(input, &params); err != nil {
		return &agentflow.ToolResult{Content: "invalid input: " + err.Error(), IsError: true}, nil
	}

	method := params.Method
	if method == "" {
		method = "GET"
	}

	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	if progress != nil {
		progress(agentflow.ProgressEvent{Message: fmt.Sprintf("%s %s", method, params.URL)})
	}

	var bodyReader io.Reader
	if params.Body != "" {
		bodyReader = strings.NewReader(params.Body)
	}

	req, err := http.NewRequestWithContext(ctx, method, params.URL, bodyReader)
	if err != nil {
		return &agentflow.ToolResult{Content: err.Error(), IsError: true}, nil
	}

	for k, v := range params.Headers {
		req.Header.Set(k, v)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return &agentflow.ToolResult{Content: err.Error(), IsError: true}, nil
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 100*1024)) // 100KB limit
	if err != nil {
		return &agentflow.ToolResult{Content: err.Error(), IsError: true}, nil
	}

	result := fmt.Sprintf("Status: %d %s\n\n%s", resp.StatusCode, resp.Status, string(body))

	isError := resp.StatusCode >= 400
	return &agentflow.ToolResult{
		Content: result,
		IsError: isError,
		Metadata: map[string]any{
			"status_code":    resp.StatusCode,
			"content_length": len(body),
		},
	}, nil
}

func (t *httpTool) IsConcurrencySafe(_ json.RawMessage) bool { return true }
func (t *httpTool) IsReadOnly(_ json.RawMessage) bool        { return true }
func (t *httpTool) Locality() agentflow.ToolLocality          { return agentflow.ToolRemoteSafe }
