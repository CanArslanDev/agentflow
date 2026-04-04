// Package ollama provides an agentflow Provider implementation for Ollama-compatible
// APIs (https://github.com/ollama/ollama/blob/main/docs/api.md).
//
// Works with local Ollama instances and RunPod serverless endpoints that expose
// the Ollama API format. Unlike OpenAI-compatible providers, Ollama uses JSONL
// streaming (one JSON object per line) instead of SSE.
//
//	provider := ollama.New("http://localhost:11434", "llama3.1:8b")
//	agent := agentflow.NewAgent(provider, agentflow.WithTools(myTool))
//
// For RunPod endpoints where the pod ID is embedded in the URL:
//
//	provider := ollama.New("https://pod123-11434.proxy.runpod.net", "llama3.1:70b")
package ollama

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/CanArslanDev/agentflow"
)

const defaultTimeout = 5 * time.Minute

// Provider implements agentflow.Provider for Ollama-compatible APIs.
type Provider struct {
	model   string
	baseURL string
	client  *http.Client
}

// ProviderOption configures the Ollama provider.
type ProviderOption func(*Provider)

// New creates an Ollama provider.
//
//	provider := ollama.New("http://localhost:11434", "llama3.1:8b")
//	provider := ollama.New("https://pod123-11434.proxy.runpod.net", "llama3.1:70b")
func New(baseURL, model string, opts ...ProviderOption) *Provider {
	p := &Provider{
		model:   model,
		baseURL: baseURL,
		client: &http.Client{
			Timeout: defaultTimeout,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(c *http.Client) ProviderOption {
	return func(p *Provider) { p.client = c }
}

// CreateStream initiates a streaming chat request to the Ollama API.
func (p *Provider) CreateStream(ctx context.Context, req *agentflow.Request) (agentflow.Stream, error) {
	body := p.buildRequestBody(req)

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("agentflow/ollama: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/api/chat", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("agentflow/ollama: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	for k, v := range req.Metadata {
		httpReq.Header.Set(k, v)
	}

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, &agentflow.ProviderError{StatusCode: 0, Message: err.Error(), Retryable: true, Err: err}
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, parseErrorResponse(resp)
	}

	return newStream(resp), nil
}

// ModelID returns the configured model identifier.
func (p *Provider) ModelID() string { return p.model }

// HealthCheck pings the Ollama API to verify it is reachable.
// Implements agentflow.HealthChecker for use with the fallback provider.
func (p *Provider) HealthCheck(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.baseURL+"/api/tags", nil)
	if err != nil {
		return fmt.Errorf("agentflow/ollama: create health check request: %w", err)
	}
	resp, err := p.client.Do(req)
	if err != nil {
		return fmt.Errorf("agentflow/ollama: health check failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("agentflow/ollama: health check returned status %d", resp.StatusCode)
	}
	return nil
}

// --- Request types ---

type chatRequest struct {
	Model    string           `json:"model"`
	Messages []requestMessage `json:"messages"`
	Stream   bool             `json:"stream"`
	Tools    []requestTool    `json:"tools,omitempty"`
	Options  *requestOptions  `json:"options,omitempty"`
}

type requestMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	Images    []string         `json:"images,omitempty"`
	ToolCalls []requestToolCall `json:"tool_calls,omitempty"`
}

type requestToolCall struct {
	Function requestFunctionCall `json:"function"`
}

type requestFunctionCall struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

type requestTool struct {
	Type     string          `json:"type"`
	Function requestFunction `json:"function"`
}

type requestFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

type requestOptions struct {
	Temperature *float64 `json:"temperature,omitempty"`
	NumPredict  int      `json:"num_predict,omitempty"`
}

func (p *Provider) buildRequestBody(req *agentflow.Request) chatRequest {
	cr := chatRequest{
		Model:    p.model,
		Messages: p.convertMessages(req.Messages, req.SystemPrompt),
		Stream:   true,
	}

	if len(req.Tools) > 0 {
		cr.Tools = convertTools(req.Tools)
	}

	if req.Temperature != nil || req.MaxTokens > 0 {
		opts := &requestOptions{}
		if req.Temperature != nil {
			opts.Temperature = req.Temperature
		}
		if req.MaxTokens > 0 {
			opts.NumPredict = req.MaxTokens
		}
		cr.Options = opts
	}

	return cr
}

func (p *Provider) convertMessages(messages []agentflow.Message, systemPrompt string) []requestMessage {
	var result []requestMessage

	if systemPrompt != "" {
		result = append(result, requestMessage{Role: "system", Content: systemPrompt})
	}

	for _, msg := range messages {
		switch msg.Role {
		case agentflow.RoleSystem:
			result = append(result, requestMessage{Role: "system", Content: msg.TextContent()})
		case agentflow.RoleUser:
			result = append(result, p.convertUserMessage(msg)...)
		case agentflow.RoleAssistant:
			result = append(result, p.convertAssistantMessage(msg))
		}
	}

	return result
}

func (p *Provider) convertUserMessage(msg agentflow.Message) []requestMessage {
	// Tool results are sent as separate role="tool" messages.
	toolResults := msg.ToolResults()
	if len(toolResults) > 0 {
		msgs := make([]requestMessage, len(toolResults))
		for i, tr := range toolResults {
			msgs[i] = requestMessage{Role: "tool", Content: tr.Content}
		}
		return msgs
	}

	rm := requestMessage{Role: "user"}
	var images []string

	for _, b := range msg.Content {
		switch b.Type {
		case agentflow.ContentText:
			rm.Content += b.Text
		case agentflow.ContentImage:
			if b.Image != nil && b.Image.Data != "" {
				images = append(images, b.Image.Data)
			}
		case agentflow.ContentDocument:
			// Ollama does not support documents natively.
			// Text-based documents are included as text; binary documents are noted.
			if b.Document != nil && b.Document.Data != "" {
				if isTextMimeType(b.Document.MediaType) {
					decoded, err := base64.StdEncoding.DecodeString(b.Document.Data)
					if err == nil {
						rm.Content += fmt.Sprintf("\n[Document: %s]\n%s\n", b.Document.Filename, string(decoded))
					}
				} else {
					rm.Content += fmt.Sprintf("\n[Document: %s (binary, not displayable)]\n", b.Document.Filename)
				}
			}
		}
	}

	if len(images) > 0 {
		rm.Images = images
	}

	return []requestMessage{rm}
}

func (p *Provider) convertAssistantMessage(msg agentflow.Message) requestMessage {
	rm := requestMessage{Role: "assistant", Content: msg.TextContent()}

	toolCalls := msg.ToolCalls()
	if len(toolCalls) > 0 {
		rm.ToolCalls = make([]requestToolCall, len(toolCalls))
		for i, tc := range toolCalls {
			var args map[string]any
			json.Unmarshal(tc.Input, &args)
			rm.ToolCalls[i] = requestToolCall{
				Function: requestFunctionCall{
					Name:      tc.Name,
					Arguments: args,
				},
			}
		}
	}

	return rm
}

func convertTools(tools []agentflow.ToolDefinition) []requestTool {
	result := make([]requestTool, len(tools))
	for i, t := range tools {
		result[i] = requestTool{
			Type: "function",
			Function: requestFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			},
		}
	}
	return result
}

func isTextMimeType(mt string) bool {
	switch mt {
	case "text/plain", "text/markdown", "text/csv", "text/html",
		"application/json", "application/xml", "application/javascript":
		return true
	}
	return false
}

func parseErrorResponse(resp *http.Response) error {
	var body struct {
		Error string `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return &agentflow.ProviderError{
			StatusCode: resp.StatusCode,
			Message:    "failed to parse error",
			Retryable:  resp.StatusCode >= 500,
		}
	}
	return &agentflow.ProviderError{
		StatusCode: resp.StatusCode,
		Message:    body.Error,
		Retryable:  resp.StatusCode == 429 || resp.StatusCode >= 500,
	}
}
