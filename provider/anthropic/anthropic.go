// Package anthropic provides an agentflow Provider implementation for the
// Anthropic Messages API (https://docs.anthropic.com/en/api/messages).
//
// Anthropic uses a different format than OpenAI — tool calls are content blocks
// (type: "tool_use") within the assistant message, not a separate tool_calls array.
//
//	provider := anthropic.New("sk-ant-...", "claude-sonnet-4-20250514")
//	agent := agentflow.NewAgent(provider, agentflow.WithTools(myTool))
package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/CanArslanDev/agentflow"
)

const (
	defaultBaseURL      = "https://api.anthropic.com/v1"
	defaultTimeout      = 5 * time.Minute
	defaultAPIVersion   = "2023-06-01"
	defaultMaxTokens    = 4096
)

// Provider implements agentflow.Provider for the Anthropic Messages API.
type Provider struct {
	apiKey     string
	model      string
	baseURL    string
	apiVersion string
	client     *http.Client
}

// ProviderOption configures the Anthropic provider.
type ProviderOption func(*Provider)

// New creates an Anthropic provider targeting the specified model.
//
//	provider := anthropic.New("sk-ant-...", "claude-sonnet-4-20250514")
//	provider := anthropic.New("sk-ant-...", "claude-haiku-4-5-20251001")
//	provider := anthropic.New("sk-ant-...", "claude-opus-4-20250514")
func New(apiKey, model string, opts ...ProviderOption) *Provider {
	p := &Provider{
		apiKey:     apiKey,
		model:      model,
		baseURL:    defaultBaseURL,
		apiVersion: defaultAPIVersion,
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

// WithBaseURL overrides the API base URL.
func WithBaseURL(url string) ProviderOption {
	return func(p *Provider) { p.baseURL = url }
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(c *http.Client) ProviderOption {
	return func(p *Provider) { p.client = c }
}

// WithAPIVersion overrides the anthropic-version header.
func WithAPIVersion(version string) ProviderOption {
	return func(p *Provider) { p.apiVersion = version }
}

// CreateStream initiates a streaming messages request.
func (p *Provider) CreateStream(ctx context.Context, req *agentflow.Request) (agentflow.Stream, error) {
	body := p.buildRequestBody(req)

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("agentflow/anthropic: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/messages", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("agentflow/anthropic: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", p.apiKey)
	httpReq.Header.Set("anthropic-version", p.apiVersion)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, &agentflow.ProviderError{StatusCode: 0, Message: err.Error(), Retryable: true, Err: err}
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, p.parseErrorResponse(resp)
	}

	return newAnthropicStream(resp), nil
}

// ModelID returns the configured model identifier.
func (p *Provider) ModelID() string { return p.model }

// buildRequestBody converts an agentflow.Request into Anthropic Messages API format.
func (p *Provider) buildRequestBody(req *agentflow.Request) messagesRequest {
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}

	mr := messagesRequest{
		Model:     p.model,
		System:    req.SystemPrompt,
		MaxTokens: maxTokens,
		Stream:    true,
		Messages:  p.convertMessages(req.Messages),
		Tools:     p.convertTools(req.Tools),
	}

	return mr
}

// convertMessages transforms agentflow messages into Anthropic format.
func (p *Provider) convertMessages(messages []agentflow.Message) []requestMessage {
	var result []requestMessage

	for _, msg := range messages {
		switch msg.Role {
		case agentflow.RoleUser:
			result = append(result, p.convertUserMessage(msg))
		case agentflow.RoleAssistant:
			result = append(result, p.convertAssistantMessage(msg))
		case agentflow.RoleSystem:
			// System messages are handled via the top-level "system" field.
			// If there are inline system messages, convert to user messages.
			if text := msg.TextContent(); text != "" {
				result = append(result, requestMessage{
					Role: "user",
					Content: []contentBlock{{Type: "text", Text: "[System] " + text}},
				})
			}
		}
	}

	return result
}

// convertUserMessage handles user messages with text, images, and tool results.
func (p *Provider) convertUserMessage(msg agentflow.Message) requestMessage {
	var blocks []contentBlock

	for _, b := range msg.Content {
		switch b.Type {
		case agentflow.ContentText:
			if b.Text != "" {
				blocks = append(blocks, contentBlock{Type: "text", Text: b.Text})
			}
		case agentflow.ContentImage:
			if b.Image != nil {
				if b.Image.Data != "" {
					mediaType := b.Image.MediaType
					if mediaType == "" {
						mediaType = "image/png"
					}
					blocks = append(blocks, contentBlock{
						Type: "image",
						Source: &imageSource{
							Type:      "base64",
							MediaType: mediaType,
							Data:      b.Image.Data,
						},
					})
				}
			}
		case agentflow.ContentToolResult:
			if b.ToolResult != nil {
				blocks = append(blocks, contentBlock{
					Type:      "tool_result",
					ToolUseID: b.ToolResult.ToolCallID,
					Content:   b.ToolResult.Content,
					IsError:   b.ToolResult.IsError,
				})
			}
		}
	}

	if len(blocks) == 0 {
		blocks = []contentBlock{{Type: "text", Text: ""}}
	}

	return requestMessage{Role: "user", Content: blocks}
}

// convertAssistantMessage handles assistant messages with text and tool calls.
func (p *Provider) convertAssistantMessage(msg agentflow.Message) requestMessage {
	var blocks []contentBlock

	for _, b := range msg.Content {
		switch b.Type {
		case agentflow.ContentText:
			if b.Text != "" {
				blocks = append(blocks, contentBlock{Type: "text", Text: b.Text})
			}
		case agentflow.ContentToolCall:
			if b.ToolCall != nil {
				var input any
				json.Unmarshal(b.ToolCall.Input, &input)
				blocks = append(blocks, contentBlock{
					Type:  "tool_use",
					ID:    b.ToolCall.ID,
					Name:  b.ToolCall.Name,
					Input: input,
				})
			}
		}
	}

	if len(blocks) == 0 {
		blocks = []contentBlock{{Type: "text", Text: ""}}
	}

	return requestMessage{Role: "assistant", Content: blocks}
}

// convertTools transforms agentflow tool definitions into Anthropic format.
func (p *Provider) convertTools(tools []agentflow.ToolDefinition) []requestTool {
	if len(tools) == 0 {
		return nil
	}

	result := make([]requestTool, len(tools))
	for i, t := range tools {
		schema := toolSchema{Type: "object"}
		if props, ok := t.InputSchema["properties"]; ok {
			if propsMap, ok := props.(map[string]any); ok {
				schema.Properties = propsMap
			}
		}
		if req, ok := t.InputSchema["required"]; ok {
			if reqArr, ok := req.([]string); ok {
				schema.Required = reqArr
			}
			// Handle []any from JSON unmarshal.
			if reqArr, ok := req.([]any); ok {
				for _, r := range reqArr {
					if s, ok := r.(string); ok {
						schema.Required = append(schema.Required, s)
					}
				}
			}
		}

		result[i] = requestTool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: schema,
		}
	}
	return result
}

// parseErrorResponse reads an error response from the Anthropic API.
func (p *Provider) parseErrorResponse(resp *http.Response) error {
	var body struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return &agentflow.ProviderError{StatusCode: resp.StatusCode, Message: "failed to parse error", Retryable: resp.StatusCode >= 500}
	}
	return &agentflow.ProviderError{
		StatusCode: resp.StatusCode,
		Message:    body.Error.Message,
		Retryable:  resp.StatusCode == 429 || resp.StatusCode >= 500 || resp.StatusCode == 529,
	}
}
