// Package groq provides an agentflow Provider implementation for the Groq API.
package groq

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/internal/sse"
)

const (
	defaultBaseURL = "https://api.groq.com/openai/v1"
	defaultTimeout = 5 * time.Minute
)

// Provider implements agentflow.Provider for the Groq API.
type Provider struct {
	apiKey  string
	model   string
	baseURL string
	client  *http.Client
}

// ProviderOption configures the Groq provider.
type ProviderOption func(*Provider)

// New creates a Groq provider targeting the specified model.
func New(apiKey, model string, opts ...ProviderOption) *Provider {
	p := &Provider{
		apiKey:  apiKey,
		model:   model,
		baseURL: defaultBaseURL,
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

// groqChatRequest wraps the shared ChatRequest to use Groq's preferred
// max_completion_tokens field instead of max_tokens.
type groqChatRequest struct {
	sse.ChatRequest
	MaxCompletionTokens int `json:"max_completion_tokens,omitempty"`
}

// CreateStream initiates a streaming chat completion request to Groq.
func (p *Provider) CreateStream(ctx context.Context, req *agentflow.Request) (agentflow.Stream, error) {
	body := sse.BuildRequestBody(p.model, req)

	// Groq API uses max_completion_tokens instead of max_tokens.
	groqBody := groqChatRequest{ChatRequest: body}
	if body.MaxTokens > 0 {
		groqBody.MaxCompletionTokens = body.MaxTokens
		groqBody.ChatRequest.MaxTokens = 0
	}

	jsonBody, err := json.Marshal(groqBody)
	if err != nil {
		return nil, fmt.Errorf("agentflow/groq: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("agentflow/groq: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
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

	return sse.NewStream(resp), nil
}

// ModelID returns the configured model identifier.
func (p *Provider) ModelID() string { return p.model }

func parseErrorResponse(resp *http.Response) error {
	var body struct {
		Error struct{ Message string `json:"message"` } `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return &agentflow.ProviderError{StatusCode: resp.StatusCode, Message: "failed to parse error", Retryable: resp.StatusCode >= 500}
	}
	return &agentflow.ProviderError{
		StatusCode: resp.StatusCode,
		Message:    body.Error.Message,
		Retryable:  resp.StatusCode == 429 || resp.StatusCode >= 500,
	}
}
