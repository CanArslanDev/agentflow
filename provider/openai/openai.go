// Package openai provides an agentflow Provider implementation for the OpenAI
// Chat Completions API (https://platform.openai.com/docs/api-reference/chat).
//
//	provider := openai.New("sk-...", "gpt-4o")
//	agent := agentflow.NewAgent(provider, agentflow.WithTools(myTool))
package openai

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
	defaultBaseURL = "https://api.openai.com/v1"
	defaultTimeout = 5 * time.Minute
)

// Provider implements agentflow.Provider for the OpenAI API.
type Provider struct {
	apiKey  string
	model   string
	baseURL string
	client  *http.Client
	orgID   string
}

// ProviderOption configures the OpenAI provider.
type ProviderOption func(*Provider)

// New creates an OpenAI provider targeting the specified model.
//
//	provider := openai.New("sk-...", "gpt-4o")
//	provider := openai.New("sk-...", "gpt-4o-mini")
//	provider := openai.New("sk-...", "gpt-4-turbo")
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

// WithBaseURL overrides the API base URL (useful for Azure OpenAI or proxies).
func WithBaseURL(url string) ProviderOption {
	return func(p *Provider) { p.baseURL = url }
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(c *http.Client) ProviderOption {
	return func(p *Provider) { p.client = c }
}

// WithOrganization sets the OpenAI-Organization header.
func WithOrganization(orgID string) ProviderOption {
	return func(p *Provider) { p.orgID = orgID }
}

// CreateStream initiates a streaming chat completion request.
func (p *Provider) CreateStream(ctx context.Context, req *agentflow.Request) (agentflow.Stream, error) {
	body := sse.BuildRequestBody(p.model, req)

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("agentflow/openai: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("agentflow/openai: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	if p.orgID != "" {
		httpReq.Header.Set("OpenAI-Organization", p.orgID)
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
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
		} `json:"error"`
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
