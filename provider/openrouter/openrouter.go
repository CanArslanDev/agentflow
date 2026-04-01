// Package openrouter provides an agentflow Provider implementation for the
// OpenRouter API (https://openrouter.ai).
package openrouter

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
	defaultBaseURL = "https://openrouter.ai/api/v1"
	defaultTimeout = 5 * time.Minute
)

// Provider implements agentflow.Provider for the OpenRouter API.
type Provider struct {
	apiKey  string
	model   string
	baseURL string
	client  *http.Client
	referer string
	title   string
}

// ProviderOption configures the OpenRouter provider.
type ProviderOption func(*Provider)

// New creates an OpenRouter provider targeting the specified model.
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

// WithReferer sets the HTTP-Referer header.
func WithReferer(referer string) ProviderOption {
	return func(p *Provider) { p.referer = referer }
}

// WithTitle sets the X-Title header.
func WithTitle(title string) ProviderOption {
	return func(p *Provider) { p.title = title }
}

// CreateStream initiates a streaming chat completion request.
func (p *Provider) CreateStream(ctx context.Context, req *agentflow.Request) (agentflow.Stream, error) {
	body := sse.BuildRequestBody(p.model, req)

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("agentflow/openrouter: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("agentflow/openrouter: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	if p.referer != "" {
		httpReq.Header.Set("HTTP-Referer", p.referer)
	}
	if p.title != "" {
		httpReq.Header.Set("X-Title", p.title)
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
