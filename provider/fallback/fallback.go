// Package fallback provides a Provider that cascades through multiple providers.
// If the primary provider fails with a retryable error, the next provider in the
// chain is tried. This gives resilience against rate limits, outages, and transient
// failures without any changes to the agent or tool code.
//
//	provider := fallback.New(
//	    openrouter.New(key, "anthropic/claude-sonnet-4-20250514"),
//	    groq.New(groqKey, "llama-3.3-70b-versatile"),
//	)
//	agent := agentflow.NewAgent(provider, ...)
package fallback

import (
	"context"
	"errors"
	"fmt"

	"github.com/CanArslanDev/agentflow"
)

// Provider tries each underlying provider in order until one succeeds.
// Only retryable errors trigger a fallback; non-retryable errors (auth failures,
// invalid requests) are returned immediately.
type Provider struct {
	providers []agentflow.Provider
	onFallback func(from, to agentflow.Provider, err error)
}

// Option configures the fallback provider.
type Option func(*Provider)

// New creates a fallback provider from the given providers. At least one provider
// is required. The first provider is the primary; subsequent ones are fallbacks
// tried in order.
func New(primary agentflow.Provider, fallbacks ...agentflow.Provider) *Provider {
	providers := make([]agentflow.Provider, 0, 1+len(fallbacks))
	providers = append(providers, primary)
	providers = append(providers, fallbacks...)
	return &Provider{providers: providers}
}

// WithOnFallback sets a callback invoked when a fallback is triggered.
// Useful for logging or metrics.
func WithOnFallback(fn func(from, to agentflow.Provider, err error)) Option {
	return func(p *Provider) {
		p.onFallback = fn
	}
}

// CreateStream tries each provider in order. The first successful stream is
// returned. If all providers fail, the last error is returned.
func (p *Provider) CreateStream(ctx context.Context, req *agentflow.Request) (agentflow.Stream, error) {
	var lastErr error

	for i, provider := range p.providers {
		stream, err := provider.CreateStream(ctx, req)
		if err == nil {
			return stream, nil
		}

		lastErr = err

		// Non-retryable errors stop the cascade immediately.
		if !isRetryable(err) {
			return nil, err
		}

		// Notify callback if set.
		if p.onFallback != nil && i+1 < len(p.providers) {
			p.onFallback(provider, p.providers[i+1], err)
		}

		// Check context before trying next provider.
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
	}

	return nil, fmt.Errorf("agentflow/fallback: all %d providers failed: %w", len(p.providers), lastErr)
}

// ModelID returns the primary provider's model ID.
func (p *Provider) ModelID() string {
	if len(p.providers) > 0 {
		return p.providers[0].ModelID()
	}
	return "fallback"
}

// Providers returns the list of underlying providers for inspection.
func (p *Provider) Providers() []agentflow.Provider {
	return p.providers
}

// isRetryable checks if an error should trigger a fallback to the next provider.
func isRetryable(err error) bool {
	var pe *agentflow.ProviderError
	if errors.As(err, &pe) {
		return pe.IsRetryable()
	}
	// Network errors, timeouts, etc. are retryable.
	return true
}
