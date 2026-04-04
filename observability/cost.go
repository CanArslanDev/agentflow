package observability

import (
	"sync"
	"time"

	"github.com/CanArslanDev/agentflow"
)

// CostTracker estimates API costs based on token usage and model pricing.
//
//	tracker := observability.NewCostTracker()
//	tracker.SetPrice("llama-3.3-70b-versatile", 0.59, 0.79) // per 1M tokens
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithOnEvent(tracker.OnEvent),
//	)
//	// after run:
//	fmt.Printf("Total cost: $%.4f\n", tracker.TotalCost())
type CostTracker struct {
	mu     sync.Mutex
	prices map[string]ModelPrice
	usage  []UsageRecord
}

// ModelPrice defines the cost per million tokens for a model.
type ModelPrice struct {
	PromptPerMillion     float64 `json:"prompt_per_million"`
	CompletionPerMillion float64 `json:"completion_per_million"`
}

// UsageRecord captures a single API call's token usage.
type UsageRecord struct {
	Timestamp        time.Time `json:"timestamp"`
	PromptTokens     int       `json:"prompt_tokens"`
	CompletionTokens int       `json:"completion_tokens"`
	EstimatedCost    float64   `json:"estimated_cost"`
}

// NewCostTracker creates a tracker with default pricing for common models.
func NewCostTracker() *CostTracker {
	ct := &CostTracker{
		prices: map[string]ModelPrice{
			"llama-3.3-70b-versatile":                   {PromptPerMillion: 0.59, CompletionPerMillion: 0.79},
			"llama-3.1-8b-instant":                      {PromptPerMillion: 0.05, CompletionPerMillion: 0.08},
			"mixtral-8x7b-32768":                        {PromptPerMillion: 0.24, CompletionPerMillion: 0.24},
			"anthropic/claude-sonnet-4-20250514":         {PromptPerMillion: 3.00, CompletionPerMillion: 15.00},
			"anthropic/claude-haiku-4-5-20251001":        {PromptPerMillion: 0.80, CompletionPerMillion: 4.00},
			"openai/gpt-4o":                             {PromptPerMillion: 2.50, CompletionPerMillion: 10.00},
			"openai/gpt-4o-mini":                        {PromptPerMillion: 0.15, CompletionPerMillion: 0.60},
			"meta-llama/llama-4-scout-17b-16e-instruct": {PromptPerMillion: 0.11, CompletionPerMillion: 0.34},
		},
	}
	return ct
}

// SetPrice sets the pricing for a model.
func (ct *CostTracker) SetPrice(model string, promptPerMillion, completionPerMillion float64) {
	ct.mu.Lock()
	ct.prices[model] = ModelPrice{
		PromptPerMillion:     promptPerMillion,
		CompletionPerMillion: completionPerMillion,
	}
	ct.mu.Unlock()
}

// OnEvent is the callback to register with WithOnEvent. It captures usage events.
func (ct *CostTracker) OnEvent(ev agentflow.Event) {
	if ev.Type != agentflow.EventUsage || ev.Usage == nil {
		return
	}

	ct.mu.Lock()
	defer ct.mu.Unlock()

	cost := ct.estimateCost(ev.Usage.Usage.PromptTokens, ev.Usage.Usage.CompletionTokens)

	ct.usage = append(ct.usage, UsageRecord{
		Timestamp:        time.Now(),
		PromptTokens:     ev.Usage.Usage.PromptTokens,
		CompletionTokens: ev.Usage.Usage.CompletionTokens,
		EstimatedCost:    cost,
	})
}

// TrackUsage manually records a usage event with a specific model for accurate pricing.
func (ct *CostTracker) TrackUsage(model string, promptTokens, completionTokens int) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	cost := ct.estimateCostForModel(model, promptTokens, completionTokens)
	ct.usage = append(ct.usage, UsageRecord{
		Timestamp:        time.Now(),
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		EstimatedCost:    cost,
	})
}

// TotalCost returns the total estimated cost across all tracked API calls.
func (ct *CostTracker) TotalCost() float64 {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	var total float64
	for _, u := range ct.usage {
		total += u.EstimatedCost
	}
	return total
}

// TotalTokens returns aggregate token counts.
func (ct *CostTracker) TotalTokens() (prompt, completion int) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	for _, u := range ct.usage {
		prompt += u.PromptTokens
		completion += u.CompletionTokens
	}
	return
}

// Records returns all usage records.
func (ct *CostTracker) Records() []UsageRecord {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	return append([]UsageRecord{}, ct.usage...)
}

func (ct *CostTracker) estimateCost(promptTokens, completionTokens int) float64 {
	// Use a generic average rate when model is unknown.
	return (float64(promptTokens) * 0.50 / 1_000_000) + (float64(completionTokens) * 1.00 / 1_000_000)
}

func (ct *CostTracker) estimateCostForModel(model string, promptTokens, completionTokens int) float64 {
	price, ok := ct.prices[model]
	if !ok {
		return ct.estimateCost(promptTokens, completionTokens)
	}
	return (float64(promptTokens) * price.PromptPerMillion / 1_000_000) +
		(float64(completionTokens) * price.CompletionPerMillion / 1_000_000)
}
