package agentflow

import "sync/atomic"

// TokenBudget controls the maximum token consumption for a single agent run.
// When the budget is exhausted, the agentic loop terminates gracefully with
// TurnEndBudgetExhausted. A warning event is emitted when the threshold is crossed.
type TokenBudget struct {
	// MaxTokens is the total token limit for the entire agent run.
	// Includes both prompt and completion tokens across all turns.
	MaxTokens int

	// WarningThreshold is the fraction (0.0–1.0) at which a budget warning
	// event is emitted. For example, 0.8 means warn at 80% consumption.
	// Zero disables the warning. Values above 1.0 are clamped to 1.0.
	WarningThreshold float64
}

// budgetTracker is the runtime state for tracking token consumption.
// It is created per agent run and is not shared across runs.
type budgetTracker struct {
	budget       TokenBudget
	consumed     atomic.Int64
	warningFired atomic.Bool
}

// newBudgetTracker creates a tracker for the given budget.
// Returns nil if no budget is configured (MaxTokens <= 0).
func newBudgetTracker(budget *TokenBudget) *budgetTracker {
	if budget == nil || budget.MaxTokens <= 0 {
		return nil
	}
	threshold := budget.WarningThreshold
	if threshold > 1.0 {
		threshold = 1.0
	}
	return &budgetTracker{
		budget: TokenBudget{
			MaxTokens:        budget.MaxTokens,
			WarningThreshold: threshold,
		},
	}
}

// record adds token usage and returns whether the budget is now exhausted.
func (bt *budgetTracker) record(usage *Usage) (exhausted bool) {
	if usage == nil {
		return false
	}
	bt.consumed.Add(int64(usage.TotalTokens))
	return bt.consumed.Load() >= int64(bt.budget.MaxTokens)
}

// shouldWarn returns true exactly once when the warning threshold is crossed.
func (bt *budgetTracker) shouldWarn() bool {
	if bt.budget.WarningThreshold <= 0 {
		return false
	}
	threshold := int64(float64(bt.budget.MaxTokens) * bt.budget.WarningThreshold)
	if bt.consumed.Load() >= threshold && bt.warningFired.CompareAndSwap(false, true) {
		return true
	}
	return false
}

// remaining returns the number of tokens left in the budget.
func (bt *budgetTracker) remaining() int {
	r := int64(bt.budget.MaxTokens) - bt.consumed.Load()
	if r < 0 {
		return 0
	}
	return int(r)
}

// totalConsumed returns the total tokens consumed so far.
func (bt *budgetTracker) totalConsumed() int {
	return int(bt.consumed.Load())
}
