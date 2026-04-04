package agentflow

import (
	"context"
	"sync"
	"time"
)

// RateLimiter controls the rate of API calls to a provider. Implementations
// block until the caller is permitted to proceed, or return an error if the
// context is cancelled while waiting.
type RateLimiter interface {
	// Wait blocks until the caller is allowed to proceed. Returns an error
	// if the context is cancelled or the limiter is otherwise unable to grant
	// permission.
	Wait(ctx context.Context) error
}

// TokenBucketLimiter implements a token bucket rate limiting algorithm.
// It allows bursts up to the bucket capacity and refills tokens at a steady rate.
//
//	limiter := agentflow.NewTokenBucketLimiter(10, time.Second) // 10 requests/sec
//	agent := agentflow.NewAgent(provider,
//	    agentflow.WithRateLimiter(limiter),
//	)
type TokenBucketLimiter struct {
	mu         sync.Mutex
	tokens     float64
	capacity   float64
	refillRate float64 // tokens per nanosecond
	lastRefill time.Time
}

// NewTokenBucketLimiter creates a rate limiter that allows up to capacity
// requests per interval. The bucket starts full, allowing an initial burst.
//
//	NewTokenBucketLimiter(5, time.Second)    // 5 req/sec, burst of 5
//	NewTokenBucketLimiter(60, time.Minute)   // 60 req/min, burst of 60
//	NewTokenBucketLimiter(1, 200*time.Millisecond) // 5 req/sec, no burst
func NewTokenBucketLimiter(capacity int, interval time.Duration) *TokenBucketLimiter {
	cap := float64(capacity)
	return &TokenBucketLimiter{
		tokens:     cap,
		capacity:   cap,
		refillRate: cap / float64(interval),
		lastRefill: time.Now(),
	}
}

// Wait blocks until a token is available or the context is cancelled.
func (l *TokenBucketLimiter) Wait(ctx context.Context) error {
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		l.mu.Lock()
		l.refill()
		if l.tokens >= 1 {
			l.tokens--
			l.mu.Unlock()
			return nil
		}
		// Calculate wait time for next token.
		waitDuration := time.Duration(float64(time.Nanosecond) * (1 - l.tokens) / l.refillRate)
		l.mu.Unlock()

		if waitDuration < time.Millisecond {
			waitDuration = time.Millisecond
		}

		select {
		case <-time.After(waitDuration):
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// refill adds tokens based on elapsed time since last refill.
// Must be called with l.mu held.
func (l *TokenBucketLimiter) refill() {
	now := time.Now()
	elapsed := now.Sub(l.lastRefill)
	l.tokens += float64(elapsed) * l.refillRate
	if l.tokens > l.capacity {
		l.tokens = l.capacity
	}
	l.lastRefill = now
}
