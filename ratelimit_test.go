package agentflow_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/mock"
)

func TestTokenBucketLimiter_ImmediateBurst(t *testing.T) {
	// 5 tokens, so first 5 calls should be immediate.
	limiter := agentflow.NewTokenBucketLimiter(5, time.Second)

	start := time.Now()
	for i := 0; i < 5; i++ {
		if err := limiter.Wait(context.Background()); err != nil {
			t.Fatalf("burst call %d: %v", i, err)
		}
	}
	elapsed := time.Since(start)

	if elapsed > 50*time.Millisecond {
		t.Errorf("burst of 5 should be near-instant, took %v", elapsed)
	}
}

func TestTokenBucketLimiter_ThrottlesAfterBurst(t *testing.T) {
	// 2 tokens per 200ms = 10 req/sec, burst of 2.
	limiter := agentflow.NewTokenBucketLimiter(2, 200*time.Millisecond)

	// Consume the burst.
	limiter.Wait(context.Background())
	limiter.Wait(context.Background())

	// Third call should wait ~100ms (1 token / 10 per sec).
	start := time.Now()
	limiter.Wait(context.Background())
	elapsed := time.Since(start)

	if elapsed < 50*time.Millisecond {
		t.Errorf("expected throttling, but elapsed only %v", elapsed)
	}
	if elapsed > 300*time.Millisecond {
		t.Errorf("waited too long: %v", elapsed)
	}
	t.Logf("Wait after burst: %v", elapsed)
}

func TestTokenBucketLimiter_ContextCancellation(t *testing.T) {
	limiter := agentflow.NewTokenBucketLimiter(1, time.Second)

	// Consume the single token.
	limiter.Wait(context.Background())

	// Next call with cancelled context should fail immediately.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := limiter.Wait(ctx)
	if err == nil {
		t.Fatal("expected context error")
	}
}

func TestTokenBucketLimiter_ConcurrentAccess(t *testing.T) {
	limiter := agentflow.NewTokenBucketLimiter(10, time.Second)

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			limiter.Wait(ctx)
		}()
	}
	wg.Wait()
}

func TestAgentWithRateLimiter(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(mock.TextDelta("response 1")),
		mock.WithResponse(mock.TextDelta("response 2")),
		mock.WithResponse(mock.TextDelta("response 3")),
	)

	// 2 req/sec — enough for the test, but forces measurable delays.
	limiter := agentflow.NewTokenBucketLimiter(2, time.Second)

	agent := agentflow.NewAgent(provider,
		agentflow.WithMaxTurns(1),
		agentflow.WithRateLimiter(limiter),
	)

	// Run 3 sequential single-turn requests.
	start := time.Now()
	for i := 0; i < 3; i++ {
		for range agent.Run(context.Background(), []agentflow.Message{
			agentflow.NewUserMessage("hello"),
		}) {
		}
	}
	elapsed := time.Since(start)

	t.Logf("3 requests took %v", elapsed)
	// First 2 should be instant (burst), third should wait ~500ms.
	if elapsed < 200*time.Millisecond {
		t.Error("expected some rate limiting delay")
	}
}
