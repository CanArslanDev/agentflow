package trigger_test

import (
	"os"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow/provider/groq"
	"github.com/CanArslanDev/agentflow/trigger"
)

func groqProvider(t *testing.T) *groq.Provider {
	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		t.Skip("GROQ_API_KEY not set")
	}
	return groq.New(key, "llama-3.3-70b-versatile")
}

func TestIntegration_TriggerExecution(t *testing.T) {
	provider := groqProvider(t)

	var result trigger.Result
	done := make(chan struct{})

	scheduler := trigger.NewScheduler()
	scheduler.Schedule(trigger.Trigger{
		ID:           "test-trigger",
		Interval:     1 * time.Hour,
		Task:         "Say 'trigger fired' in exactly 2 words.",
		Provider:     provider,
		SystemPrompt: "You are a minimal responder. Reply in exactly the words requested.",
		MaxTurns:     1,
		MaxTokens:    50,
		OnResult: func(r trigger.Result) {
			result = r
			close(done)
		},
	})

	select {
	case <-done:
		t.Logf("Trigger result: %q (duration: %v)", result.Response, result.Duration)
		if result.Error != nil {
			t.Errorf("trigger error: %v", result.Error)
		}
		if result.Response == "" {
			t.Error("empty trigger response")
		}
	case <-time.After(30 * time.Second):
		t.Fatal("trigger timed out")
	}

	scheduler.CancelAll()
}
