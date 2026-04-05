package builtin

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/CanArslanDev/agentflow"
)

// CurrentDateTime returns a tool that provides the current date, time, and
// day of week from the system clock. Models cannot reliably determine the
// current date from training data or web searches (JS-rendered sites,
// hallucinated dates). This tool provides an authoritative answer.
//
//	agent := agentflow.NewAgent(provider, agentflow.WithTool(builtin.CurrentDateTime()))
func CurrentDateTime() agentflow.Tool { return &currentDateTimeTool{} }

type currentDateTimeTool struct{}

func (t *currentDateTimeTool) Name() string { return "get_current_datetime" }
func (t *currentDateTimeTool) Description() string {
	return "Get the current date, day of week, and time. Use this when the user asks about today's date, current time, what day it is, or any time-related question. Returns accurate real-time data from the system clock."
}

func (t *currentDateTimeTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"timezone": map[string]any{
				"type":        "string",
				"description": "IANA timezone (e.g. Europe/Istanbul, America/New_York, Asia/Tokyo). Defaults to UTC if not specified.",
			},
		},
	}
}

func (t *currentDateTimeTool) Execute(ctx context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var params struct {
		Timezone string `json:"timezone"`
	}
	json.Unmarshal(input, &params)

	if params.Timezone == "" {
		params.Timezone = "UTC"
	}

	loc, err := time.LoadLocation(params.Timezone)
	if err != nil {
		return &agentflow.ToolResult{
			Content: fmt.Sprintf("Unknown timezone: %q. Use IANA format like Europe/Istanbul, America/New_York, or Asia/Tokyo.", params.Timezone),
			IsError: true,
		}, nil
	}

	now := time.Now().In(loc)
	return &agentflow.ToolResult{
		Content: fmt.Sprintf(`{"date":"%s","day":"%s","time":"%s","timezone":"%s"}`,
			now.Format("2006-01-02"),
			now.Format("Monday"),
			now.Format("15:04:05"),
			params.Timezone,
		),
	}, nil
}

func (t *currentDateTimeTool) IsConcurrencySafe(_ json.RawMessage) bool { return true }
func (t *currentDateTimeTool) IsReadOnly(_ json.RawMessage) bool        { return true }
func (t *currentDateTimeTool) Locality() agentflow.ToolLocality          { return agentflow.ToolAny }
