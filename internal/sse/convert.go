package sse

import (
	"fmt"

	"github.com/CanArslanDev/agentflow"
)

// ConvertMessages transforms agentflow messages into the OpenAI-compatible format.
// Handles text, tool calls, tool results, and multimodal (image) content.
func ConvertMessages(messages []agentflow.Message, systemPrompt string) []RequestMessage {
	var result []RequestMessage

	if systemPrompt != "" {
		result = append(result, RequestMessage{Role: "system", Content: systemPrompt})
	}

	for _, msg := range messages {
		switch msg.Role {
		case agentflow.RoleUser:
			result = append(result, convertUserMessage(msg)...)
		case agentflow.RoleAssistant:
			result = append(result, convertAssistantMessage(msg))
		case agentflow.RoleSystem:
			result = append(result, RequestMessage{Role: "system", Content: msg.TextContent()})
		}
	}

	return result
}

// convertUserMessage handles user messages that may contain text, images, or tool results.
func convertUserMessage(msg agentflow.Message) []RequestMessage {
	toolResults := msg.ToolResults()
	if len(toolResults) > 0 {
		msgs := make([]RequestMessage, len(toolResults))
		for i, tr := range toolResults {
			msgs[i] = RequestMessage{
				Role:       "tool",
				Content:    tr.Content,
				ToolCallID: tr.ToolCallID,
			}
		}
		return msgs
	}

	// Check for multimodal content (images).
	images := msg.Images()
	if len(images) > 0 {
		parts := buildMultimodalParts(msg)
		return []RequestMessage{{Role: "user", Content: parts}}
	}

	return []RequestMessage{{Role: "user", Content: msg.TextContent()}}
}

// buildMultimodalParts builds an array of content parts for multimodal messages.
func buildMultimodalParts(msg agentflow.Message) []ContentPart {
	var parts []ContentPart
	for _, block := range msg.Content {
		switch block.Type {
		case agentflow.ContentText:
			if block.Text != "" {
				parts = append(parts, ContentPart{Type: "text", Text: block.Text})
			}
		case agentflow.ContentImage:
			if block.Image != nil {
				url := block.Image.URL
				if url == "" && block.Image.Data != "" {
					// Convert base64 to data URI.
					mediaType := block.Image.MediaType
					if mediaType == "" {
						mediaType = "image/png"
					}
					url = fmt.Sprintf("data:%s;base64,%s", mediaType, block.Image.Data)
				}
				if url != "" {
					parts = append(parts, ContentPart{
						Type:     "image_url",
						ImageURL: &ImageURL{URL: url},
					})
				}
			}
		}
	}
	return parts
}

// convertAssistantMessage handles assistant messages with optional tool calls.
func convertAssistantMessage(msg agentflow.Message) RequestMessage {
	rm := RequestMessage{Role: "assistant", Content: msg.TextContent()}
	toolCalls := msg.ToolCalls()
	if len(toolCalls) > 0 {
		rm.ToolCalls = make([]RequestToolCall, len(toolCalls))
		for i, tc := range toolCalls {
			rm.ToolCalls[i] = RequestToolCall{
				ID:   tc.ID,
				Type: "function",
				Function: FunctionCall{
					Name:      tc.Name,
					Arguments: string(tc.Input),
				},
			}
		}
	}
	return rm
}

// ConvertTools transforms agentflow tool definitions into OpenAI format.
func ConvertTools(tools []agentflow.ToolDefinition) []RequestTool {
	if len(tools) == 0 {
		return nil
	}
	result := make([]RequestTool, len(tools))
	for i, t := range tools {
		result[i] = RequestTool{
			Type: "function",
			Function: RequestFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			},
		}
	}
	return result
}

// BuildRequestBody creates an OpenAI-compatible chat request from agentflow types.
func BuildRequestBody(model string, req *agentflow.Request) ChatRequest {
	messages := ConvertMessages(req.Messages, req.SystemPrompt)
	tools := ConvertTools(req.Tools)

	cr := ChatRequest{
		Model:    model,
		Messages: messages,
		Stream:   true,
	}
	if len(tools) > 0 {
		cr.Tools = tools
	}
	if req.MaxTokens > 0 {
		cr.MaxTokens = req.MaxTokens
	}
	if req.Temperature != nil {
		cr.Temperature = req.Temperature
	}
	if len(req.StopSequences) > 0 {
		cr.Stop = req.StopSequences
	}
	return cr
}
