// Package sse provides shared types and SSE stream parsing for OpenAI-compatible
// chat completion APIs. Used by provider/openrouter, provider/groq, and any
// future providers that speak the OpenAI wire format.
package sse

import "encoding/json"

// --- Request types ---

// ChatRequest is the OpenAI-compatible chat completion request body.
type ChatRequest struct {
	Model       string           `json:"model"`
	Messages    []RequestMessage `json:"messages"`
	Tools       []RequestTool    `json:"tools,omitempty"`
	MaxTokens   int              `json:"max_tokens,omitempty"`
	Temperature *float64         `json:"temperature,omitempty"`
	Stream      bool             `json:"stream"`
	Stop        []string         `json:"stop,omitempty"`
}

// RequestMessage is a single message in the OpenAI format.
type RequestMessage struct {
	Role       string            `json:"role"`
	Content    any               `json:"content"`
	ToolCalls  []RequestToolCall `json:"tool_calls,omitempty"`
	ToolCallID string            `json:"tool_call_id,omitempty"`
}

// ContentPart is an element within a multimodal message content array.
// Used when a message contains text, images, or documents.
type ContentPart struct {
	Type     string       `json:"type"`               // "text", "image_url", or "file"
	Text     string       `json:"text,omitempty"`      // For type "text"
	ImageURL *ImageURL    `json:"image_url,omitempty"` // For type "image_url"
	File     *FileContent `json:"file,omitempty"`      // For type "file"
}

// ImageURL holds an image reference for multimodal messages.
type ImageURL struct {
	URL string `json:"url"` // base64 data URI or HTTP URL
}

// FileContent holds a file reference for document content parts.
type FileContent struct {
	Filename string `json:"filename"`  // original filename
	FileData string `json:"file_data"` // data URI or URL
}

// RequestTool describes a tool available to the model.
type RequestTool struct {
	Type     string          `json:"type"`
	Function RequestFunction `json:"function"`
}

// RequestFunction describes a callable function.
type RequestFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// RequestToolCall is an assistant-generated tool call in conversation history.
type RequestToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall carries the function name and JSON arguments.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// --- Response types (SSE streaming) ---

// StreamChunk is a single SSE chunk from a streaming chat completion.
type StreamChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
	Usage   *ChunkUsage    `json:"usage,omitempty"`
}

// StreamChoice is a single choice within a streaming chunk.
type StreamChoice struct {
	Index        int         `json:"index"`
	Delta        StreamDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason,omitempty"`
}

// StreamDelta carries incremental content from the model.
type StreamDelta struct {
	Role      string                `json:"role,omitempty"`
	Content   *string               `json:"content,omitempty"`
	ToolCalls []StreamToolCallDelta `json:"tool_calls,omitempty"`
	Reasoning *string               `json:"reasoning,omitempty"`
}

// StreamToolCallDelta carries incremental tool call data.
type StreamToolCallDelta struct {
	Index    int                 `json:"index"`
	ID       string              `json:"id,omitempty"`
	Type     string              `json:"type,omitempty"`
	Function StreamFunctionDelta `json:"function"`
}

// StreamFunctionDelta carries incremental function call data.
type StreamFunctionDelta struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// ChunkUsage reports token usage in a streaming chunk.
type ChunkUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ToolCallAccumulator assembles a complete tool call from streaming deltas.
type ToolCallAccumulator struct {
	ID        string
	Name      string
	Arguments string
}

// ToJSON returns the accumulated arguments as json.RawMessage.
// Returns "{}" if the accumulated arguments are empty or not valid JSON.
func (acc *ToolCallAccumulator) ToJSON() json.RawMessage {
	if acc.Arguments == "" {
		return json.RawMessage("{}")
	}
	if !json.Valid([]byte(acc.Arguments)) {
		return json.RawMessage("{}")
	}
	return json.RawMessage(acc.Arguments)
}
