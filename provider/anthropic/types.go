package anthropic

// --- Request types (Anthropic Messages API format) ---

type messagesRequest struct {
	Model     string           `json:"model"`
	Messages  []requestMessage `json:"messages"`
	System    string           `json:"system,omitempty"`
	MaxTokens int              `json:"max_tokens"`
	Stream    bool             `json:"stream"`
	Tools     []requestTool    `json:"tools,omitempty"`
}

type requestMessage struct {
	Role    string         `json:"role"`
	Content []contentBlock `json:"content"`
}

type contentBlock struct {
	Type string `json:"type"`

	// type: "text"
	Text string `json:"text,omitempty"`

	// type: "image"
	Source *imageSource `json:"source,omitempty"`

	// type: "tool_use"
	ID    string `json:"id,omitempty"`
	Name  string `json:"name,omitempty"`
	Input any    `json:"input,omitempty"`

	// type: "tool_result"
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   any    `json:"content,omitempty"` // string or []contentBlock
	IsError   bool   `json:"is_error,omitempty"`
}

type imageSource struct {
	Type      string `json:"type"`       // "base64"
	MediaType string `json:"media_type"` // "image/png", etc.
	Data      string `json:"data"`       // base64 encoded
}

type requestTool struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	InputSchema toolSchema `json:"input_schema"`
}

type toolSchema struct {
	Type       string         `json:"type"`
	Properties map[string]any `json:"properties,omitempty"`
	Required   []string       `json:"required,omitempty"`
}

// --- Response types (Anthropic SSE streaming) ---

// Anthropic SSE events have these types:
// message_start, content_block_start, content_block_delta, content_block_stop,
// message_delta, message_stop

type sseEvent struct {
	Type string `json:"type"`

	// message_start
	Message *messageStartPayload `json:"message,omitempty"`

	// content_block_start
	Index        int           `json:"index,omitempty"`
	ContentBlock *contentBlock `json:"content_block,omitempty"`

	// content_block_delta
	Delta *deltaPayload `json:"delta,omitempty"`

	// message_delta
	Usage *usagePayload `json:"usage,omitempty"`
}

type messageStartPayload struct {
	ID    string        `json:"id"`
	Model string        `json:"model"`
	Usage *usagePayload `json:"usage,omitempty"`
}

type deltaPayload struct {
	Type string `json:"type"` // "text_delta", "input_json_delta"
	Text string `json:"text,omitempty"`

	// For input_json_delta (tool use arguments streaming)
	PartialJSON string `json:"partial_json,omitempty"`

	// For message_delta
	StopReason string `json:"stop_reason,omitempty"`
}

type usagePayload struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}
