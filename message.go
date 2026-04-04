package agentflow

import "encoding/json"

// Role identifies the author of a message in the conversation.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleSystem    Role = "system"
)

// ContentBlockType discriminates the variant within a ContentBlock.
type ContentBlockType int

const (
	// ContentText represents a plain text content block.
	ContentText ContentBlockType = iota

	// ContentToolCall represents the model's request to invoke a tool.
	ContentToolCall

	// ContentToolResult represents the outcome of a tool execution sent back to the model.
	ContentToolResult

	// ContentImage represents an image content block (base64 or URL).
	ContentImage

	// ContentDocument represents a document/file content block (base64 or URL).
	// Supported by providers that accept file inputs (PDF, text, CSV, etc.).
	ContentDocument
)

// Message is a single entry in the conversation history. Each message has a role
// and one or more content blocks that can be text, tool calls, or tool results.
type Message struct {
	Role    Role           `json:"role"`
	Content []ContentBlock `json:"content"`
}

// ContentBlock is a discriminated union within a message. Exactly one of the
// typed fields is set, determined by the Type field.
type ContentBlock struct {
	Type ContentBlockType `json:"type"`

	// Text is set when Type == ContentText.
	Text string `json:"text,omitempty"`

	// ToolCall is set when Type == ContentToolCall.
	ToolCall *ToolCall `json:"tool_call,omitempty"`

	// ToolResult is set when Type == ContentToolResult.
	ToolResult *ToolResultBlock `json:"tool_result,omitempty"`

	// Image is set when Type == ContentImage.
	Image *ImageContent `json:"image,omitempty"`

	// Document is set when Type == ContentDocument.
	Document *DocumentContent `json:"document,omitempty"`
}

// ImageContent holds image data for multimodal messages. Either Data (base64)
// or URL should be set, not both.
type ImageContent struct {
	// MediaType is the MIME type (e.g., "image/png", "image/jpeg", "image/webp", "image/gif").
	MediaType string `json:"media_type"`

	// Data is the base64-encoded image data. Set this for inline images.
	Data string `json:"data,omitempty"`

	// URL is a publicly accessible image URL. Set this for URL-referenced images.
	URL string `json:"url,omitempty"`
}

// DocumentContent holds document data for file-based messages. Either Data (base64)
// or URL should be set, not both.
type DocumentContent struct {
	// Filename is the original filename (e.g., "report.pdf").
	Filename string `json:"filename"`

	// MediaType is the MIME type (e.g., "application/pdf", "text/plain", "text/csv").
	MediaType string `json:"media_type"`

	// Data is the base64-encoded file content. Set this for inline documents.
	Data string `json:"data,omitempty"`

	// URL is a publicly accessible document URL. Set this for URL-referenced documents.
	URL string `json:"url,omitempty"`
}

// ToolResultBlock is the content block sent back to the model after tool execution.
// It references the original ToolCall by ID so the model can correlate results.
type ToolResultBlock struct {
	ToolCallID string `json:"tool_use_id"`
	Content    string `json:"content"`
	IsError    bool   `json:"is_error,omitempty"`
}

// NewUserMessage creates a Message with a single text content block from the user.
func NewUserMessage(text string) Message {
	return Message{
		Role: RoleUser,
		Content: []ContentBlock{
			{Type: ContentText, Text: text},
		},
	}
}

// NewImageMessage creates a user Message with text and one or more images.
// Use for vision/multimodal requests where you want the model to analyze images.
//
//	msg := agentflow.NewImageMessage("What's in this image?",
//	    agentflow.ImageContent{MediaType: "image/png", Data: base64Data},
//	)
func NewImageMessage(text string, images ...ImageContent) Message {
	blocks := make([]ContentBlock, 0, 1+len(images))
	if text != "" {
		blocks = append(blocks, ContentBlock{Type: ContentText, Text: text})
	}
	for i := range images {
		blocks = append(blocks, ContentBlock{Type: ContentImage, Image: &images[i]})
	}
	return Message{Role: RoleUser, Content: blocks}
}

// NewImageURLMessage is a convenience for creating an image message from a URL.
func NewImageURLMessage(text, imageURL string) Message {
	return NewImageMessage(text, ImageContent{URL: imageURL})
}

// Images extracts all image content blocks from the message.
func (m Message) Images() []ImageContent {
	var images []ImageContent
	for _, block := range m.Content {
		if block.Type == ContentImage && block.Image != nil {
			images = append(images, *block.Image)
		}
	}
	return images
}

// NewDocumentMessage creates a user Message with text and one or more documents.
// Use for requests where you want the model to analyze uploaded files.
//
//	msg := agentflow.NewDocumentMessage("Summarize this PDF",
//	    agentflow.DocumentContent{Filename: "report.pdf", MediaType: "application/pdf", Data: base64Data},
//	)
func NewDocumentMessage(text string, docs ...DocumentContent) Message {
	blocks := make([]ContentBlock, 0, 1+len(docs))
	if text != "" {
		blocks = append(blocks, ContentBlock{Type: ContentText, Text: text})
	}
	for i := range docs {
		blocks = append(blocks, ContentBlock{Type: ContentDocument, Document: &docs[i]})
	}
	return Message{Role: RoleUser, Content: blocks}
}

// Documents extracts all document content blocks from the message.
func (m Message) Documents() []DocumentContent {
	var docs []DocumentContent
	for _, block := range m.Content {
		if block.Type == ContentDocument && block.Document != nil {
			docs = append(docs, *block.Document)
		}
	}
	return docs
}

// NewAssistantMessage creates a Message with a single text content block from the assistant.
func NewAssistantMessage(text string) Message {
	return Message{
		Role: RoleAssistant,
		Content: []ContentBlock{
			{Type: ContentText, Text: text},
		},
	}
}

// TextContent extracts and concatenates all text blocks from the message.
func (m Message) TextContent() string {
	var result string
	for _, block := range m.Content {
		if block.Type == ContentText {
			result += block.Text
		}
	}
	return result
}

// ToolCalls extracts all tool call blocks from the message.
func (m Message) ToolCalls() []ToolCall {
	var calls []ToolCall
	for _, block := range m.Content {
		if block.Type == ContentToolCall && block.ToolCall != nil {
			calls = append(calls, *block.ToolCall)
		}
	}
	return calls
}

// ToolResults extracts all tool result blocks from the message.
func (m Message) ToolResults() []ToolResultBlock {
	var results []ToolResultBlock
	for _, block := range m.Content {
		if block.Type == ContentToolResult && block.ToolResult != nil {
			results = append(results, *block.ToolResult)
		}
	}
	return results
}

// newToolResultMessage creates a user message containing tool result blocks
// from the given execution results. This is the message format expected by
// AI providers: a user message with tool_result content blocks.
func newToolResultMessage(results []toolExecResult) Message {
	blocks := make([]ContentBlock, 0, len(results))
	for _, r := range results {
		blocks = append(blocks, ContentBlock{
			Type: ContentToolResult,
			ToolResult: &ToolResultBlock{
				ToolCallID: r.callID,
				Content:    r.result.Content,
				IsError:    r.result.IsError,
			},
		})
	}
	return Message{
		Role:    RoleUser,
		Content: blocks,
	}
}

// toolExecResult pairs a tool call ID with its execution result.
// This is an internal type used to bridge the executor and message builder.
type toolExecResult struct {
	callID string
	result *ToolResult
}

// ToolCall represents the model's request to invoke a tool. The ID is generated
// by the provider and must be referenced in the corresponding ToolResultBlock.
type ToolCall struct {
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}
