// Package gemini provides an agentflow Provider implementation for the Google
// Gemini API (https://ai.google.dev/api/generate-content).
//
// Gemini uses a different format than OpenAI — function calls are in a separate
// "functionCall" part, and the API key is passed as a query parameter.
//
//	provider := gemini.New("AIza...", "gemini-2.0-flash")
//	agent := agentflow.NewAgent(provider, agentflow.WithTools(myTool))
package gemini

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/CanArslanDev/agentflow"
)

const (
	defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"
	defaultTimeout = 5 * time.Minute
)

// Provider implements agentflow.Provider for the Google Gemini API.
type Provider struct {
	apiKey  string
	model   string
	baseURL string
	client  *http.Client
}

// ProviderOption configures the Gemini provider.
type ProviderOption func(*Provider)

// New creates a Gemini provider targeting the specified model.
//
//	provider := gemini.New("AIza...", "gemini-2.0-flash")
//	provider := gemini.New("AIza...", "gemini-2.5-pro-preview-06-05")
//	provider := gemini.New("AIza...", "gemini-2.5-flash-preview-05-20")
func New(apiKey, model string, opts ...ProviderOption) *Provider {
	p := &Provider{
		apiKey:  apiKey,
		model:   model,
		baseURL: defaultBaseURL,
		client: &http.Client{
			Timeout: defaultTimeout,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

// WithBaseURL overrides the API base URL.
func WithBaseURL(url string) ProviderOption {
	return func(p *Provider) { p.baseURL = url }
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(c *http.Client) ProviderOption {
	return func(p *Provider) { p.client = c }
}

// CreateStream initiates a streaming generateContent request.
func (p *Provider) CreateStream(ctx context.Context, req *agentflow.Request) (agentflow.Stream, error) {
	body := p.buildRequestBody(req)

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("agentflow/gemini: marshal request: %w", err)
	}

	// Gemini uses API key as query parameter, not in Authorization header.
	url := fmt.Sprintf("%s/models/%s:streamGenerateContent?alt=sse&key=%s", p.baseURL, p.model, p.apiKey)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("agentflow/gemini: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	for k, v := range req.Metadata {
		httpReq.Header.Set(k, v)
	}

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, &agentflow.ProviderError{StatusCode: 0, Message: err.Error(), Retryable: true, Err: err}
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, p.parseErrorResponse(resp)
	}

	return newGeminiStream(resp), nil
}

// ModelID returns the configured model identifier.
func (p *Provider) ModelID() string { return p.model }

// --- Request body building ---

type generateRequest struct {
	Contents         []geminiContent      `json:"contents"`
	SystemInstruction *geminiContent      `json:"systemInstruction,omitempty"`
	Tools            []geminiToolDef      `json:"tools,omitempty"`
	GenerationConfig *generationConfig    `json:"generationConfig,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text             string            `json:"text,omitempty"`
	FunctionCall     *functionCall     `json:"functionCall,omitempty"`
	FunctionResponse *functionResponse `json:"functionResponse,omitempty"`
	InlineData       *inlineData       `json:"inlineData,omitempty"`
}

type functionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

type functionResponse struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type inlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type geminiToolDef struct {
	FunctionDeclarations []functionDeclaration `json:"functionDeclarations"`
}

type functionDeclaration struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

type generationConfig struct {
	MaxOutputTokens int      `json:"maxOutputTokens,omitempty"`
	Temperature     *float64 `json:"temperature,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
}

func (p *Provider) buildRequestBody(req *agentflow.Request) generateRequest {
	gr := generateRequest{
		Contents: p.convertMessages(req.Messages),
	}

	if req.SystemPrompt != "" {
		gr.SystemInstruction = &geminiContent{
			Role:  "user",
			Parts: []geminiPart{{Text: req.SystemPrompt}},
		}
	}

	if len(req.Tools) > 0 {
		decls := make([]functionDeclaration, len(req.Tools))
		for i, t := range req.Tools {
			decls[i] = functionDeclaration{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			}
		}
		gr.Tools = []geminiToolDef{{FunctionDeclarations: decls}}
	}

	if req.MaxTokens > 0 || req.Temperature != nil || len(req.StopSequences) > 0 {
		gc := &generationConfig{}
		if req.MaxTokens > 0 {
			gc.MaxOutputTokens = req.MaxTokens
		}
		if req.Temperature != nil {
			gc.Temperature = req.Temperature
		}
		if len(req.StopSequences) > 0 {
			gc.StopSequences = req.StopSequences
		}
		gr.GenerationConfig = gc
	}

	return gr
}

func (p *Provider) convertMessages(messages []agentflow.Message) []geminiContent {
	var result []geminiContent

	for _, msg := range messages {
		role := "user"
		if msg.Role == agentflow.RoleAssistant {
			role = "model"
		}

		var parts []geminiPart
		for _, b := range msg.Content {
			switch b.Type {
			case agentflow.ContentText:
				if b.Text != "" {
					parts = append(parts, geminiPart{Text: b.Text})
				}
			case agentflow.ContentImage:
				if b.Image != nil && b.Image.Data != "" {
					mt := b.Image.MediaType
					if mt == "" {
						mt = "image/png"
					}
					parts = append(parts, geminiPart{
						InlineData: &inlineData{MimeType: mt, Data: b.Image.Data},
					})
				}
			case agentflow.ContentDocument:
				if b.Document != nil && b.Document.Data != "" {
					mt := b.Document.MediaType
					if mt == "" {
						mt = "application/octet-stream"
					}
					parts = append(parts, geminiPart{
						InlineData: &inlineData{MimeType: mt, Data: b.Document.Data},
					})
				}
			case agentflow.ContentToolCall:
				if b.ToolCall != nil {
					var args map[string]any
					json.Unmarshal(b.ToolCall.Input, &args)
					parts = append(parts, geminiPart{
						FunctionCall: &functionCall{Name: b.ToolCall.Name, Args: args},
					})
				}
			case agentflow.ContentToolResult:
				if b.ToolResult != nil {
					parts = append(parts, geminiPart{
						FunctionResponse: &functionResponse{
							Name:     b.ToolResult.ToolCallID,
							Response: map[string]any{"result": b.ToolResult.Content},
						},
					})
				}
			}
		}

		if len(parts) == 0 {
			parts = []geminiPart{{Text: ""}}
		}
		result = append(result, geminiContent{Role: role, Parts: parts})
	}

	return result
}

func (p *Provider) parseErrorResponse(resp *http.Response) error {
	var body struct {
		Error struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
			Status  string `json:"status"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return &agentflow.ProviderError{StatusCode: resp.StatusCode, Message: "failed to parse error", Retryable: resp.StatusCode >= 500}
	}
	return &agentflow.ProviderError{
		StatusCode: resp.StatusCode,
		Message:    body.Error.Message,
		Retryable:  resp.StatusCode == 429 || resp.StatusCode >= 500,
	}
}

// --- Gemini SSE Stream ---

type geminiStreamResponse struct {
	Candidates    []candidate    `json:"candidates"`
	UsageMetadata *usageMetadata `json:"usageMetadata,omitempty"`
}

type candidate struct {
	Content       geminiContent `json:"content"`
	FinishReason  string        `json:"finishReason,omitempty"`
}

type usageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

type geminiStream struct {
	resp    *http.Response
	scanner *bufio.Scanner
	usage   *agentflow.Usage
	done    bool
	callIdx int

	// Pending events from a single chunk with multiple parts.
	pending []agentflow.StreamEvent
}

func newGeminiStream(resp *http.Response) *geminiStream {
	return &geminiStream{
		resp:    resp,
		scanner: bufio.NewScanner(resp.Body),
	}
}

func (s *geminiStream) Next() (agentflow.StreamEvent, error) {
	// Drain any pending events from a previous multi-part chunk.
	if len(s.pending) > 0 {
		ev := s.pending[0]
		s.pending = s.pending[1:]
		return ev, nil
	}

	for {
		if s.done {
			return agentflow.StreamEvent{}, io.EOF
		}

		if !s.scanner.Scan() {
			if err := s.scanner.Err(); err != nil {
				return agentflow.StreamEvent{}, err
			}
			return agentflow.StreamEvent{}, io.EOF
		}

		line := s.scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		var resp geminiStreamResponse
		if err := json.Unmarshal([]byte(data), &resp); err != nil {
			continue
		}

		if resp.UsageMetadata != nil {
			s.usage = &agentflow.Usage{
				PromptTokens:     resp.UsageMetadata.PromptTokenCount,
				CompletionTokens: resp.UsageMetadata.CandidatesTokenCount,
				TotalTokens:      resp.UsageMetadata.TotalTokenCount,
			}
		}

		if len(resp.Candidates) == 0 {
			continue
		}

		cand := resp.Candidates[0]

		// Collect events from all parts in this chunk.
		var events []agentflow.StreamEvent
		for _, part := range cand.Content.Parts {
			if part.Text != "" {
				events = append(events, agentflow.StreamEvent{
					Type:  agentflow.StreamEventDelta,
					Delta: &agentflow.ContentDelta{Text: part.Text},
				})
			}
			if part.FunctionCall != nil {
				argsJSON, _ := json.Marshal(part.FunctionCall.Args)
				if !json.Valid(argsJSON) {
					argsJSON = json.RawMessage("{}")
				}
				s.callIdx++
				events = append(events, agentflow.StreamEvent{
					Type: agentflow.StreamEventToolCall,
					ToolCall: &agentflow.ToolCall{
						ID:    fmt.Sprintf("call_%d", s.callIdx),
						Name:  part.FunctionCall.Name,
						Input: argsJSON,
					},
				})
			}
		}

		if len(events) > 0 {
			if len(events) > 1 {
				s.pending = events[1:]
			}
			return events[0], nil
		}

		if cand.FinishReason == "STOP" || cand.FinishReason == "MAX_TOKENS" {
			s.done = true
			return agentflow.StreamEvent{}, io.EOF
		}
	}
}

func (s *geminiStream) Close() error {
	if s.resp != nil && s.resp.Body != nil {
		return s.resp.Body.Close()
	}
	return nil
}

func (s *geminiStream) Usage() *agentflow.Usage {
	return s.usage
}
