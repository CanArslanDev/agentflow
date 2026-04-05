package builtin

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/CanArslanDev/agentflow"
)

// URLReader returns a tool that fetches a web page and extracts readable text
// content. Unlike http_request which returns raw response data, this tool
// strips HTML tags, scripts, and styles to return clean, readable text that
// models can understand.
//
//	agent := agentflow.NewAgent(provider, agentflow.WithTool(builtin.URLReader()))
func URLReader() agentflow.Tool { return &urlReaderTool{} }

type urlReaderTool struct{}

func (t *urlReaderTool) Name() string { return "read_url" }
func (t *urlReaderTool) Description() string {
	return "Fetch a web page and extract its readable text content. Use this when you need to read a specific URL that the user provided or that you found in search results. Returns clean text without HTML markup."
}

func (t *urlReaderTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"url": map[string]any{
				"type":        "string",
				"description": "The URL to read (e.g. https://example.com/article)",
			},
		},
		"required": []string{"url"},
	}
}

const (
	urlReaderTimeout    = 15 * time.Second
	urlReaderMaxBytes   = 512 * 1024 // 512KB max download
	urlReaderMaxContent = 10000      // 10K chars max output
)

func (t *urlReaderTool) Execute(ctx context.Context, input json.RawMessage, progress agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var params struct {
		URL string `json:"url"`
	}
	if err := json.Unmarshal(input, &params); err != nil {
		return &agentflow.ToolResult{Content: "invalid input: " + err.Error(), IsError: true}, nil
	}

	url := strings.TrimSpace(params.URL)
	if url == "" {
		return &agentflow.ToolResult{Content: "URL is required", IsError: true}, nil
	}
	if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
		url = "https://" + url
	}

	progress(agentflow.ProgressEvent{Message: "Fetching: " + url})

	reqCtx, cancel := context.WithTimeout(ctx, urlReaderTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, url, nil)
	if err != nil {
		return &agentflow.ToolResult{Content: "invalid URL: " + err.Error(), IsError: true}, nil
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; agentflow/1.0)")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,text/plain,*/*")

	client := &http.Client{
		Timeout: urlReaderTimeout,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 5 {
				return fmt.Errorf("too many redirects")
			}
			return nil
		},
	}

	resp, err := client.Do(req)
	if err != nil {
		return &agentflow.ToolResult{Content: "failed to fetch URL: " + err.Error(), IsError: true}, nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return &agentflow.ToolResult{
			Content: fmt.Sprintf("HTTP %d: %s", resp.StatusCode, resp.Status),
			IsError: true,
		}, nil
	}

	// Read body with size limit.
	body, err := io.ReadAll(io.LimitReader(resp.Body, urlReaderMaxBytes))
	if err != nil {
		return &agentflow.ToolResult{Content: "failed to read response: " + err.Error(), IsError: true}, nil
	}

	// Extract readable text from HTML.
	text := extractReadableText(string(body))
	if text == "" {
		return &agentflow.ToolResult{Content: "page returned no readable content"}, nil
	}

	// Truncate if too long.
	if len(text) > urlReaderMaxContent {
		text = text[:urlReaderMaxContent] + "\n\n[content truncated]"
	}

	return &agentflow.ToolResult{
		Content: text,
		Metadata: map[string]any{
			"url":           url,
			"status":        resp.StatusCode,
			"content_length": len(text),
		},
	}, nil
}

func (t *urlReaderTool) IsConcurrencySafe(_ json.RawMessage) bool { return true }
func (t *urlReaderTool) IsReadOnly(_ json.RawMessage) bool        { return true }
func (t *urlReaderTool) Locality() agentflow.ToolLocality          { return agentflow.ToolRemoteSafe }

// NOTE: extractReadableText is defined in deepsearch.go and shared with this tool.
