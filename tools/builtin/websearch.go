package builtin

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/canarslan/agentflow"
)

// WebSearch returns a tool that searches the web using DuckDuckGo's HTML API.
// Remote-safe — performs only network calls, no local filesystem access.
func WebSearch() agentflow.Tool { return &webSearchTool{} }

type webSearchTool struct{}

func (t *webSearchTool) Name() string { return "web_search" }
func (t *webSearchTool) Description() string {
	return "Search the web for current information using a search query. Returns a list of relevant results with titles, URLs, and snippets. Use this when you need up-to-date information."
}
func (t *webSearchTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query": map[string]any{
				"type":        "string",
				"description": "The search query",
			},
			"max_results": map[string]any{
				"type":        "integer",
				"description": "Maximum number of results to return (default: 5, max: 10)",
			},
		},
		"required": []string{"query"},
	}
}

func (t *webSearchTool) Execute(ctx context.Context, input json.RawMessage, progress agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var params struct {
		Query      string `json:"query"`
		MaxResults int    `json:"max_results"`
	}
	if err := json.Unmarshal(input, &params); err != nil {
		return &agentflow.ToolResult{Content: "invalid input: " + err.Error(), IsError: true}, nil
	}

	if params.Query == "" {
		return &agentflow.ToolResult{Content: "query is required", IsError: true}, nil
	}

	maxResults := params.MaxResults
	if maxResults <= 0 {
		maxResults = 5
	}
	if maxResults > 10 {
		maxResults = 10
	}

	if progress != nil {
		progress(agentflow.ProgressEvent{Message: "Searching: " + params.Query})
	}

	results, err := duckDuckGoSearch(ctx, params.Query, maxResults)
	if err != nil {
		return &agentflow.ToolResult{Content: "search failed: " + err.Error(), IsError: true}, nil
	}

	if len(results) == 0 {
		return &agentflow.ToolResult{Content: "no results found for: " + params.Query}, nil
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Search results for \"%s\":\n\n", params.Query))
	for i, r := range results {
		sb.WriteString(fmt.Sprintf("%d. %s\n   %s\n   %s\n\n", i+1, r.title, r.url, r.snippet))
	}

	return &agentflow.ToolResult{
		Content:  sb.String(),
		Metadata: map[string]any{"result_count": len(results)},
	}, nil
}

func (t *webSearchTool) IsConcurrencySafe(_ json.RawMessage) bool { return true }
func (t *webSearchTool) IsReadOnly(_ json.RawMessage) bool        { return true }
func (t *webSearchTool) Locality() agentflow.ToolLocality          { return agentflow.ToolRemoteSafe }

// --- DuckDuckGo HTML search ---

type searchResult struct {
	title   string
	url     string
	snippet string
}

// duckDuckGoSearch queries DuckDuckGo's HTML endpoint and parses results.
func duckDuckGoSearch(ctx context.Context, query string, maxResults int) ([]searchResult, error) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	searchURL := "https://html.duckduckgo.com/html/?q=" + url.QueryEscape(query)

	req, err := http.NewRequestWithContext(ctx, "GET", searchURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "agentflow/1.0 (web search tool)")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 512*1024))
	if err != nil {
		return nil, err
	}

	return parseSearchResults(string(body), maxResults), nil
}

// parseSearchResults extracts results from DuckDuckGo HTML response.
// Simple, robust parsing without external HTML parser dependency.
func parseSearchResults(html string, maxResults int) []searchResult {
	var results []searchResult

	// DuckDuckGo HTML results are in <a class="result__a"> tags.
	remaining := html
	for len(results) < maxResults {
		// Find result link.
		idx := strings.Index(remaining, "class=\"result__a\"")
		if idx == -1 {
			break
		}
		remaining = remaining[idx:]

		// Extract href.
		hrefStart := strings.Index(remaining, "href=\"")
		if hrefStart == -1 {
			break
		}
		remaining = remaining[hrefStart+6:]
		hrefEnd := strings.Index(remaining, "\"")
		if hrefEnd == -1 {
			break
		}
		rawURL := remaining[:hrefEnd]
		remaining = remaining[hrefEnd:]

		// Decode DuckDuckGo redirect URL.
		resultURL := decodeDDGURL(rawURL)

		// Extract title (text between > and </a>).
		titleStart := strings.Index(remaining, ">")
		if titleStart == -1 {
			break
		}
		remaining = remaining[titleStart+1:]
		titleEnd := strings.Index(remaining, "</a>")
		if titleEnd == -1 {
			break
		}
		title := stripHTML(remaining[:titleEnd])
		remaining = remaining[titleEnd:]

		// Extract snippet from result__snippet class.
		snippet := ""
		snippetIdx := strings.Index(remaining, "class=\"result__snippet\"")
		if snippetIdx != -1 && snippetIdx < 2000 { // within reasonable distance
			snippetHTML := remaining[snippetIdx:]
			snipStart := strings.Index(snippetHTML, ">")
			if snipStart != -1 {
				snippetHTML = snippetHTML[snipStart+1:]
				snipEnd := strings.Index(snippetHTML, "</")
				if snipEnd != -1 {
					snippet = stripHTML(snippetHTML[:snipEnd])
				}
			}
		}

		if resultURL != "" && title != "" {
			results = append(results, searchResult{
				title:   strings.TrimSpace(title),
				url:     resultURL,
				snippet: strings.TrimSpace(snippet),
			})
		}
	}

	return results
}

// decodeDDGURL extracts the real URL from DuckDuckGo's redirect wrapper.
func decodeDDGURL(raw string) string {
	// DDG wraps URLs like: //duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com&...
	if strings.Contains(raw, "uddg=") {
		parts := strings.SplitN(raw, "uddg=", 2)
		if len(parts) == 2 {
			encoded := parts[1]
			ampIdx := strings.Index(encoded, "&")
			if ampIdx != -1 {
				encoded = encoded[:ampIdx]
			}
			decoded, err := url.QueryUnescape(encoded)
			if err == nil {
				return decoded
			}
		}
	}
	// Direct URL.
	if strings.HasPrefix(raw, "http") {
		return raw
	}
	return ""
}

// stripHTML removes HTML tags from a string.
func stripHTML(s string) string {
	var result strings.Builder
	inTag := false
	for _, r := range s {
		if r == '<' {
			inTag = true
			continue
		}
		if r == '>' {
			inTag = false
			continue
		}
		if !inTag {
			result.WriteRune(r)
		}
	}
	return result.String()
}
