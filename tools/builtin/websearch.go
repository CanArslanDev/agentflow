package builtin

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/CanArslanDev/agentflow"
)

// WebSearch returns a tool that searches the web and fetches actual page content.
// For each search result, it fetches the URL and extracts readable text, giving
// the model real website content instead of just snippets.
// Remote-safe — performs only network calls, no local filesystem access.
func WebSearch() agentflow.Tool { return &webSearchTool{} }

type webSearchTool struct{}

func (t *webSearchTool) Name() string { return "web_search" }
func (t *webSearchTool) Description() string {
	return "Search the web and fetch actual page content. Returns search results with real website text, not just snippets. Use this when you need up-to-date information, facts, or current events."
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

const (
	searchTimeout       = 10 * time.Second
	contentFetchTimeout = 7 * time.Second
	maxContentPerPage   = 3000 // chars per page
	maxConcurrentFetch  = 5
)

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

	// Step 1: DuckDuckGo search for URLs.
	results, err := duckDuckGoSearch(ctx, params.Query, maxResults)
	if err != nil {
		return &agentflow.ToolResult{Content: "search failed: " + err.Error(), IsError: true}, nil
	}

	if len(results) == 0 {
		return &agentflow.ToolResult{Content: "no results found for: " + params.Query}, nil
	}

	if progress != nil {
		progress(agentflow.ProgressEvent{Message: fmt.Sprintf("Found %d results, fetching content...", len(results))})
	}

	// Step 2: Fetch actual page content for each result (parallel, with timeout).
	contentMap := fetchPageContents(ctx, results)

	// Step 3: Format results with content.
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Search results for \"%s\":\n\n", params.Query))
	for i, r := range results {
		sb.WriteString(fmt.Sprintf("%d. %s\n", i+1, r.title))
		sb.WriteString(fmt.Sprintf("   URL: %s\n", r.url))
		if r.snippet != "" {
			sb.WriteString(fmt.Sprintf("   Description: %s\n", r.snippet))
		}
		if content, ok := contentMap[r.url]; ok && content != "" {
			sb.WriteString(fmt.Sprintf("   Content: %s\n", content))
		}
		sb.WriteString("\n")
	}

	return &agentflow.ToolResult{
		Content:  sb.String(),
		Metadata: map[string]any{"result_count": len(results), "query": params.Query},
	}, nil
}

func (t *webSearchTool) IsConcurrencySafe(_ json.RawMessage) bool { return true }
func (t *webSearchTool) IsReadOnly(_ json.RawMessage) bool        { return true }
func (t *webSearchTool) Locality() agentflow.ToolLocality          { return agentflow.ToolRemoteSafe }

// --- Page content fetching ---

// fetchPageContents fetches the readable text content of each search result URL
// concurrently. Returns a map of URL -> extracted text content.
func fetchPageContents(ctx context.Context, results []searchResult) map[string]string {
	contentMap := make(map[string]string)
	var mu sync.Mutex
	var wg sync.WaitGroup

	sem := make(chan struct{}, maxConcurrentFetch)

	for _, r := range results {
		resultURL := r.url
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			content := fetchSinglePage(ctx, resultURL)
			if content != "" {
				mu.Lock()
				contentMap[resultURL] = content
				mu.Unlock()
			}
		}()
	}

	wg.Wait()
	return contentMap
}

// fetchSinglePage fetches a URL and extracts readable text content.
func fetchSinglePage(ctx context.Context, pageURL string) string {
	reqCtx, cancel := context.WithTimeout(ctx, contentFetchTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, pageURL, nil)
	if err != nil {
		return ""
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; agentflow/1.0)")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,text/plain,*/*")

	client := &http.Client{
		Timeout: contentFetchTimeout,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 3 {
				return fmt.Errorf("too many redirects")
			}
			return nil
		},
	}

	resp, err := client.Do(req)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return ""
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 512*1024))
	if err != nil {
		return ""
	}

	text := extractReadableText(string(body))
	if len(text) > maxContentPerPage {
		text = text[:maxContentPerPage] + "..."
	}
	return text
}

// --- DuckDuckGo HTML search ---

type searchResult struct {
	title   string
	url     string
	snippet string
}

// duckDuckGoSearch queries DuckDuckGo's HTML endpoint with retry on rate limit.
func duckDuckGoSearch(ctx context.Context, query string, maxResults int) ([]searchResult, error) {
	searchURL := "https://html.duckduckgo.com/html/?q=" + url.QueryEscape(query)

	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			select {
			case <-time.After(time.Duration(attempt) * time.Second):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		reqCtx, cancel := context.WithTimeout(ctx, searchTimeout)
		req, err := http.NewRequestWithContext(reqCtx, "GET", searchURL, nil)
		if err != nil {
			cancel()
			return nil, err
		}
		req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; agentflow/1.0)")
		req.Header.Set("Accept", "text/html")
		req.Header.Set("Accept-Language", "en-US,en;q=0.9")

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			cancel()
			lastErr = err
			continue
		}

		if resp.StatusCode == 429 || resp.StatusCode >= 500 {
			resp.Body.Close()
			cancel()
			lastErr = fmt.Errorf("HTTP %d from search", resp.StatusCode)
			continue
		}

		body, err := io.ReadAll(io.LimitReader(resp.Body, 512*1024))
		resp.Body.Close()
		cancel()
		if err != nil {
			return nil, err
		}

		results := parseSearchResults(string(body), maxResults)
		return results, nil
	}

	return nil, fmt.Errorf("search failed after 3 attempts: %w", lastErr)
}

// parseSearchResults extracts results from DuckDuckGo HTML response.
func parseSearchResults(html string, maxResults int) []searchResult {
	var results []searchResult

	remaining := html
	for len(results) < maxResults {
		idx := strings.Index(remaining, "class=\"result__a\"")
		if idx == -1 {
			break
		}
		remaining = remaining[idx:]

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

		resultURL := decodeDDGURL(rawURL)

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

		snippet := ""
		snippetIdx := strings.Index(remaining, "class=\"result__snippet\"")
		if snippetIdx != -1 && snippetIdx < 2000 {
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

func decodeDDGURL(raw string) string {
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
	if strings.HasPrefix(raw, "http") {
		return raw
	}
	return ""
}

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
