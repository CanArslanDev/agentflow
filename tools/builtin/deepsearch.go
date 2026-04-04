package builtin

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/CanArslanDev/agentflow"
)

// DeepSearch returns a tool that performs multi-step web research. Unlike
// web_search which only returns titles and snippets, deep_search also fetches
// the content of the most relevant result pages and extracts readable text.
//
// Flow: search query -> get results -> fetch top N pages -> extract text -> combine
func DeepSearch() agentflow.Tool { return &deepSearchTool{} }

type deepSearchTool struct{}

func (t *deepSearchTool) Name() string { return "deep_search" }
func (t *deepSearchTool) Description() string {
	return "Perform deep web research on a topic. Searches the web, then fetches and reads the content of the top result pages to gather detailed information. Use this when you need thorough, detailed information rather than just search snippets. Returns search results plus extracted page content."
}
func (t *deepSearchTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query": map[string]any{
				"type":        "string",
				"description": "The search query to research",
			},
			"max_pages": map[string]any{
				"type":        "integer",
				"description": "Number of result pages to fetch and read (default: 3, max: 5)",
			},
			"max_content_chars": map[string]any{
				"type":        "integer",
				"description": "Maximum characters of content to extract per page (default: 5000, max: 15000)",
			},
		},
		"required": []string{"query"},
	}
}

func (t *deepSearchTool) Execute(ctx context.Context, input json.RawMessage, progress agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var params struct {
		Query           string `json:"query"`
		MaxPages        int    `json:"max_pages"`
		MaxContentChars int    `json:"max_content_chars"`
	}
	if err := json.Unmarshal(input, &params); err != nil {
		return &agentflow.ToolResult{Content: "invalid input: " + err.Error(), IsError: true}, nil
	}

	if params.Query == "" {
		return &agentflow.ToolResult{Content: "query is required", IsError: true}, nil
	}

	maxPages := params.MaxPages
	if maxPages <= 0 {
		maxPages = 3
	}
	if maxPages > 5 {
		maxPages = 5
	}

	maxContentChars := params.MaxContentChars
	if maxContentChars <= 0 {
		maxContentChars = 5000
	}
	if maxContentChars > 15000 {
		maxContentChars = 15000
	}

	// Step 1: Search.
	if progress != nil {
		progress(agentflow.ProgressEvent{Message: "Searching: " + params.Query})
	}

	results, err := duckDuckGoSearch(ctx, params.Query, maxPages+2) // fetch a few extra in case some fail
	if err != nil {
		return &agentflow.ToolResult{Content: "search failed: " + err.Error(), IsError: true}, nil
	}

	if len(results) == 0 {
		return &agentflow.ToolResult{Content: "no results found for: " + params.Query}, nil
	}

	// Step 2: Fetch top pages concurrently.
	if progress != nil {
		progress(agentflow.ProgressEvent{Message: fmt.Sprintf("Fetching content from %d pages...", min(maxPages, len(results)))})
	}

	type pageContent struct {
		index   int
		url     string
		title   string
		content string
		err     error
	}

	pagesToFetch := min(maxPages, len(results))
	contents := make([]pageContent, pagesToFetch)
	var wg sync.WaitGroup

	for i := 0; i < pagesToFetch; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			fetchCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
			defer cancel()

			content, err := fetchPageContent(fetchCtx, results[idx].url, maxContentChars)
			contents[idx] = pageContent{
				index:   idx,
				url:     results[idx].url,
				title:   results[idx].title,
				content: content,
				err:     err,
			}
		}(i)
	}

	wg.Wait()

	// Step 3: Build combined result.
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("## Deep Search Results for: %s\n\n", params.Query))

	// Search results summary.
	sb.WriteString("### Search Results\n\n")
	for i, r := range results {
		sb.WriteString(fmt.Sprintf("%d. **%s**\n   %s\n   %s\n\n", i+1, r.title, r.url, r.snippet))
	}

	// Page contents.
	sb.WriteString("### Page Contents\n\n")
	fetchedCount := 0
	for _, pc := range contents {
		if pc.err != nil {
			sb.WriteString(fmt.Sprintf("--- [%s] (failed: %v) ---\n\n", pc.title, pc.err))
			continue
		}
		if pc.content == "" {
			continue
		}
		fetchedCount++
		sb.WriteString(fmt.Sprintf("--- [%s] (%s) ---\n\n", pc.title, pc.url))
		sb.WriteString(pc.content)
		sb.WriteString("\n\n")
	}

	if fetchedCount == 0 {
		sb.WriteString("(no page content could be extracted)\n")
	}

	return &agentflow.ToolResult{
		Content: sb.String(),
		Metadata: map[string]any{
			"search_results": len(results),
			"pages_fetched":  fetchedCount,
		},
	}, nil
}

func (t *deepSearchTool) IsConcurrencySafe(_ json.RawMessage) bool { return true }
func (t *deepSearchTool) IsReadOnly(_ json.RawMessage) bool        { return true }
func (t *deepSearchTool) Locality() agentflow.ToolLocality          { return agentflow.ToolRemoteSafe }

// fetchPageContent fetches a URL and extracts readable text content.
func fetchPageContent(ctx context.Context, pageURL string, maxChars int) (string, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", pageURL, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; agentflow/1.0)")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,text/plain")

	client := &http.Client{
		Timeout: 15 * time.Second,
		CheckRedirect: func(_ *http.Request, via []*http.Request) error {
			if len(via) >= 5 {
				return fmt.Errorf("too many redirects")
			}
			return nil
		},
	}

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return "", fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	// Read limited body.
	body, err := io.ReadAll(io.LimitReader(resp.Body, 512*1024)) // 512KB raw limit
	if err != nil {
		return "", err
	}

	text := extractReadableText(string(body))

	// Truncate to maxChars.
	if utf8.RuneCountInString(text) > maxChars {
		runes := []rune(text)
		text = string(runes[:maxChars]) + "\n...(truncated)"
	}

	return text, nil
}

// extractReadableText strips HTML tags, scripts, styles, and extracts
// meaningful text content from an HTML page.
func extractReadableText(html string) string {
	// Remove script and style blocks entirely.
	for _, tag := range []string{"script", "style", "noscript", "nav", "footer", "header"} {
		for {
			openTag := "<" + tag
			idx := strings.Index(strings.ToLower(html), openTag)
			if idx == -1 {
				break
			}
			closeTag := "</" + tag + ">"
			endIdx := strings.Index(strings.ToLower(html[idx:]), closeTag)
			if endIdx == -1 {
				// Remove to end if no closing tag.
				html = html[:idx]
				break
			}
			html = html[:idx] + html[idx+endIdx+len(closeTag):]
		}
	}

	// Strip remaining HTML tags.
	text := stripHTML(html)

	// Clean up whitespace.
	lines := strings.Split(text, "\n")
	var cleaned []string
	prevEmpty := false
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			if !prevEmpty {
				cleaned = append(cleaned, "")
				prevEmpty = true
			}
			continue
		}
		prevEmpty = false
		cleaned = append(cleaned, line)
	}

	result := strings.Join(cleaned, "\n")
	result = strings.TrimSpace(result)

	return result
}
