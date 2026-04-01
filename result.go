package agentflow

import (
	"fmt"
	"strings"
)

// ResultLimiter controls how oversized tool results are handled before being
// sent back to the model. Large results can exhaust the context window and
// degrade model performance.
type ResultLimiter interface {
	// Limit inspects a tool result and returns a potentially modified version.
	// If the result is within acceptable bounds, it is returned unchanged.
	// maxChars is the configured maximum character count.
	Limit(result *ToolResult, maxChars int) *ToolResult
}

// Default maximum result size in characters.
const DefaultMaxResultChars = 50000

// --- Built-in ResultLimiter implementations ---

// TruncateLimiter truncates oversized results with a notice showing how much
// content was omitted. The truncation preserves the beginning and end of the
// content for maximum context.
type TruncateLimiter struct{}

// Limit truncates the result content if it exceeds maxChars. It preserves the
// first 80% and last 20% of the allowed size, inserting a truncation notice
// in the middle.
func (TruncateLimiter) Limit(result *ToolResult, maxChars int) *ToolResult {
	if maxChars <= 0 || len(result.Content) <= maxChars {
		return result
	}

	omitted := len(result.Content) - maxChars
	notice := fmt.Sprintf("\n\n[... %d characters truncated ...]\n\n", omitted)

	// Reserve space for the notice itself.
	available := maxChars - len(notice)
	if available <= 0 {
		return &ToolResult{
			Content:  fmt.Sprintf("[content truncated: %d characters omitted]", len(result.Content)),
			IsError:  result.IsError,
			Metadata: result.Metadata,
		}
	}

	headSize := available * 4 / 5 // 80% head
	tailSize := available - headSize

	var b strings.Builder
	b.Grow(maxChars)
	b.WriteString(result.Content[:headSize])
	b.WriteString(notice)
	b.WriteString(result.Content[len(result.Content)-tailSize:])

	return &ToolResult{
		Content:  b.String(),
		IsError:  result.IsError,
		Metadata: mergeMetadata(result.Metadata, map[string]any{"truncated_chars": omitted}),
	}
}

// HeadTailLimiter keeps only the first N and last M characters, discarding
// the middle. Simpler than TruncateLimiter with configurable head/tail ratio.
type HeadTailLimiter struct {
	// HeadRatio is the fraction of maxChars allocated to the head (0.0–1.0).
	// Default: 0.7 (70% head, 30% tail).
	HeadRatio float64
}

// Limit applies head/tail truncation.
func (l HeadTailLimiter) Limit(result *ToolResult, maxChars int) *ToolResult {
	if maxChars <= 0 || len(result.Content) <= maxChars {
		return result
	}

	ratio := l.HeadRatio
	if ratio <= 0 || ratio >= 1 {
		ratio = 0.7
	}

	omitted := len(result.Content) - maxChars
	notice := fmt.Sprintf("\n[... %d chars omitted ...]\n", omitted)
	available := maxChars - len(notice)
	if available <= 0 {
		return &ToolResult{
			Content:  fmt.Sprintf("[truncated: %d chars]", len(result.Content)),
			IsError:  result.IsError,
			Metadata: result.Metadata,
		}
	}

	head := int(float64(available) * ratio)
	tail := available - head

	var b strings.Builder
	b.Grow(maxChars)
	b.WriteString(result.Content[:head])
	b.WriteString(notice)
	b.WriteString(result.Content[len(result.Content)-tail:])

	return &ToolResult{
		Content:  b.String(),
		IsError:  result.IsError,
		Metadata: mergeMetadata(result.Metadata, map[string]any{"truncated_chars": omitted}),
	}
}

// NoLimiter passes all results through without modification. Use when you
// handle result sizing externally or don't need protection.
type NoLimiter struct{}

// Limit returns the result unchanged.
func (NoLimiter) Limit(result *ToolResult, _ int) *ToolResult {
	return result
}

// mergeMetadata creates a new map combining base and extra entries.
func mergeMetadata(base, extra map[string]any) map[string]any {
	result := make(map[string]any, len(base)+len(extra))
	for k, v := range base {
		result[k] = v
	}
	for k, v := range extra {
		result[k] = v
	}
	return result
}
