package sse

import (
	"testing"
)

func TestThinkTagParser_NoTags(t *testing.T) {
	var p thinkTagParser
	segments := p.process("Hello World")

	if len(segments) != 1 {
		t.Fatalf("expected 1 segment, got %d", len(segments))
	}
	if segments[0].text != "Hello World" || segments[0].thinking {
		t.Errorf("expected normal 'Hello World', got %+v", segments[0])
	}
}

func TestThinkTagParser_FullThinkBlock(t *testing.T) {
	var p thinkTagParser
	segments := p.process("<think>Let me think about this.</think>The answer is 42.")

	if len(segments) != 2 {
		t.Fatalf("expected 2 segments, got %d: %+v", len(segments), segments)
	}
	if segments[0].text != "Let me think about this." || !segments[0].thinking {
		t.Errorf("segment 0: %+v", segments[0])
	}
	if segments[1].text != "The answer is 42." || segments[1].thinking {
		t.Errorf("segment 1: %+v", segments[1])
	}
}

func TestThinkTagParser_AcrossChunks(t *testing.T) {
	var p thinkTagParser

	// Chunk 1: opening tag + partial thinking
	seg1 := p.process("<think>Let me analyze")
	if len(seg1) != 1 || seg1[0].text != "Let me analyze" || !seg1[0].thinking {
		t.Errorf("chunk 1: %+v", seg1)
	}

	// Chunk 2: more thinking + closing tag + normal text
	seg2 := p.process(" this question.</think>\n\nThe answer is 42.")
	if len(seg2) != 2 {
		t.Fatalf("chunk 2: expected 2 segments, got %d: %+v", len(seg2), seg2)
	}
	if seg2[0].text != " this question." || !seg2[0].thinking {
		t.Errorf("chunk 2, segment 0: %+v", seg2[0])
	}
	if seg2[1].text != "\n\nThe answer is 42." || seg2[1].thinking {
		t.Errorf("chunk 2, segment 1: %+v", seg2[1])
	}
}

func TestThinkTagParser_SplitOpenTag(t *testing.T) {
	var p thinkTagParser

	// Tag split: "<thi" | "nk>thinking"
	seg1 := p.process("Hello<thi")
	if len(seg1) != 1 || seg1[0].text != "Hello" || seg1[0].thinking {
		t.Errorf("chunk 1: %+v", seg1)
	}

	seg2 := p.process("nk>thinking content")
	if len(seg2) != 1 || seg2[0].text != "thinking content" || !seg2[0].thinking {
		t.Errorf("chunk 2: %+v", seg2)
	}
}

func TestThinkTagParser_SplitCloseTag(t *testing.T) {
	var p thinkTagParser

	// Enter thinking state.
	p.process("<think>start")

	// Close tag split: "end</thi" | "nk>after"
	seg1 := p.process("end</thi")
	if len(seg1) != 1 || seg1[0].text != "end" || !seg1[0].thinking {
		t.Errorf("chunk 1: %+v", seg1)
	}

	seg2 := p.process("nk>after")
	if len(seg2) != 1 || seg2[0].text != "after" || seg2[0].thinking {
		t.Errorf("chunk 2: %+v", seg2)
	}
}

func TestThinkTagParser_OnlyThinking(t *testing.T) {
	var p thinkTagParser
	segments := p.process("<think>All thinking, no answer.</think>")

	if len(segments) != 1 {
		t.Fatalf("expected 1 segment, got %d: %+v", len(segments), segments)
	}
	if segments[0].text != "All thinking, no answer." || !segments[0].thinking {
		t.Errorf("segment: %+v", segments[0])
	}
}

func TestThinkTagParser_EmptyThinking(t *testing.T) {
	var p thinkTagParser
	segments := p.process("<think></think>The answer.")

	if len(segments) != 1 {
		t.Fatalf("expected 1 segment, got %d: %+v", len(segments), segments)
	}
	if segments[0].text != "The answer." || segments[0].thinking {
		t.Errorf("segment: %+v", segments[0])
	}
}

func TestThinkTagParser_ThinkingOnlyNoClose(t *testing.T) {
	var p thinkTagParser

	// Thinking that never closes (still streaming).
	seg1 := p.process("<think>Still thinking")
	if len(seg1) != 1 || seg1[0].text != "Still thinking" || !seg1[0].thinking {
		t.Errorf("chunk 1: %+v", seg1)
	}

	seg2 := p.process(" more thoughts")
	if len(seg2) != 1 || seg2[0].text != " more thoughts" || !seg2[0].thinking {
		t.Errorf("chunk 2: %+v", seg2)
	}
}

func TestThinkTagParser_TagAtChunkBoundary_Exact(t *testing.T) {
	var p thinkTagParser

	// Tag exactly at chunk boundary.
	seg1 := p.process("<think>")
	if len(seg1) != 0 {
		t.Errorf("expected 0 segments for bare <think>, got %d: %+v", len(seg1), seg1)
	}
	if !p.thinking {
		t.Error("should be in thinking state")
	}

	seg2 := p.process("content</think>")
	if len(seg2) != 1 || seg2[0].text != "content" || !seg2[0].thinking {
		t.Errorf("chunk 2: %+v", seg2)
	}
	if p.thinking {
		t.Error("should be in normal state")
	}
}

func TestThinkTagParser_TextBeforeAndAfterTag(t *testing.T) {
	var p thinkTagParser
	segments := p.process("Before<think>Inside</think>After")

	if len(segments) != 3 {
		t.Fatalf("expected 3 segments, got %d: %+v", len(segments), segments)
	}
	if segments[0].text != "Before" || segments[0].thinking {
		t.Errorf("segment 0: %+v", segments[0])
	}
	if segments[1].text != "Inside" || !segments[1].thinking {
		t.Errorf("segment 1: %+v", segments[1])
	}
	if segments[2].text != "After" || segments[2].thinking {
		t.Errorf("segment 2: %+v", segments[2])
	}
}

func TestThinkTagParser_GroqExampleFlow(t *testing.T) {
	// Simulate the exact Groq compound-beta flow from the bug report.
	var p thinkTagParser

	seg1 := p.process("<think>\nLet me analyze")
	if len(seg1) != 1 || seg1[0].text != "\nLet me analyze" || !seg1[0].thinking {
		t.Errorf("chunk 1: %+v", seg1)
	}

	seg2 := p.process(" this question.\n</think>\n\n")
	if len(seg2) != 2 {
		t.Fatalf("chunk 2: expected 2 segments, got %d: %+v", len(seg2), seg2)
	}
	if seg2[0].text != " this question.\n" || !seg2[0].thinking {
		t.Errorf("chunk 2 seg 0: %+v", seg2[0])
	}
	if seg2[1].text != "\n\n" || seg2[1].thinking {
		t.Errorf("chunk 2 seg 1: %+v", seg2[1])
	}

	seg3 := p.process("The answer is 42.")
	if len(seg3) != 1 || seg3[0].text != "The answer is 42." || seg3[0].thinking {
		t.Errorf("chunk 3: %+v", seg3)
	}
}

// --- thinkTagStripper tests (reasoning field tag removal) ---

func TestThinkTagStripper_StripToolTags(t *testing.T) {
	var s thinkTagStripper
	result := s.strip("I need to calculate.\n\n<tool>python(print(15*28))</tool>\n<output>420\n</output>\nThe answer is 420.")

	expected := "I need to calculate.\n\npython(print(15*28))\n420\n\nThe answer is 420."
	if result != expected {
		t.Errorf("got:  %q\nwant: %q", result, expected)
	}
}

func TestThinkTagStripper_StripAllTags(t *testing.T) {
	var s thinkTagStripper
	result := s.strip("<think>Hello<tool>python(x)</tool><output>result</output></think>")

	expected := "Hellopython(x)result"
	if result != expected {
		t.Errorf("got: %q, want: %q", result, expected)
	}
}

func TestThinkTagStripper_SplitToolTag(t *testing.T) {
	var s thinkTagStripper

	r1 := s.strip("before<too")
	if r1 != "before" {
		t.Errorf("chunk 1: %q", r1)
	}

	r2 := s.strip("l>python(x)</tool>after")
	if r2 != "python(x)after" {
		t.Errorf("chunk 2: %q", r2)
	}
}

func TestThinkTagStripper_SplitOutputTag(t *testing.T) {
	var s thinkTagStripper

	r1 := s.strip("data</outp")
	if r1 != "data" {
		t.Errorf("chunk 1: %q", r1)
	}

	r2 := s.strip("ut>after")
	if r2 != "after" {
		t.Errorf("chunk 2: %q", r2)
	}
}

func TestThinkTagStripper_NoTags(t *testing.T) {
	var s thinkTagStripper
	result := s.strip("Just normal text")
	if result != "Just normal text" {
		t.Errorf("got: %q", result)
	}
}

func TestThinkTagStripper_GroqCompoundExample(t *testing.T) {
	// Simulate the exact Groq compound-beta reasoning format.
	var s thinkTagStripper

	r1 := s.strip("<think>\nToday's date is 2026-04-05.\n\n")
	if r1 != "\nToday's date is 2026-04-05.\n\n" {
		t.Errorf("chunk 1: %q", r1)
	}

	r2 := s.strip("<tool>python(import datetime; print(datetime.datetime(2026, 4, 5).strftime(\"%A\")))</tool>\n")
	if r2 != "python(import datetime; print(datetime.datetime(2026, 4, 5).strftime(\"%A\")))\n" {
		t.Errorf("chunk 2: %q", r2)
	}

	r3 := s.strip("<output>Sunday\n</output>\n\nThe answer is Sunday.\n")
	if r3 != "Sunday\n\n\nThe answer is Sunday.\n" {
		t.Errorf("chunk 3: %q", r3)
	}

	r4 := s.strip("</think>\n\nBugun Pazar.")
	if r4 != "\n\nBugun Pazar." {
		t.Errorf("chunk 4: %q", r4)
	}
}

func TestMatchPartialSuffix(t *testing.T) {
	tests := []struct {
		text     string
		tag      string
		expected int
	}{
		{"hello<thi", "<think>", 4},    // "<thi" matches "<thi" (first 4 chars of <think>)
		{"hello<", "<think>", 1},       // "<" matches "<" (first 1 char)
		{"hello<think", "<think>", 6},  // "<think" matches (first 6 chars)
		{"hello", "<think>", 0},        // no match
		{"end</", "</think>", 2},       // "</" matches
		{"end</think", "</think>", 7},  // "</think" matches
		{"x<t", "<think>", 2},         // "<t" matches
		{"", "<think>", 0},            // empty
	}

	for _, tt := range tests {
		got := matchPartialSuffix(tt.text, tt.tag)
		if got != tt.expected {
			t.Errorf("matchPartialSuffix(%q, %q) = %d, want %d", tt.text, tt.tag, got, tt.expected)
		}
	}
}
