package builtin

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode"

	"github.com/CanArslanDev/agentflow"
)

// Calculator returns a tool that safely evaluates mathematical expressions.
// Models frequently make errors in arithmetic, percentages, and multi-step
// calculations. This tool provides exact results.
//
//	agent := agentflow.NewAgent(provider, agentflow.WithTool(builtin.Calculator()))
func Calculator() agentflow.Tool { return &calculatorTool{} }

type calculatorTool struct{}

func (t *calculatorTool) Name() string { return "calculator" }
func (t *calculatorTool) Description() string {
	return "Evaluate a mathematical expression and return the exact result. Supports +, -, *, /, parentheses, %, and ^ (power). Use this for any arithmetic, percentage, or multi-step calculation instead of computing mentally."
}

func (t *calculatorTool) InputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"expression": map[string]any{
				"type":        "string",
				"description": "Mathematical expression to evaluate (e.g. \"15 * 28\", \"(100 - 15) / 3\", \"2 ^ 10\", \"150 * 1.18\")",
			},
		},
		"required": []string{"expression"},
	}
}

func (t *calculatorTool) Execute(ctx context.Context, input json.RawMessage, _ agentflow.ProgressFunc) (*agentflow.ToolResult, error) {
	var params struct {
		Expression string `json:"expression"`
	}
	if err := json.Unmarshal(input, &params); err != nil {
		return &agentflow.ToolResult{Content: "invalid input: " + err.Error(), IsError: true}, nil
	}

	expr := strings.TrimSpace(params.Expression)
	if expr == "" {
		return &agentflow.ToolResult{Content: "empty expression", IsError: true}, nil
	}

	result, err := evalExpr(expr)
	if err != nil {
		return &agentflow.ToolResult{
			Content: fmt.Sprintf("calculation error: %s", err.Error()),
			IsError: true,
		}, nil
	}

	// Format: integer if whole number, otherwise up to 10 decimal places.
	var formatted string
	if result == math.Trunc(result) && !math.IsInf(result, 0) {
		formatted = strconv.FormatFloat(result, 'f', 0, 64)
	} else {
		formatted = strconv.FormatFloat(result, 'f', -1, 64)
	}

	return &agentflow.ToolResult{
		Content: fmt.Sprintf(`{"expression":"%s","result":%s}`, expr, formatted),
	}, nil
}

func (t *calculatorTool) IsConcurrencySafe(_ json.RawMessage) bool { return true }
func (t *calculatorTool) IsReadOnly(_ json.RawMessage) bool        { return true }
func (t *calculatorTool) Locality() agentflow.ToolLocality          { return agentflow.ToolAny }

// --- Safe expression evaluator (no eval, no external deps) ---

// evalExpr evaluates a mathematical expression using recursive descent parsing.
// Supports: +, -, *, /, ^ (power), % (modulo), parentheses, unary minus.
func evalExpr(expr string) (float64, error) {
	p := &exprParser{input: expr}
	result := p.parseExpression()
	if p.err != nil {
		return 0, p.err
	}
	if p.pos < len(p.input) {
		return 0, fmt.Errorf("unexpected character at position %d: %q", p.pos, string(p.input[p.pos]))
	}
	return result, nil
}

type exprParser struct {
	input string
	pos   int
	err   error
}

func (p *exprParser) skipSpaces() {
	for p.pos < len(p.input) && unicode.IsSpace(rune(p.input[p.pos])) {
		p.pos++
	}
}

func (p *exprParser) parseExpression() float64 {
	return p.parseAddSub()
}

func (p *exprParser) parseAddSub() float64 {
	left := p.parseMulDiv()
	for p.err == nil && p.pos < len(p.input) {
		p.skipSpaces()
		if p.pos >= len(p.input) {
			break
		}
		op := p.input[p.pos]
		if op != '+' && op != '-' {
			break
		}
		p.pos++
		right := p.parseMulDiv()
		if op == '+' {
			left += right
		} else {
			left -= right
		}
	}
	return left
}

func (p *exprParser) parseMulDiv() float64 {
	left := p.parsePower()
	for p.err == nil && p.pos < len(p.input) {
		p.skipSpaces()
		if p.pos >= len(p.input) {
			break
		}
		op := p.input[p.pos]
		if op != '*' && op != '/' && op != '%' {
			break
		}
		p.pos++
		right := p.parsePower()
		switch op {
		case '*':
			left *= right
		case '/':
			if right == 0 {
				p.err = fmt.Errorf("division by zero")
				return 0
			}
			left /= right
		case '%':
			if right == 0 {
				p.err = fmt.Errorf("modulo by zero")
				return 0
			}
			left = math.Mod(left, right)
		}
	}
	return left
}

func (p *exprParser) parsePower() float64 {
	base := p.parseUnary()
	p.skipSpaces()
	if p.err == nil && p.pos < len(p.input) && p.input[p.pos] == '^' {
		p.pos++
		exp := p.parseUnary()
		return math.Pow(base, exp)
	}
	return base
}

func (p *exprParser) parseUnary() float64 {
	p.skipSpaces()
	if p.pos < len(p.input) && p.input[p.pos] == '-' {
		p.pos++
		return -p.parseUnary()
	}
	if p.pos < len(p.input) && p.input[p.pos] == '+' {
		p.pos++
		return p.parseUnary()
	}
	return p.parsePrimary()
}

func (p *exprParser) parsePrimary() float64 {
	p.skipSpaces()
	if p.err != nil {
		return 0
	}
	if p.pos >= len(p.input) {
		p.err = fmt.Errorf("unexpected end of expression")
		return 0
	}

	// Parenthesized sub-expression.
	if p.input[p.pos] == '(' {
		p.pos++
		val := p.parseExpression()
		p.skipSpaces()
		if p.pos < len(p.input) && p.input[p.pos] == ')' {
			p.pos++
		} else {
			p.err = fmt.Errorf("missing closing parenthesis")
		}
		return val
	}

	// Number literal.
	start := p.pos
	for p.pos < len(p.input) && (p.input[p.pos] >= '0' && p.input[p.pos] <= '9' || p.input[p.pos] == '.') {
		p.pos++
	}
	if p.pos == start {
		p.err = fmt.Errorf("expected number at position %d", p.pos)
		return 0
	}
	val, err := strconv.ParseFloat(p.input[start:p.pos], 64)
	if err != nil {
		p.err = fmt.Errorf("invalid number: %s", p.input[start:p.pos])
		return 0
	}
	return val
}
