package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/groq"
	"github.com/CanArslanDev/agentflow/tools/builtin"
)

func main() {
	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		fmt.Fprintln(os.Stderr, "GROQ_API_KEY environment variable required")
		os.Exit(1)
	}

	provider := groq.New(key, "llama-3.3-70b-versatile")

	agent := agentflow.NewAgent(provider,
		agentflow.WithTools(builtin.All()...),
		agentflow.WithSystemPrompt("You are a helpful assistant. Use tools when needed."),
		agentflow.WithMaxTurns(10),
		agentflow.WithMaxTokens(2048),
	)

	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println("agentflow chat (type 'exit' to quit)")
	fmt.Println("---")

	for {
		fmt.Print("\n> ")
		if !scanner.Scan() {
			break
		}
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if input == "exit" || input == "quit" {
			break
		}

		fmt.Println()
		for ev := range agent.Run(context.Background(), []agentflow.Message{
			agentflow.NewUserMessage(input),
		}) {
			switch ev.Type {
			case agentflow.EventTextDelta:
				fmt.Print(ev.TextDelta.Text)
			case agentflow.EventToolStart:
				fmt.Printf("\n[tool: %s]\n", ev.ToolStart.ToolCall.Name)
			case agentflow.EventToolEnd:
				if ev.ToolEnd.Result.IsError {
					fmt.Printf("[error: %s]\n", ev.ToolEnd.Result.Content)
				} else {
					content := ev.ToolEnd.Result.Content
					if len(content) > 200 {
						content = content[:200] + "..."
					}
					fmt.Printf("[result: %s]\n\n", content)
				}
			case agentflow.EventTurnEnd:
				fmt.Printf("\n--- turn %d: %s ---\n", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
			}
		}
	}
}
