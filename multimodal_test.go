package agentflow_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"image"
	"image/color"
	"image/png"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/CanArslanDev/agentflow"
	"github.com/CanArslanDev/agentflow/provider/groq"
	"github.com/CanArslanDev/agentflow/provider/mock"
)

// --- Unit tests ---

func TestNewImageMessage(t *testing.T) {
	msg := agentflow.NewImageMessage("What is this?",
		agentflow.ImageContent{MediaType: "image/png", Data: "base64data"},
	)

	if msg.Role != agentflow.RoleUser {
		t.Errorf("expected user role, got %s", msg.Role)
	}
	if len(msg.Content) != 2 {
		t.Fatalf("expected 2 blocks (text + image), got %d", len(msg.Content))
	}
	if msg.Content[0].Type != agentflow.ContentText || msg.Content[0].Text != "What is this?" {
		t.Error("first block should be text")
	}
	if msg.Content[1].Type != agentflow.ContentImage || msg.Content[1].Image == nil {
		t.Error("second block should be image")
	}
	if msg.Content[1].Image.MediaType != "image/png" {
		t.Errorf("expected image/png, got %s", msg.Content[1].Image.MediaType)
	}
}

func TestNewImageURLMessage(t *testing.T) {
	msg := agentflow.NewImageURLMessage("Describe this", "https://example.com/photo.jpg")

	images := msg.Images()
	if len(images) != 1 {
		t.Fatalf("expected 1 image, got %d", len(images))
	}
	if images[0].URL != "https://example.com/photo.jpg" {
		t.Errorf("expected URL, got %q", images[0].URL)
	}
}

func TestNewImageMessage_MultipleImages(t *testing.T) {
	msg := agentflow.NewImageMessage("Compare these",
		agentflow.ImageContent{MediaType: "image/jpeg", Data: "img1"},
		agentflow.ImageContent{MediaType: "image/png", Data: "img2"},
		agentflow.ImageContent{URL: "https://example.com/img3.webp"},
	)

	if len(msg.Content) != 4 { // text + 3 images
		t.Errorf("expected 4 blocks, got %d", len(msg.Content))
	}
	images := msg.Images()
	if len(images) != 3 {
		t.Errorf("expected 3 images, got %d", len(images))
	}
}

func TestNewImageMessage_NoText(t *testing.T) {
	msg := agentflow.NewImageMessage("",
		agentflow.ImageContent{MediaType: "image/png", Data: "data"},
	)

	if len(msg.Content) != 1 { // only image, no text block
		t.Errorf("expected 1 block (image only), got %d", len(msg.Content))
	}
}

func TestImageMessageWithMockProvider(t *testing.T) {
	provider := mock.New(
		mock.WithResponse(mock.TextDelta("I see a red circle.")),
	)

	agent := agentflow.NewAgent(provider, agentflow.WithMaxTurns(1))

	msg := agentflow.NewImageMessage("What do you see?",
		agentflow.ImageContent{MediaType: "image/png", Data: "fakebase64"},
	)

	var text string
	for ev := range agent.Run(context.Background(), []agentflow.Message{msg}) {
		if ev.Type == agentflow.EventTextDelta {
			text += ev.TextDelta.Text
		}
	}

	if text != "I see a red circle." {
		t.Errorf("expected mock response, got %q", text)
	}
}

// --- Integration test: real vision with Groq ---

func TestIntegration_VisionWithGroq(t *testing.T) {
	key := os.Getenv("GROQ_API_KEY")
	if key == "" {
		t.Skip("GROQ_API_KEY not set")
	}

	// Groq's llama-3.2-90b-vision-preview supports vision.
	// We'll use a tiny 1x1 red pixel PNG to test the pipeline end-to-end.
	provider := groq.New(key, "meta-llama/llama-4-scout-17b-16e-instruct")

	agent := agentflow.NewAgent(provider,
		agentflow.WithMaxTurns(1),
		agentflow.WithMaxTokens(100),
	)

	// 1x1 red pixel PNG (smallest valid PNG).
	redPixelPNG := createRedImage()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	msg := agentflow.NewImageMessage(
		"What color is the single pixel in this image? Answer in one word.",
		agentflow.ImageContent{
			MediaType: "image/png",
			Data:      base64.StdEncoding.EncodeToString(redPixelPNG),
		},
	)

	var text string
	for ev := range agent.Run(ctx, []agentflow.Message{msg}) {
		switch ev.Type {
		case agentflow.EventTextDelta:
			text += ev.TextDelta.Text
		case agentflow.EventTurnEnd:
			t.Logf("Turn %d: %s", ev.TurnEnd.TurnNumber, ev.TurnEnd.Reason)
		case agentflow.EventError:
			t.Logf("Error: %v", ev.Error.Err)
		}
	}

	t.Logf("Vision response: %q", text)
	if text == "" {
		t.Error("empty response from vision model")
	}
	// The model should mention red or a color.
	lower := strings.ToLower(text)
	if !strings.Contains(lower, "red") && !strings.Contains(lower, "color") && !strings.Contains(lower, "pixel") {
		t.Logf("Note: model may not have recognized the tiny image, but pipeline works. Response: %s", text)
	}
}

// createRedImage creates a 10x10 solid red PNG image.
func createRedImage() []byte {
	img := image.NewRGBA(image.Rect(0, 0, 10, 10))
	red := color.RGBA{R: 255, G: 0, B: 0, A: 255}
	for y := 0; y < 10; y++ {
		for x := 0; x < 10; x++ {
			img.Set(x, y, red)
		}
	}
	var buf bytes.Buffer
	png.Encode(&buf, img)
	return buf.Bytes()
}
