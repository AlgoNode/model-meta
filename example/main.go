// Command example enumerates the catalogue of a vLLM (or any
// OpenAI-compatible) endpoint and prints the deduplicated, feature-resolved
// result as JSON.
//
// Usage:
//
//	go run ./example                       # http://localhost:8000
//	go run ./example http://host:8000      # explicit endpoint
//	VLLM_API_KEY=... HF_TOKEN=... go run ./example
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/algonode/model-meta"
)

const defaultEndpoint = "http://localhost:8000"

func main() {
	endpoint := defaultEndpoint
	if len(os.Args) > 1 && os.Args[1] != "" {
		endpoint = os.Args[1]
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	e := &modelmeta.Enumerator{
		EndpointURL: endpoint,
		APIKey:      os.Getenv("VLLM_API_KEY"),
		HFToken:     os.Getenv("HF_TOKEN"),
	}

	models, err := e.Enumerate(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "enumerate %s: %v\n", endpoint, err)
		os.Exit(1)
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(models); err != nil {
		fmt.Fprintf(os.Stderr, "encode: %v\n", err)
		os.Exit(1)
	}
}
