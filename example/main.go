// Command example resolves model metadata in one of two modes:
//
//   - default: enumerate a vLLM (or any OpenAI-compatible) endpoint and
//     print the deduplicated, feature-resolved catalogue.
//   - -m HF_ID: skip the endpoint and resolve a single HuggingFace model
//     id directly. Useful for previewing what Enumerate would produce
//     for a specific model without standing up a server.
//
// Usage:
//
//	go run ./example                                # default endpoint
//	go run ./example http://host:8000               # explicit vLLM endpoint
//	go run ./example -m meta-llama/Meta-Llama-3-8B  # single HF model
//	VLLM_API_KEY=... HF_TOKEN=... go run ./example
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	modelmeta "github.com/algonode/model-meta"
)

const defaultEndpoint = "http://localhost:8000"

func main() {
	var modelID string
	flag.StringVar(&modelID, "m", "", "Resolve a single HuggingFace model id (skips vLLM enumeration)")
	flag.Parse()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	e := &modelmeta.Enumerator{
		APIKey:  os.Getenv("VLLM_API_KEY"),
		HFToken: os.Getenv("HF_TOKEN"),
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")

	if modelID != "" {
		m, err := e.Resolve(ctx, modelID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "resolve %s: %v\n", modelID, err)
			os.Exit(1)
		}
		if err := enc.Encode(m); err != nil {
			fmt.Fprintf(os.Stderr, "encode: %v\n", err)
			os.Exit(1)
		}
		return
	}

	endpoint := defaultEndpoint
	if args := flag.Args(); len(args) > 0 && args[0] != "" {
		endpoint = args[0]
	}
	e.EndpointURL = endpoint

	models, err := e.Enumerate(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "enumerate %s: %v\n", endpoint, err)
		os.Exit(1)
	}
	if err := enc.Encode(models); err != nil {
		fmt.Fprintf(os.Stderr, "encode: %v\n", err)
		os.Exit(1)
	}
}
