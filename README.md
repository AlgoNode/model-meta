# model-meta

Go library that enumerates models served by a vLLM (or any OpenAI-compatible) endpoint
and resolves their feature set and lineage from the HuggingFace Hub.

## What it does

1. Calls `GET /v1/models` on a target endpoint.
2. Resolves each model against `https://huggingface.co/api/models/{id}` for tags,
   architecture, context length, and `base_model` lineage.
3. Aggregates by `root`, collapsing aliases (e.g. short names that map to the same
   underlying weights) into a single `Model` entry.

## Quick start

```go
// Enumerate a vLLM endpoint
e := modelmeta.Enumerator{EndpointURL: "http://localhost:8000"}
models, err := e.Enumerate(ctx)

// Or resolve a single HuggingFace model without an endpoint
m, err := e.Resolve(ctx, "meta-llama/Meta-Llama-3-8B")
```

Each returned `Model` carries the canonical `Root` ID, any `Aliases` the endpoint
exposes for it, and a `Features` block with the capabilities the library could
infer (text generation, vision, tool use, embedding, quantization, etc.).

The example CLI mirrors both modes:

```sh
go run ./example                                # default endpoint
go run ./example http://host:8000               # explicit vLLM endpoint
go run ./example -m meta-llama/Meta-Llama-3-8B  # single HF model
```

## Development

```sh
make test       # run unit tests
make test-race  # race detector
make cover      # coverage report
make all        # fmt + vet + test
```
