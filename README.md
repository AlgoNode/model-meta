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
e := modelmeta.Enumerator{EndpointURL: "http://localhost:8000"}
models, err := e.Enumerate(ctx)
```

Each returned `Model` carries the canonical `Root` ID, any `Aliases` the endpoint
exposes for it, and a `Features` block with the capabilities the library could
infer (text generation, vision, tool use, embedding, quantization, etc.).

## Development

```sh
make test       # run unit tests
make test-race  # race detector
make cover      # coverage report
make all        # fmt + vet + test
```
