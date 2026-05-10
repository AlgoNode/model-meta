# model-meta

Go library that enumerates models served by a vLLM (or any OpenAI-compatible)
endpoint and resolves rich metadata for each one from the HuggingFace Hub:
features, license, lineage, weight quantization, and tags. Also resolves a
single HuggingFace model id directly when you don't need an endpoint.

## Install

```sh
go get github.com/algonode/model-meta
```

Requires Go 1.24+.

## Quick start

```go
import modelmeta "github.com/algonode/model-meta"

e := &modelmeta.Enumerator{
    EndpointURL: "http://localhost:8000", // any OpenAI-compatible /v1
    APIKey:      os.Getenv("VLLM_API_KEY"),
    HFToken:     os.Getenv("HF_TOKEN"),
}

// Enumerate everything the endpoint serves.
models, err := e.Enumerate(ctx)

// Or resolve one HuggingFace model without contacting the endpoint.
m, err := e.Resolve(ctx, "meta-llama/Meta-Llama-3-8B")
```

## CLI

A `resolve-model` binary ships with the module — install with:

```sh
go install github.com/algonode/model-meta/resolve-model@latest
```

It mirrors both library modes:

```sh
resolve-model                                   # default localhost:8000
resolve-model http://host:8000                  # explicit vLLM endpoint
resolve-model -m meta-llama/Meta-Llama-3-8B     # single HF model, no endpoint
VLLM_API_KEY=... HF_TOKEN=... resolve-model
```

## What you get back

Each `Model` aggregates one set of weights and every endpoint-exposed alias
that resolves to it:

```json
{
  "root": "meta-llama/Meta-Llama-3-8B-Instruct",
  "aliases": ["default", "llama3"],
  "max_model_len": 8192,
  "owned_by": "meta-llama",
  "features": {
    "text_generation": true,
    "tool_use": true,
    "quantization": "bf16",
    "architectures": ["LlamaForCausalLM"],
    "pipeline": "text-generation"
  },
  "lineage": ["meta-llama/Meta-Llama-3-8B"],
  "tags": {
    "huggingface": ["transformers", "tool-use", "license:llama3"],
    "compliance":  []
  },
  "license": { "id": "llama3" },
  "flags": {
    "compliant":   false,
    "huggingface": true,
    "lineage":     true,
    "quantized":   false
  }
}
```

## How resolution works

### Aggregation

`/v1/models` entries are grouped by their `root` field. Anything whose `id`
differs from `root` becomes an alias. The result is sorted by `Root`.

### Quantization

Detected with the following priority — the first source that fires wins:

1. `quantization_config.quant_method` from the API summary (NVFP4 recognized
   from either `quant_method == "nvfp4"` or a `format` containing `"nvfp4"`).
2. **Compressed-tensors refine.** When step 1 yields `"compressed-tensors"`
   the API summary has truncated the real format. The library does a
   follow-up `GET /{id}/resolve/main/config.json` and inspects both the
   top-level `quantization_config.format` and per-group
   `config_groups.*.format` for an NVFP4 marker. Cached, with 404
   short-circuit, so non-affected models pay no extra HTTP cost.
3. **GGUF filename.** If the repo is a GGUF repo (library_name, tag, or
   `.gguf` siblings), the canonical file is picked — id pin first
   (`Foo-GGUF-Q5_K_M`), then a llama.cpp preference order
   (Q4_K_M > Q5_K_M > Q5_K_S > …), then size, then lexicographic. Its
   tier is parsed from the filename. Multi-part shards
   (`-NNNNN-of-NNNNN.gguf`) are collapsed to part 1.
4. `torch_dtype` mapping: `bfloat16 → bf16`, `float16/half → fp16`,
   `float32 → fp32`, `float8* → fp8`, `float4* → fp4`.
5. Vendor suffix on the vLLM id: `-AWQ`, `-GPTQ`, `-FP8`, `-NVFP4`, …
6. **GGUF tier suffix** on the id: `Q4_K_M`, `IQ3_XXS`, `BF16`, … This is
   the practical fallback for llama.cpp endpoints, whose `id` is typically
   the local filename rather than an HF path.

### Features

Pipeline tag, HF tags, and architecture names feed boolean flags:
`TextGeneration`, `Embedding`, `Vision`, `Audio`, `ToolUse`, `Reasoning`,
`Code`. Best-effort — `false` means "not detected", not "definitely
unsupported".

### License

`License.ID` comes from `cardData.license`, with a `license:<id>` HF tag as
fallback. `Name` and `Link` are filled when the model card declares an
`other`-style license with a custom title and URL. Set only when HF
resolution succeeded.

### Tags

`Tags.HuggingFace` is the deduped union of `info.tags` and `cardData.tags`.
Set only when HF resolution succeeded.

`Tags.Compliance` is matched against a curated regex watchlist
(`Uncensored`, `Abliterated`, `Dolphin`, `Hermes`, `OpenHermes`, `NSFW`,
`RP`, `ERP`, `Wizard`, …). `RP`/`ERP` are word-boundary anchored so
identifiers like `RPCS3`/`ERPNext` don't trigger. Useful for routing or
policy decisions.

`Model.Tags` is omitted from the JSON entirely when both lists are empty.

### Lineage

Walks `cardData.base_model` outward, depth-capped (default 8) and
cycle-safe. `Lineage[0]` is the immediate parent; the last element is
the deepest declared ancestor.

### Foundation

`Model.Foundation` carries a fully-resolved (non-recursive) view of the
top ancestor we could identify:

- When `Lineage` is non-empty, `Foundation` resolves the **last** entry
  (the deepest declared `base_model`).
- When direct HF lookup for the model itself fails (e.g. llama.cpp ids
  like `qwen2.5-7b-instruct-q4_k_m`), the library searches HF
  (`/api/models?search=…&sort=downloads`), normalizes the candidate's
  name, and accepts the top hit only if at least 60% of normalized
  query tokens overlap with the candidate id. Disable the search
  fallback with `Enumerator.SkipGuessParent`.

The non-recursion guarantee: `Foundation.Foundation` is always nil.
Foundation entries do walk their own `Lineage` though, so you can still
see the foundation's declared ancestors.

### MaxModelLen

Endpoint-reported value wins (it reflects the *configured* serving limit,
e.g. `--max-model-len 32768` on a 128k-context model). If the endpoint
omits it, falls back to `config.max_position_embeddings` from HF.

## Configuration

`Enumerator` fields:

| Field             | Purpose                                                    |
|-------------------|------------------------------------------------------------|
| `EndpointURL`     | OpenAI-compatible base; `/v1` and `/v1/models` both accepted. |
| `APIKey`          | Bearer token for the endpoint.                             |
| `HFBaseURL`       | Override the Hub root (tests / private mirrors).           |
| `HFToken`         | Bearer token for HuggingFace.                              |
| `HTTPClient`      | Custom client for endpoint requests (default: 30s timeout).|
| `HFHTTPClient`    | Custom client for HF requests (default: 30s timeout).      |
| `MaxLineageDepth` | Cap on `base_model` traversal (default 8).                 |
| `SkipHF`          | Disable HF resolution; only id-derived signals are used.   |
| `SkipGuessParent` | Disable the HF search fallback used to populate Foundation when direct resolution fails. Lineage-tip Foundation is unaffected. |

## Development

```sh
make test       # unit tests
make test-race  # race detector
make cover      # coverage report
make all        # fmt + vet + test
```

The repo follows a tight convention: every functional change ships in its
own commit with a short rationale, and every new or modified feature has a
test.
