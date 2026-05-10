// Package modelmeta enumerates models served by an OpenAI-compatible endpoint
// (typically vLLM) and resolves their feature set and lineage from the
// HuggingFace Hub.
package modelmeta

// Model is a deduplicated entry for one underlying set of weights served by an
// endpoint. Aliases collects every endpoint-exposed ID that resolves to the
// same Root.
type Model struct {
	// Root is the canonical model identifier as reported by the endpoint
	// (the `root` field of the OpenAI /v1/models response). It is the
	// HuggingFace path when the endpoint is configured with a Hub model.
	Root string `json:"root"`

	// Aliases lists the alternate IDs the endpoint accepts for this model,
	// excluding Root itself. They are sorted lexicographically.
	Aliases []string `json:"aliases,omitempty"`

	// MaxModelLen is the context window reported by the endpoint when
	// available, otherwise the value resolved from HuggingFace config.
	MaxModelLen int `json:"max_model_len,omitempty"`

	// OwnedBy mirrors the OpenAI field; usually the org name.
	OwnedBy string `json:"owned_by,omitempty"`

	// Features describes the capabilities resolved for this model.
	Features Features `json:"features"`

	// Flags is a small set of booleans summarizing the resolution result.
	// Always present — useful for quick filtering without walking the
	// full Model.
	Flags Flags `json:"flags"`

	// Lineage is the chain of base models, ordered from immediate parent to
	// the oldest known ancestor. Empty when no base_model is declared.
	Lineage []string `json:"lineage,omitempty"`

	// Tags groups the various tag sets associated with this model
	// (currently HuggingFace's verbatim list plus the compliance
	// watchlist matches). Nil when neither set has any entry.
	Tags *Tags `json:"tags,omitempty"`

	// License carries the model's declared license, resolved from
	// cardData.license (with fallback to a `license:*` HF tag). Nil when
	// HF resolution failed, was skipped, or no license was declared.
	License *License `json:"license,omitempty"`

	// Foundation is the resolved metadata for the top-most ancestor we
	// could identify — either the last entry in Lineage (the deepest
	// declared base_model) when the model declares a chain, or a
	// search-guessed parent when the model itself is unknown to the
	// Hub. Resolution is non-recursive: Foundation.Foundation is
	// always nil. Disabled via Enumerator.SkipGuessParent.
	Foundation *Model `json:"foundation,omitempty"`
}

// Flags summarizes a Model's resolution result with a small set of
// booleans that are cheap to filter on. Always present in the JSON.
type Flags struct {
	// Compliant is true when the compliance tag list (Tags.Compliance)
	// has at least one entry — i.e. the model's id or one of its
	// aliases matched the curated watchlist (Uncensored, Dolphin, RP,
	// ...). The name follows the field's literal definition; treat it
	// as a "matched the compliance watchlist" flag rather than as a
	// policy-clean indicator.
	Compliant bool `json:"compliant"`

	// HuggingFace is true when the model resolved successfully against
	// the HuggingFace Hub (the /api/models/{id} request returned 2xx).
	// Always false when SkipHF is set.
	HuggingFace bool `json:"huggingface"`

	// Lineage is true when at least one ancestor was resolved via
	// cardData.base_model.
	Lineage bool `json:"lineage"`

	// Quantized is true when Features.Quantization is set to anything
	// other than a native float dtype (bf16, fp16, fp32). Empty
	// quantization (no signal) is reported as false.
	Quantized bool `json:"quantized"`
}

// Tags collects the named tag sets attached to a Model.
type Tags struct {
	// HuggingFace is the verbatim tag list returned by the Hub for this
	// model (info.tags merged with cardData.tags, deduplicated). Set
	// only when HF resolution succeeded.
	HuggingFace []string `json:"huggingface,omitempty"`

	// Compliance lists labels from a curated watchlist (e.g. "Uncensored",
	// "Dolphin", "RP") that match the model's root id or any of its
	// aliases. Useful for routing/policy decisions.
	Compliance []string `json:"compliance,omitempty"`
}

// License describes the license declared on a HuggingFace model. ID is the
// canonical short identifier (e.g. "apache-2.0", "mit", "llama3.1", "other").
// Name and Link are populated for "other"-style licenses where the model
// author supplied a custom title and URL on the model card.
type License struct {
	ID   string `json:"id,omitempty"`
	Name string `json:"name,omitempty"`
	Link string `json:"link,omitempty"`
}

// Features captures the capabilities and properties resolved for a model.
// Boolean fields are best-effort: false means "not detected", not "definitely
// unsupported".
type Features struct {
	TextGeneration bool     `json:"text_generation,omitempty"`
	Embedding      bool     `json:"embedding,omitempty"`
	Vision         bool     `json:"vision,omitempty"`
	Audio          bool     `json:"audio,omitempty"`
	ToolUse        bool     `json:"tool_use,omitempty"`
	Reasoning      bool     `json:"reasoning,omitempty"`
	Code           bool     `json:"code,omitempty"`
	Quantization  string   `json:"quantization,omitempty"`
	Architectures []string `json:"architectures,omitempty"`
	Pipeline      string   `json:"pipeline,omitempty"`
}
