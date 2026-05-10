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

	// Lineage is the chain of base models, ordered from immediate parent to
	// the oldest known ancestor. Empty when no base_model is declared.
	Lineage []string `json:"lineage,omitempty"`

	// HFTags is the verbatim tag list HuggingFace returns for this model
	// (info.tags merged with cardData.tags, deduplicated). Set only when
	// the Hub resolution succeeds; absent on 404 or when SkipHF is true.
	HFTags []string `json:"hf_tags,omitempty"`
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

	// ComplianceTags lists labels from a curated watchlist (e.g. "Uncensored",
	// "Dolphin", "RP") that match the model's root id or any of its aliases.
	// Useful for routing/policy decisions; never surface as user-facing UI.
	ComplianceTags []string `json:"compliance_tags,omitempty"`
}
