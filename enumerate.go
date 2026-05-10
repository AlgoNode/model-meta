package modelmeta

import (
	"context"
	"errors"
	"net/http"
	"sort"
	"strings"
	"time"
)

// Enumerator resolves the catalogue of a single endpoint.
//
// Zero-value usage requires only EndpointURL. HFBaseURL defaults to the public
// Hub; HTTPClient and HFHTTPClient default to http.DefaultClient with a small
// per-request timeout when none is configured.
type Enumerator struct {
	// EndpointURL is the OpenAI-compatible base URL. /v1 or /v1/models
	// suffixes are accepted; a bare host is also fine.
	EndpointURL string

	// APIKey is the optional bearer token for the endpoint.
	APIKey string

	// HFBaseURL overrides the HuggingFace Hub root (used by tests).
	HFBaseURL string

	// HFToken is the optional HuggingFace access token.
	HFToken string

	// HTTPClient is used for endpoint requests. nil falls back to a client
	// with a 30s timeout.
	HTTPClient *http.Client

	// HFHTTPClient is used for HuggingFace requests. nil falls back to a
	// client with a 30s timeout.
	HFHTTPClient *http.Client

	// MaxLineageDepth caps base_model traversal. 0 picks a sensible default.
	MaxLineageDepth int

	// SkipHF disables HuggingFace resolution; only vLLM-reported metadata
	// is used. Useful for offline or air-gapped setups.
	SkipHF bool

	// SkipGuessParent disables the HF search-and-guess fallback used to
	// populate Ancestor when direct HF resolution fails (e.g.
	// llama.cpp endpoints whose id is the local filename). Default
	// behavior is to guess. The lineage-tip path to Ancestor is
	// always taken when applicable; this flag only gates the search
	// fallback.
	SkipGuessParent bool
}

// Enumerate calls /v1/models on the endpoint, resolves each unique root model
// against HuggingFace, and returns the deduplicated catalogue. Aliases are
// gathered from every endpoint id whose `root` matches the canonical id.
//
// Models are returned sorted by Root.
func (e *Enumerator) Enumerate(ctx context.Context) ([]Model, error) {
	if e == nil {
		return nil, errors.New("modelmeta: nil Enumerator")
	}
	httpc := e.HTTPClient
	if httpc == nil {
		httpc = &http.Client{Timeout: 30 * time.Second}
	}

	raw, err := fetchVLLMModels(ctx, httpc, e.EndpointURL, e.APIKey)
	if err != nil {
		return nil, err
	}

	groups := groupByRoot(raw)

	var hf *hfClient
	if !e.SkipHF {
		hfHTTP := e.HFHTTPClient
		if hfHTTP == nil {
			hfHTTP = &http.Client{Timeout: 30 * time.Second}
		}
		hf = newHFClient(e.HFBaseURL, e.HFToken, hfHTTP)
	}

	out := make([]Model, 0, len(groups))
	for _, g := range groups {
		out = append(out, e.resolveModel(ctx, hf, g))
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Root < out[j].Root })
	return out, nil
}

// Resolve fetches metadata for a single HuggingFace model id without
// consulting any vLLM endpoint. The returned Model has empty Aliases (no
// endpoint catalogue exists to derive them from) and is otherwise filled
// the same way Enumerate fills its entries: features, license, lineage,
// quant detection (incl. compressed-tensors refinement), and the tags
// block. SkipHF is honored — with it set, only id-derived signals are
// used.
func (e *Enumerator) Resolve(ctx context.Context, modelID string) (*Model, error) {
	if e == nil {
		return nil, errors.New("modelmeta: nil Enumerator")
	}
	if strings.TrimSpace(modelID) == "" {
		return nil, errors.New("modelmeta: empty model id")
	}
	var hf *hfClient
	if !e.SkipHF {
		hfHTTP := e.HFHTTPClient
		if hfHTTP == nil {
			hfHTTP = &http.Client{Timeout: 30 * time.Second}
		}
		hf = newHFClient(e.HFBaseURL, e.HFToken, hfHTTP)
	}
	m := e.resolveModel(ctx, hf, rootGroup{root: modelID})
	return &m, nil
}

// resolveModel produces a Model for one rootGroup, including a
// non-recursive Ancestor when one can be identified. Shared by
// Enumerate and Resolve.
func (e *Enumerator) resolveModel(ctx context.Context, hf *hfClient, g rootGroup) Model {
	m := e.resolveSingle(ctx, hf, g)

	ancestorID := ""
	switch {
	case len(m.Lineage) > 0:
		// Top of lineage = deepest declared base_model.
		ancestorID = m.Lineage[len(m.Lineage)-1]
	case hf != nil && !e.SkipGuessParent && (!m.Flags.HuggingFace || m.Flags.Quantized):
		// Either: direct HF lookup failed (likely a llama.cpp local
		// id), or the model is on HF but declared no base_model AND
		// is quantized — a strong signal that it's a derivative whose
		// upstream the author didn't fill in. Native-dtype models
		// without lineage are assumed to be true bases and not
		// searched, to avoid pointing them at sibling derivatives.
		ancestorID = hf.guessParent(ctx, g.root)
	}
	if ancestorID != "" && ancestorID != g.root {
		anc := e.resolveSingle(ctx, hf, rootGroup{root: ancestorID})
		// Enforce non-recursion: an Ancestor never carries its own
		// Ancestor, even if resolveSingle ever started populating it.
		anc.Ancestor = nil
		m.Ancestor = &anc
	}
	return m
}

// resolveSingle fills a Model from a single rootGroup without computing
// the Ancestor. It is the per-model body shared between the top-level
// resolution and the bounded ancestor resolution.
func (e *Enumerator) resolveSingle(ctx context.Context, hf *hfClient, g rootGroup) Model {
	m := Model{
		Root:        g.root,
		Aliases:     g.aliases,
		MaxModelLen: g.maxModelLen,
		OwnedBy:     g.ownedBy,
	}
	var hfTags []string
	var hfResolved bool
	if hf != nil {
		info, ferr := hf.fetch(ctx, g.root)
		if ferr == nil {
			hfResolved = true
			m.Features, hfTags = extractFeatures(info, g.root)
			m.Lineage = hf.resolveLineage(ctx, g.root, e.MaxLineageDepth)
			m.License = extractLicense(info.CardData, hfTags)
			if m.MaxModelLen == 0 && info.Config.MaxPositionEmbed > 0 {
				m.MaxModelLen = info.Config.MaxPositionEmbed
			}
			// The API summary truncates compressed-tensors metadata;
			// fetch the raw config.json to recover the real format.
			if m.Features.Quantization == "compressed-tensors" {
				if body, err := hf.fetchRawConfig(ctx, g.root); err == nil {
					if refined := refineCompressedQuant(body); refined != "" {
						m.Features.Quantization = refined
					}
				}
			}
		} else {
			// Without HF, still derive what we can from the id alone.
			m.Features, _ = extractFeatures(nil, g.root)
		}
	} else {
		m.Features, _ = extractFeatures(nil, g.root)
	}
	complianceTags := matchComplianceTags(g.root, g.aliases)
	if len(hfTags) > 0 || len(complianceTags) > 0 {
		m.Tags = &Tags{HuggingFace: hfTags, Compliance: complianceTags}
	}
	m.Flags = Flags{
		Compliant:   len(complianceTags) > 0,
		HuggingFace: hfResolved,
		Lineage:     len(m.Lineage) > 0,
		Quantized:   isQuantized(m.Features.Quantization),
	}
	return m
}

// isQuantized returns true for any quantization label other than the
// native float dtypes (bf16, fp16, fp32). An empty label (no signal)
// is treated as not quantized.
func isQuantized(q string) bool {
	switch q {
	case "", "bf16", "fp16", "fp32":
		return false
	}
	return true
}

// rootGroup collects vLLM entries that share the same root id.
type rootGroup struct {
	root        string
	aliases     []string
	maxModelLen int
	ownedBy     string
}

// groupByRoot deduplicates a /v1/models payload. When `root` is empty (some
// non-vLLM servers omit it) the id is treated as its own root. Aliases are
// sorted and exclude the root id itself.
func groupByRoot(raw []vllmModel) []rootGroup {
	byRoot := make(map[string]*rootGroup)
	order := make([]string, 0)
	for _, m := range raw {
		root := m.Root
		if root == "" {
			root = m.ID
		}
		g, ok := byRoot[root]
		if !ok {
			g = &rootGroup{root: root, ownedBy: m.OwnedBy, maxModelLen: m.MaxModelLen}
			byRoot[root] = g
			order = append(order, root)
		}
		if g.maxModelLen == 0 && m.MaxModelLen > 0 {
			g.maxModelLen = m.MaxModelLen
		}
		if g.ownedBy == "" && m.OwnedBy != "" {
			g.ownedBy = m.OwnedBy
		}
		if m.ID != root {
			g.aliases = append(g.aliases, m.ID)
		}
	}
	out := make([]rootGroup, 0, len(order))
	for _, r := range order {
		g := byRoot[r]
		g.aliases = dedupSort(g.aliases)
		out = append(out, *g)
	}
	return out
}

func dedupSort(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(in))
	out := make([]string, 0, len(in))
	for _, s := range in {
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}
