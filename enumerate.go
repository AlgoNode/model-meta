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

// resolveModel produces a Model for one rootGroup. When hf is nil (SkipHF
// or constructor failure) only id-derived signals are used. Shared by
// Enumerate and Resolve so the two entry points stay in lockstep.
func (e *Enumerator) resolveModel(ctx context.Context, hf *hfClient, g rootGroup) Model {
	m := Model{
		Root:        g.root,
		Aliases:     g.aliases,
		MaxModelLen: g.maxModelLen,
		OwnedBy:     g.ownedBy,
	}
	var hfTags []string
	if hf != nil {
		info, ferr := hf.fetch(ctx, g.root)
		if ferr == nil {
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
	return m
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
