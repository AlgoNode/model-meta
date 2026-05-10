package modelmeta

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"sync"
)

// hfBaseURL is the default HuggingFace Hub API root. Tests override it.
const hfBaseURL = "https://huggingface.co"

// errHFNotFound is returned when the Hub responds 404 for a model id.
var errHFNotFound = errors.New("huggingface model not found")

// hfModelInfo is the subset of /api/models/{id} we consume.
type hfModelInfo struct {
	ID          string      `json:"id"`
	ModelID     string      `json:"modelId"`
	Pipeline    string      `json:"pipeline_tag"`
	LibraryName string      `json:"library_name"`
	Tags        []string    `json:"tags"`
	Config      hfConfig    `json:"config"`
	CardData    hfCardData  `json:"cardData"`
	Siblings    []hfSibling `json:"siblings"`
}

// hfSibling is one entry in the HuggingFace `siblings` array describing a
// file in the model repo. Size may be zero when the basic API response
// omits it.
type hfSibling struct {
	Filename string `json:"rfilename"`
	Size     int64  `json:"size"`
}

type hfConfig struct {
	Architectures      []string      `json:"architectures"`
	ModelType          string        `json:"model_type"`
	MaxPositionEmbed   int           `json:"max_position_embeddings"`
	TorchDtype         string        `json:"torch_dtype"`
	QuantizationConfig hfQuantConfig `json:"quantization_config"`
}

type hfQuantConfig struct {
	QuantMethod string `json:"quant_method"`
	Format      string `json:"format"`
	Bits        int    `json:"bits"`
}

// hfCardData carries optional fields from the model card YAML. base_model can
// be either a string or an array of strings, hence the custom unmarshal.
type hfCardData struct {
	BaseModel   hfStringList `json:"base_model"`
	Pipeline    string       `json:"pipeline_tag"`
	Tags        []string     `json:"tags"`
	License     string       `json:"license"`
	LicenseName string       `json:"license_name"`
	LicenseLink string       `json:"license_link"`
}

// hfStringList accepts either a JSON string or a JSON array of strings.
type hfStringList []string

func (s *hfStringList) UnmarshalJSON(data []byte) error {
	data = trimSpace(data)
	if len(data) == 0 || string(data) == "null" {
		return nil
	}
	if data[0] == '[' {
		var arr []string
		if err := json.Unmarshal(data, &arr); err != nil {
			return err
		}
		*s = arr
		return nil
	}
	var one string
	if err := json.Unmarshal(data, &one); err != nil {
		return err
	}
	if one != "" {
		*s = []string{one}
	}
	return nil
}

// escapeModelID escapes each path segment of a HuggingFace id while preserving
// the "/" separator between owner and model name.
func escapeModelID(id string) string {
	parts := strings.Split(id, "/")
	for i, p := range parts {
		parts[i] = url.PathEscape(p)
	}
	return strings.Join(parts, "/")
}

func trimSpace(b []byte) []byte {
	for len(b) > 0 && (b[0] == ' ' || b[0] == '\t' || b[0] == '\n' || b[0] == '\r') {
		b = b[1:]
	}
	for len(b) > 0 {
		c := b[len(b)-1]
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			break
		}
		b = b[:len(b)-1]
	}
	return b
}

// hfClient fetches and caches model metadata from a HuggingFace-compatible API.
type hfClient struct {
	baseURL string
	token   string
	http    *http.Client

	mu       sync.Mutex
	cache    map[string]*hfModelInfo
	miss     map[string]struct{}
	rawCache map[string][]byte
	rawMiss  map[string]struct{}
}

func newHFClient(baseURL, token string, httpc *http.Client) *hfClient {
	if baseURL == "" {
		baseURL = hfBaseURL
	}
	if httpc == nil {
		httpc = http.DefaultClient
	}
	return &hfClient{
		baseURL:  strings.TrimRight(baseURL, "/"),
		token:    token,
		http:     httpc,
		cache:    make(map[string]*hfModelInfo),
		miss:     make(map[string]struct{}),
		rawCache: make(map[string][]byte),
		rawMiss:  make(map[string]struct{}),
	}
}

// fetch returns the model info for id, with an in-memory cache for both hits
// and 404s so repeated lookups during one Enumerate stay cheap.
func (c *hfClient) fetch(ctx context.Context, id string) (*hfModelInfo, error) {
	if id == "" {
		return nil, errHFNotFound
	}
	c.mu.Lock()
	if v, ok := c.cache[id]; ok {
		c.mu.Unlock()
		return v, nil
	}
	if _, ok := c.miss[id]; ok {
		c.mu.Unlock()
		return nil, errHFNotFound
	}
	c.mu.Unlock()

	u := c.baseURL + "/api/models/" + escapeModelID(id)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	req.Header.Set("Accept", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("huggingface fetch %s: %w", id, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		c.mu.Lock()
		c.miss[id] = struct{}{}
		c.mu.Unlock()
		return nil, errHFNotFound
	}
	if resp.StatusCode/100 != 2 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<14))
		return nil, fmt.Errorf("huggingface fetch %s: status %d: %s", id, resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var info hfModelInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("huggingface decode %s: %w", id, err)
	}
	if info.ID == "" {
		info.ID = info.ModelID
	}
	if info.ID == "" {
		info.ID = id
	}
	c.mu.Lock()
	c.cache[id] = &info
	c.mu.Unlock()
	return &info, nil
}

// fetchRawConfig retrieves the model's raw config.json (the file itself,
// not the API summary). Used to recover detail the API drops — most
// notably the nested compressed-tensors format. Cached, with 404
// short-circuit, like fetch.
func (c *hfClient) fetchRawConfig(ctx context.Context, id string) ([]byte, error) {
	if id == "" {
		return nil, errHFNotFound
	}
	c.mu.Lock()
	if v, ok := c.rawCache[id]; ok {
		c.mu.Unlock()
		return v, nil
	}
	if _, ok := c.rawMiss[id]; ok {
		c.mu.Unlock()
		return nil, errHFNotFound
	}
	c.mu.Unlock()

	u := c.baseURL + "/" + escapeModelID(id) + "/resolve/main/config.json"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("huggingface fetch raw config %s: %w", id, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		c.mu.Lock()
		c.rawMiss[id] = struct{}{}
		c.mu.Unlock()
		return nil, errHFNotFound
	}
	if resp.StatusCode/100 != 2 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<14))
		return nil, fmt.Errorf("huggingface raw config %s: status %d: %s", id, resp.StatusCode, strings.TrimSpace(string(body)))
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("huggingface read raw config %s: %w", id, err)
	}
	c.mu.Lock()
	c.rawCache[id] = body
	c.mu.Unlock()
	return body, nil
}

// refineCompressedQuant inspects the raw config.json for a model whose
// API-reported quant_method is "compressed-tensors" and returns a more
// specific label (currently NVFP4) when one is implied by the
// quantization_config. Compressed-tensors stores the format either at
// the top level or, per layer group, under config_groups.*.format —
// we check both. Returns "" when no refinement applies.
func refineCompressedQuant(body []byte) string {
	var raw struct {
		QuantizationConfig struct {
			Format       string                     `json:"format"`
			ConfigGroups map[string]json.RawMessage `json:"config_groups"`
		} `json:"quantization_config"`
	}
	if err := json.Unmarshal(body, &raw); err != nil {
		return ""
	}
	if strings.Contains(strings.ToLower(raw.QuantizationConfig.Format), "nvfp4") {
		return "nvfp4"
	}
	for _, g := range raw.QuantizationConfig.ConfigGroups {
		var group struct {
			Format string `json:"format"`
		}
		if json.Unmarshal(g, &group) == nil {
			if strings.Contains(strings.ToLower(group.Format), "nvfp4") {
				return "nvfp4"
			}
		}
	}
	return ""
}

// hfModelSearchResult is the slim subset of a /api/models?search=... entry
// we consume. Tags are read so we can drop quantized candidates without an
// extra per-candidate fetch.
type hfModelSearchResult struct {
	ID      string   `json:"id"`
	ModelID string   `json:"modelId"`
	Tags    []string `json:"tags"`
}

// quantTagWords flags HF tags that signal a model is quantized. Used to
// drop quantized candidates from guess-parent search results.
var quantTagWords = map[string]struct{}{
	"gguf":               {},
	"compressed-tensors": {},
	"bitsandbytes":       {},
	"awq":                {},
	"gptq":               {},
	"4-bit":              {},
	"8-bit":              {},
	"nf4":                {},
	"exl2":               {},
	"exllamav2":          {},
}

// tagsLookQuantized reports whether any tag indicates a quantized model.
// Comparison is lowercase-insensitive against quantTagWords.
func tagsLookQuantized(tags []string) bool {
	for _, t := range tags {
		if _, ok := quantTagWords[strings.ToLower(t)]; ok {
			return true
		}
	}
	return false
}

// forkMarkerWords are tokens that strongly indicate a model is a derivative
// rather than an upstream base. Stripping them from both the search query
// and candidate normalization keeps HF's ranking from biasing toward
// other forks ("uncensored" matching "ultra-uncensored-heretic" etc.).
var forkMarkerWords = map[string]struct{}{
	// Compliance watchlist words.
	"uncensored":  {},
	"abliterated": {},
	"ablit":       {},
	"toxic":       {},
	"dolphin":     {},
	"lumimaid":    {},
	"capybara":    {},
	"hermes":      {},
	"openhermes":  {},
	"nsfw":        {},
	"rp":          {},
	"erp":         {},
	"unfiltered":  {},
	"heretic":     {},
	"aggressive":  {},
	"wizard":      {},
	// Merge / training method markers.
	"merge":  {},
	"merged": {},
	"slerp":  {},
	"dare":   {},
	"ties":   {},
	"lora":   {},
	"dpo":    {},
	"sft":    {},
	"rlhf":   {},
	// Common intensity / fork suffixes.
	"ultra": {},
}

// baseModelFromTags extracts the first id referenced by a `base_model:[<rel>:]<id>`
// HF tag. Returns "" when no such tag is present. Examples:
//
//	"base_model:google/gemma-4-26b-a4b-it"               -> "google/gemma-4-26b-a4b-it"
//	"base_model:finetune:google/gemma-4-26b-a4b-it"      -> "google/gemma-4-26b-a4b-it"
//	"base_model:quantized:foo/bar"                        -> "foo/bar"
func baseModelFromTags(tags []string) string {
	for _, t := range tags {
		rest, ok := strings.CutPrefix(t, "base_model:")
		if !ok {
			continue
		}
		rest = strings.TrimSpace(rest)
		if rest == "" {
			continue
		}
		// `rest` is either "<id>" or "<relationship>:<id>". Split on
		// the first ":" only when the prefix has no "/" (relationship
		// names never contain a slash, ids always do).
		if idx := strings.Index(rest, ":"); idx >= 0 && !strings.Contains(rest[:idx], "/") {
			rest = rest[idx+1:]
		}
		if rest != "" && strings.Contains(rest, "/") {
			return rest
		}
	}
	return ""
}

// parentOfInfo returns the immediate parent id declared by a model card
// or by HF's own `base_model:*` tags. CardData is preferred (authors set
// it directly); the tag fallback covers cases where the API summary has
// dropped cardData but HF still surfaces the relationship.
func parentOfInfo(info *hfModelInfo) string {
	if info == nil {
		return ""
	}
	if len(info.CardData.BaseModel) > 0 {
		if v := strings.TrimSpace(info.CardData.BaseModel[0]); v != "" {
			return v
		}
	}
	return baseModelFromTags(info.Tags)
}

// searchModels queries the HF Hub search endpoint and returns up to
// `limit` results sorted by downloads (descending). Used for the
// guessParent fallback when a direct id resolution fails.
func (c *hfClient) searchModels(ctx context.Context, query string, limit int) ([]hfModelSearchResult, error) {
	if strings.TrimSpace(query) == "" {
		return nil, nil
	}
	if limit <= 0 {
		limit = 5
	}
	params := url.Values{
		"search":    {query},
		"limit":     {strconv.Itoa(limit)},
		"sort":      {"downloads"},
		"direction": {"-1"},
	}
	u := c.baseURL + "/api/models?" + params.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	req.Header.Set("Accept", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("huggingface search: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<14))
		return nil, fmt.Errorf("huggingface search: status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var out []hfModelSearchResult
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("huggingface search decode: %w", err)
	}
	return out, nil
}

// guessParentMinSimilarity is the forward-direction (query ⊆ candidate)
// gate: at least this fraction of normalized query tokens must appear
// in the candidate. Loose enough to tolerate minor noise the normalizer
// missed, strict enough to drop obvious mismatches.
const guessParentMinSimilarity = 0.6

// guessParentMinReverseSimilarity is the reverse-direction
// (candidate ⊆ query) gate: at least this fraction of the candidate's
// own tokens must appear in the query. A true parent's name is a
// (near-)subset of the model's name, so siblings that introduce extra
// tokens beyond the query lose. Combined with fork-marker stripping,
// this keeps unrelated forks out even when HF's ranker prefers them.
const guessParentMinReverseSimilarity = 0.8

// guessParent searches HF for a likely parent of `id` and returns the
// best candidate's HF id, or "" when nothing crosses the similarity
// gate. The candidate must (a) differ from `id`, (b) share at least
// guessParentMinSimilarity fraction of normalized tokens with the query,
// and (c) come from a query of at least two tokens — single-token
// queries (e.g. "default") are too generic.
func (c *hfClient) guessParent(ctx context.Context, id string) string {
	query := normalizeForSearch(id)
	if len(strings.Fields(query)) < 2 {
		return ""
	}
	results, err := c.searchModels(ctx, query, 5)
	if err != nil || len(results) == 0 {
		return ""
	}
	for _, r := range results {
		cand := r.ID
		if cand == "" {
			cand = r.ModelID
		}
		if cand == "" || cand == id {
			continue
		}
		// Drop quantized candidates: a quant of a quant isn't an
		// ancestor we want to surface.
		if tagsLookQuantized(r.Tags) {
			continue
		}
		candNorm := normalizeForSearch(cand)
		if similarityScore(query, candNorm) < guessParentMinSimilarity {
			continue
		}
		if similarityScore(candNorm, query) < guessParentMinReverseSimilarity {
			// Candidate introduces too many tokens not in the
			// query — almost certainly a sibling fork rather
			// than an upstream parent.
			continue
		}
		return cand
	}
	return ""
}

// normalizeForSearch canonicalizes a model id into a space-separated
// keyword string suitable for the HF search endpoint. Drops the org/
// path prefix, strips quant suffixes (vendor and GGUF tier) so they
// don't poison the search, replaces separators with spaces, lowercases
// the result, and drops fork-marker words (uncensored, abliterated,
// dolphin, merge, …) so HF doesn't rank derivatives ahead of the
// actual upstream.
func normalizeForSearch(id string) string {
	if idx := strings.LastIndex(id, "/"); idx >= 0 {
		id = id[idx+1:]
	}
	id = quantPattern.ReplaceAllString(id, " ")
	id = ggufIDQuantPattern.ReplaceAllString(id, " ")
	id = strings.NewReplacer("-", " ", "_", " ", ".", " ", "/", " ").Replace(id)
	id = strings.ToLower(id)
	fields := strings.Fields(id)
	out := fields[:0]
	for _, t := range fields {
		if _, skip := forkMarkerWords[t]; skip {
			continue
		}
		out = append(out, t)
	}
	return strings.Join(out, " ")
}

// similarityScore returns the fraction of `a` tokens that also appear
// in `b`. Both inputs are expected to be normalized (lowercase,
// space-separated). Returns 0 for empty `a`.
func similarityScore(a, b string) float64 {
	aTokens := strings.Fields(a)
	if len(aTokens) == 0 {
		return 0
	}
	bSet := make(map[string]struct{}, len(strings.Fields(b)))
	for _, t := range strings.Fields(b) {
		bSet[t] = struct{}{}
	}
	matched := 0
	for _, t := range aTokens {
		if _, ok := bSet[t]; ok {
			matched++
		}
	}
	return float64(matched) / float64(len(aTokens))
}

// resolveLineage walks base_model links from the given id outward, stopping
// at the first model not found on the Hub or a previously visited node. The
// starting id itself is not included in the returned slice. Each step
// reads the parent via parentOfInfo so we honor both cardData.base_model
// and the authoritative `base_model:*` HF tags.
func (c *hfClient) resolveLineage(ctx context.Context, id string, maxDepth int) []string {
	if maxDepth <= 0 {
		maxDepth = 8
	}
	visited := map[string]struct{}{id: {}}
	var out []string
	cur := id
	for depth := 0; depth < maxDepth; depth++ {
		info, err := c.fetch(ctx, cur)
		if err != nil || info == nil {
			return out
		}
		next := parentOfInfo(info)
		if next == "" || next == cur {
			return out
		}
		if _, seen := visited[next]; seen {
			return out
		}
		visited[next] = struct{}{}
		out = append(out, next)
		cur = next
	}
	return out
}

// extractFeatures derives a Features value from HF metadata plus the original
// vLLM-reported id (which often carries quantization suffixes like "-AWQ").
// The second return value is the merged HuggingFace tag list, useful for
// surfacing as Model.HFTags; it is nil when info is nil.
func extractFeatures(info *hfModelInfo, vllmID string) (Features, []string) {
	var f Features
	var tags []string
	if info != nil {
		f.Pipeline = info.Pipeline
		if f.Pipeline == "" {
			f.Pipeline = info.CardData.Pipeline
		}
		f.Architectures = append([]string(nil), info.Config.Architectures...)
		tags = mergeTags(info.Tags, info.CardData.Tags)
		// Quant priority for HF-resolved models:
		//   1. explicit quantization_config (NVFP4 + quant_method)
		//   2. GGUF filename (when this is a GGUF repo)
		//   3. torch_dtype (bf16, fp16, fp32, fp8, fp4)
		// Falls through to quantFromName(vllmID) below as last resort.
		if q := quantFromQuantizationConfig(info.Config.QuantizationConfig); q != "" {
			f.Quantization = q
		} else if isGGUFRepo(info) {
			if file := pickCanonicalGGUF(info.Siblings, vllmID); file != "" {
				f.Quantization = quantFromGGUF(file)
			}
		}
		if f.Quantization == "" {
			f.Quantization = dtypeToLabel(info.Config.TorchDtype)
		}
	}
	if f.Quantization == "" {
		f.Quantization = quantFromName(vllmID)
	}
	applyFeatureFlags(&f, tags)
	return f, tags
}

func mergeTags(a, b []string) []string {
	if len(a) == 0 && len(b) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(a)+len(b))
	out := make([]string, 0, len(a)+len(b))
	for _, src := range [][]string{a, b} {
		for _, t := range src {
			t = strings.TrimSpace(t)
			if t == "" {
				continue
			}
			if _, ok := seen[t]; ok {
				continue
			}
			seen[t] = struct{}{}
			out = append(out, t)
		}
	}
	return out
}

// extractLicense pulls a License from the model card data, falling back to a
// `license:<id>` entry in the merged tag list. Returns nil when no license is
// declared.
func extractLicense(card hfCardData, tags []string) *License {
	id := strings.TrimSpace(card.License)
	if id == "" {
		for _, t := range tags {
			if rest, ok := strings.CutPrefix(strings.ToLower(t), "license:"); ok {
				id = strings.TrimSpace(rest)
				break
			}
		}
	}
	if id == "" {
		return nil
	}
	return &License{
		ID:   id,
		Name: strings.TrimSpace(card.LicenseName),
		Link: strings.TrimSpace(card.LicenseLink),
	}
}

// quantPattern matches the common quantization suffixes vLLM users append to
// model ids: AWQ, GPTQ, FP8, INT8/INT4, GGUF, BNB-4BIT, etc.
var quantPattern = regexp.MustCompile(`(?i)(?:^|[-_.])(awq|gptq|gguf|nvfp4|fp8|fp4|int8|int4|bnb-4bit|bnb-8bit|nf4|w8a8|w4a16)(?:[-_.]|$)`)

// quantFromName extracts a normalized quantization label from a model id.
// It first tries vLLM-style vendor suffixes (AWQ, GPTQ, FP8, NVFP4, ...)
// and falls back to llama.cpp-style GGUF tier suffixes (Q4_K_M, IQ3_XXS,
// BF16, ...) so llama.cpp endpoints — whose `id` is typically the local
// model filename — still get useful quant detection without HF.
func quantFromName(id string) string {
	if m := quantPattern.FindStringSubmatch(id); len(m) >= 2 {
		return strings.ToLower(m[1])
	}
	return ggufQuantInID(id)
}

// quantFromQuantizationConfig handles the parsed config.json
// `quantization_config` object. It recognizes NVFP4 via either
// quant_method=="nvfp4" or compressed-tensors format=="nvfp4-pack-quantized",
// and otherwise returns the lowercased quant_method ("awq", "gptq", "fp8",
// ...). Returns "" when no quantization is declared.
func quantFromQuantizationConfig(qc hfQuantConfig) string {
	method := strings.ToLower(strings.TrimSpace(qc.QuantMethod))
	format := strings.ToLower(strings.TrimSpace(qc.Format))
	if method == "nvfp4" || strings.Contains(format, "nvfp4") {
		return "nvfp4"
	}
	return method
}

// quantFromConfig is a convenience composition of
// quantFromQuantizationConfig and dtypeToLabel for callers that only need
// config.json-derived signal (no GGUF lookup).
func quantFromConfig(cfg hfConfig) string {
	if q := quantFromQuantizationConfig(cfg.QuantizationConfig); q != "" {
		return q
	}
	return dtypeToLabel(cfg.TorchDtype)
}

// dtypeToLabel maps a torch_dtype string to the short label this library uses
// in Features.Quantization. Returns "" for unrecognized dtypes.
func dtypeToLabel(dtype string) string {
	d := strings.ToLower(strings.TrimSpace(dtype))
	switch {
	case d == "":
		return ""
	case d == "bfloat16" || d == "bf16":
		return "bf16"
	case d == "float16" || d == "half" || d == "fp16":
		return "fp16"
	case d == "float32" || d == "float" || d == "fp32":
		return "fp32"
	case strings.HasPrefix(d, "float8"):
		return "fp8"
	case strings.HasPrefix(d, "float4"):
		return "fp4"
	}
	return ""
}

// applyFeatureFlags populates the boolean capability fields based on pipeline,
// architectures, and the supplied HuggingFace tag list.
func applyFeatureFlags(f *Features, tags []string) {
	tagSet := make(map[string]struct{}, len(tags))
	for _, t := range tags {
		tagSet[strings.ToLower(t)] = struct{}{}
	}
	hasTag := func(prefixes ...string) bool {
		for t := range tagSet {
			for _, p := range prefixes {
				if t == p || strings.HasPrefix(t, p+":") || strings.Contains(t, p) {
					return true
				}
			}
		}
		return false
	}
	pipeline := strings.ToLower(f.Pipeline)
	switch pipeline {
	case "text-generation", "text2text-generation", "conversational":
		f.TextGeneration = true
	case "feature-extraction", "sentence-similarity":
		f.Embedding = true
	case "image-text-to-text", "visual-question-answering", "image-to-text":
		f.TextGeneration = true
		f.Vision = true
	case "automatic-speech-recognition", "audio-classification", "text-to-audio", "text-to-speech":
		f.Audio = true
	}
	if hasTag("vision", "multimodal", "image-text-to-text", "image-to-text", "vlm") {
		f.Vision = true
	}
	if hasTag("audio", "speech", "asr") {
		f.Audio = true
	}
	if hasTag("tool", "function-calling", "tool-use", "tools") {
		f.ToolUse = true
	}
	if hasTag("reasoning", "chain-of-thought", "thinking") {
		f.Reasoning = true
	}
	if hasTag("code", "coder", "programming") {
		f.Code = true
	}
	if hasTag("sentence-similarity", "sentence-transformers", "feature-extraction") {
		f.Embedding = true
	}
	for _, a := range f.Architectures {
		la := strings.ToLower(a)
		if strings.Contains(la, "forcausallm") || strings.Contains(la, "forconditionalgeneration") {
			f.TextGeneration = true
		}
		if strings.Contains(la, "vision") || strings.Contains(la, "vl") || strings.Contains(la, "imagetext") {
			f.Vision = true
		}
		if strings.Contains(la, "embedding") || strings.HasSuffix(la, "model") && f.Embedding {
			f.Embedding = true
		}
	}
}
