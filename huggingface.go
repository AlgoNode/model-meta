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

	mu    sync.Mutex
	cache map[string]*hfModelInfo
	miss  map[string]struct{}
}

func newHFClient(baseURL, token string, httpc *http.Client) *hfClient {
	if baseURL == "" {
		baseURL = hfBaseURL
	}
	if httpc == nil {
		httpc = http.DefaultClient
	}
	return &hfClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		token:   token,
		http:    httpc,
		cache:   make(map[string]*hfModelInfo),
		miss:    make(map[string]struct{}),
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

// resolveLineage walks base_model links from the given id outward, stopping
// at the first model not found on the Hub or a previously visited node. The
// starting id itself is not included in the returned slice.
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
		if len(info.CardData.BaseModel) == 0 {
			return out
		}
		next := strings.TrimSpace(info.CardData.BaseModel[0])
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

func quantFromName(id string) string {
	m := quantPattern.FindStringSubmatch(id)
	if len(m) < 2 {
		return ""
	}
	return strings.ToLower(m[1])
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
