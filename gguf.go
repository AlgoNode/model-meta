package modelmeta

import (
	"regexp"
	"sort"
	"strings"
)

// ggufQuantPattern extracts the quant tier from a GGUF filename. Anchoring
// on `.gguf` (with an optional `-NNNNN-of-NNNNN` shard suffix) avoids
// false positives in repo paths or owner names that happen to contain
// Q4 / F16 etc. Recognized tiers cover the full llama.cpp set: K-quants
// (Q2_K, Q3_K_S/M/L, Q4_K_S/M, Q5_K_S/M, Q6_K), legacy block quants
// (Q4_0/1, Q5_0/1, Q8_0), i-quants (IQ1_S/M, IQ2_XXS/XS/S/M,
// IQ3_XXS/XS/S/M, IQ4_NL/XS), and full-precision dumps (F16, BF16, F32).
var ggufQuantPattern = regexp.MustCompile(`(?i)[-_.](I?Q\d(?:_[A-Z0-9]+)*|F16|BF16|F32)(?:-\d{5}-of-\d{5})?\.gguf$`)

// ggufIDQuantPattern matches a quant tier embedded in a vLLM model id
// (without a `.gguf` extension). Used to honor user pins like
// `TheBloke/Model-GGUF-Q5_K_M` when picking which sibling to report.
var ggufIDQuantPattern = regexp.MustCompile(`(?i)(?:^|[-_.])(I?Q\d(?:_[A-Z0-9]+)*|F16|BF16|F32)(?:[-_.]|$)`)

// ggufSplitPattern matches the multi-part shard suffix that llama.cpp uses
// when a single quant is split across files: `-00001-of-00003.gguf`. The
// captured group is the shard number; we use it to keep only the first
// shard when picking a canonical file (all shards share the same quant).
var ggufSplitPattern = regexp.MustCompile(`(?i)-(\d{5})-of-\d{5}\.gguf$`)

// ggufQuantPreference orders the most-recommended quants first. Used to
// pick a representative file when a repo hosts many quants and the vLLM id
// gives no hint. Q4_K_M is the de-facto default; the rest descends through
// the usual quality/size trade-off.
var ggufQuantPreference = []string{
	"q4_k_m", "q5_k_m", "q5_k_s", "q4_k_s",
	"q6_k", "q8_0",
	"q3_k_l", "q3_k_m", "q3_k_s", "q2_k",
	"q5_1", "q5_0", "q4_1", "q4_0",
	"iq4_nl", "iq4_xs",
	"iq3_m", "iq3_s", "iq3_xs", "iq3_xxs",
	"iq2_m", "iq2_s", "iq2_xs", "iq2_xxs",
	"iq1_m", "iq1_s",
	"bf16", "f16", "f32",
}

// isGGUFRepo returns true when the HF model info clearly belongs to a GGUF
// repo: library_name=="gguf", a "gguf" tag (info or cardData), or any
// .gguf sibling filename.
func isGGUFRepo(info *hfModelInfo) bool {
	if info == nil {
		return false
	}
	if strings.EqualFold(info.LibraryName, "gguf") {
		return true
	}
	for _, t := range info.Tags {
		if strings.EqualFold(t, "gguf") {
			return true
		}
	}
	for _, t := range info.CardData.Tags {
		if strings.EqualFold(t, "gguf") {
			return true
		}
	}
	for _, s := range info.Siblings {
		if strings.HasSuffix(strings.ToLower(s.Filename), ".gguf") {
			return true
		}
	}
	return false
}

// quantFromGGUF parses the quant tier from a GGUF filename, returning the
// normalized lowercase label (e.g. "q4_k_m", "iq3_xxs", "bf16"). Returns
// "" when no tier can be matched.
func quantFromGGUF(filename string) string {
	m := ggufQuantPattern.FindStringSubmatch(filename)
	if len(m) < 2 {
		return ""
	}
	return strings.ToLower(m[1])
}

// ggufQuantInID returns the GGUF quant tier embedded in a model id like
// "TheBloke/Model-GGUF-Q5_K_M", or "" when none is present. When multiple
// tier-shaped tokens appear, the last one wins — quant pins are
// conventionally placed at the end of an id.
func ggufQuantInID(id string) string {
	matches := ggufIDQuantPattern.FindAllStringSubmatch(id, -1)
	if len(matches) == 0 {
		return ""
	}
	return strings.ToLower(matches[len(matches)-1][1])
}

// pickCanonicalGGUF chooses one .gguf sibling to represent a GGUF repo.
// Selection priority:
//
//  1. file whose parsed quant matches a quant suffix in vllmID
//     (so users who pin Foo-GGUF:Q4_K_M get a deterministic answer)
//  2. file whose quant comes earliest in ggufQuantPreference
//  3. largest reported size (when sizes are present)
//  4. lexicographically first remaining .gguf file
//
// Multi-part shards are collapsed to their first part (00001-of-NNNNN);
// other shards share the same quant and would only inflate the candidate
// set. Returns "" when no .gguf siblings exist.
func pickCanonicalGGUF(siblings []hfSibling, vllmID string) string {
	type cand struct {
		name  string
		quant string
		size  int64
	}
	var cands []cand
	for _, s := range siblings {
		lower := strings.ToLower(s.Filename)
		if !strings.HasSuffix(lower, ".gguf") {
			continue
		}
		if m := ggufSplitPattern.FindStringSubmatch(s.Filename); len(m) > 0 && m[1] != "00001" {
			continue
		}
		cands = append(cands, cand{name: s.Filename, quant: quantFromGGUF(s.Filename), size: s.Size})
	}
	if len(cands) == 0 {
		return ""
	}

	if idQuant := ggufQuantInID(vllmID); idQuant != "" {
		for _, c := range cands {
			if c.quant == idQuant {
				return c.name
			}
		}
	}

	for _, want := range ggufQuantPreference {
		for _, c := range cands {
			if c.quant == want {
				return c.name
			}
		}
	}

	sort.Slice(cands, func(i, j int) bool {
		if cands[i].size != cands[j].size {
			return cands[i].size > cands[j].size
		}
		return cands[i].name < cands[j].name
	})
	return cands[0].name
}
