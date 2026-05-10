package modelmeta

import (
	"context"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

func TestNormalizeForSearch(t *testing.T) {
	cases := map[string]string{
		"qwen2.5-7b-instruct-q4_k_m":              "qwen2 5 7b instruct",
		"meta-llama/Meta-Llama-3-8B-Instruct-AWQ": "meta llama 3 8b instruct",
		"TheBloke/Llama-3-8B-GGUF":                "llama 3 8b",
		"foo-IQ3_XXS":                             "foo",
		"default":                                 "default",
		"":                                        "",

		// Fork-marker words (compliance + merge/training + intensity) are
		// stripped so HF search isn't biased toward derivatives.
		"AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4":              "gemma 4 26b a4b it",
		"llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic":    "gemma 4 26b a4b it",
		"some-org/foo-7B-merge-dare-ties":                         "foo 7b",
		"some-org/foo-7B-Hermes-LoRA":                             "foo 7b",
	}
	for in, want := range cases {
		if got := normalizeForSearch(in); got != want {
			t.Errorf("normalizeForSearch(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestBaseModelFromTags(t *testing.T) {
	cases := []struct {
		name string
		tags []string
		want string
	}{
		{name: "empty", tags: nil, want: ""},
		{
			name: "plain id",
			tags: []string{"transformers", "base_model:google/gemma-4-26b-a4b-it"},
			want: "google/gemma-4-26b-a4b-it",
		},
		{
			name: "with finetune relationship prefix",
			tags: []string{"base_model:finetune:google/gemma-4-26b-a4b-it"},
			want: "google/gemma-4-26b-a4b-it",
		},
		{
			name: "with quantized relationship prefix",
			tags: []string{"base_model:quantized:foo/bar"},
			want: "foo/bar",
		},
		{
			name: "first wins when duplicates with prefix",
			tags: []string{"base_model:google/gemma-4-26b-a4b-it", "base_model:finetune:google/gemma-4-26b-a4b-it"},
			want: "google/gemma-4-26b-a4b-it",
		},
		{name: "missing slash is rejected", tags: []string{"base_model:notanid"}, want: ""},
		{name: "unrelated tag", tags: []string{"transformers", "license:apache-2.0"}, want: ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := baseModelFromTags(c.tags); got != c.want {
				t.Errorf("baseModelFromTags(%v) = %q, want %q", c.tags, got, c.want)
			}
		})
	}
}

func TestTagsLookQuantized(t *testing.T) {
	cases := []struct {
		name string
		tags []string
		want bool
	}{
		{name: "empty", tags: nil, want: false},
		{name: "transformers + safetensors", tags: []string{"transformers", "safetensors"}, want: false},
		{name: "gguf", tags: []string{"transformers", "gguf"}, want: true},
		{name: "compressed-tensors", tags: []string{"compressed-tensors"}, want: true},
		{name: "uppercase still matches", tags: []string{"GGUF"}, want: true},
		{name: "AWQ", tags: []string{"awq"}, want: true},
		{name: "4-bit", tags: []string{"4-bit"}, want: true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := tagsLookQuantized(c.tags); got != c.want {
				t.Errorf("tagsLookQuantized(%v) = %v, want %v", c.tags, got, c.want)
			}
		})
	}
}

func TestSimilarityScore(t *testing.T) {
	cases := []struct {
		a, b string
		want float64
	}{
		{"", "anything", 0},
		{"qwen2 7b instruct", "qwen2 7b instruct", 1.0},
		{"qwen2 7b instruct", "qwen2 7b", 2.0 / 3.0}, // 2 of 3 query tokens hit
		{"a b c d", "a", 0.25},
		{"a b", "x y", 0},
	}
	for _, c := range cases {
		got := similarityScore(c.a, c.b)
		if (got-c.want) > 1e-9 || (c.want-got) > 1e-9 {
			t.Errorf("similarityScore(%q,%q) = %v, want %v", c.a, c.b, got, c.want)
		}
	}
}

func TestHFClientSearchModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("search") != "qwen2 5 7b instruct" {
			t.Errorf("unexpected search query: %q", r.URL.Query().Get("search"))
		}
		if r.URL.Query().Get("sort") != "downloads" || r.URL.Query().Get("direction") != "-1" {
			t.Errorf("missing sort params: %v", r.URL.Query())
		}
		_, _ = w.Write([]byte(`[
			{"id":"Qwen/Qwen2.5-7B-Instruct","modelId":"Qwen/Qwen2.5-7B-Instruct"},
			{"id":"Qwen/Qwen2.5-7B"}
		]`))
	}))
	defer srv.Close()

	c := newHFClient(srv.URL, "", srv.Client())
	results, err := c.searchModels(context.Background(), "qwen2 5 7b instruct", 5)
	if err != nil {
		t.Fatalf("searchModels: %v", err)
	}
	if len(results) != 2 || results[0].ID != "Qwen/Qwen2.5-7B-Instruct" {
		t.Fatalf("results = %+v", results)
	}
}

func TestGuessParent(t *testing.T) {
	t.Run("happy path: top result above threshold", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`[{"id":"Qwen/Qwen2.5-7B-Instruct"}]`))
		}))
		defer srv.Close()
		c := newHFClient(srv.URL, "", srv.Client())
		got := c.guessParent(context.Background(), "qwen2.5-7b-instruct-q4_k_m")
		if got != "Qwen/Qwen2.5-7B-Instruct" {
			t.Errorf("got %q", got)
		}
	})

	t.Run("rejected when below similarity threshold", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			// Top result shares 1 of 4 query tokens => 0.25 < 0.6 threshold.
			_, _ = w.Write([]byte(`[{"id":"some-org/totally-different-model"}]`))
		}))
		defer srv.Close()
		c := newHFClient(srv.URL, "", srv.Client())
		got := c.guessParent(context.Background(), "qwen2.5-7b-instruct-merge")
		if got != "" {
			t.Errorf("expected empty, got %q", got)
		}
	})

	t.Run("single-token query is not searched", func(t *testing.T) {
		called := false
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			called = true
			_, _ = w.Write([]byte(`[]`))
		}))
		defer srv.Close()
		c := newHFClient(srv.URL, "", srv.Client())
		if got := c.guessParent(context.Background(), "default"); got != "" {
			t.Errorf("got %q, want empty", got)
		}
		if called {
			t.Error("search endpoint should not be hit for single-token queries")
		}
	})

	t.Run("self-id is skipped", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`[{"id":"Qwen/Qwen2.5-7B-Instruct"}]`))
		}))
		defer srv.Close()
		c := newHFClient(srv.URL, "", srv.Client())
		got := c.guessParent(context.Background(), "Qwen/Qwen2.5-7B-Instruct")
		if got != "" {
			t.Errorf("self-id should be skipped, got %q", got)
		}
	})

	t.Run("quantized candidates are filtered", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			// Top result is a GGUF fork; second is the unquantized
			// upstream. The quant filter must skip the first.
			_, _ = w.Write([]byte(`[
				{"id":"someone/Qwen2.5-7B-Instruct-GGUF","tags":["gguf","transformers"]},
				{"id":"Qwen/Qwen2.5-7B-Instruct","tags":["transformers","safetensors"]}
			]`))
		}))
		defer srv.Close()
		c := newHFClient(srv.URL, "", srv.Client())
		got := c.guessParent(context.Background(), "qwen2.5-7b-instruct-q4_k_m")
		if got != "Qwen/Qwen2.5-7B-Instruct" {
			t.Errorf("got %q, want Qwen/Qwen2.5-7B-Instruct (quant fork must be skipped)", got)
		}
	})

	t.Run("siblings rejected by reverse-similarity gate", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			// Candidate normalizes to "qwen2 5 7b instruct rogue
			// merge" — query-tokens-in-candidate is 4/4 but
			// candidate-tokens-in-query is only 4/6 = 0.67 < 0.8,
			// so it must be rejected even though the forward
			// similarity is 1.0.
			//
			// "merge" is a fork marker, gets stripped — so the
			// fixture uses a non-marker noise word.
			_, _ = w.Write([]byte(`[
				{"id":"someone/Qwen2.5-7B-Instruct-rogue-extra-noise-banana","tags":["transformers"]}
			]`))
		}))
		defer srv.Close()
		c := newHFClient(srv.URL, "", srv.Client())
		if got := c.guessParent(context.Background(), "qwen2.5-7b-instruct-q4_k_m"); got != "" {
			t.Errorf("got %q, want empty (sibling-with-extras must be rejected)", got)
		}
	})
}

// TestEnumerateAncestorFromLineage verifies that when a model has a
// declared base_model chain, Ancestor is the deepest entry and is
// fully resolved.
func TestEnumerateAncestorFromLineage(t *testing.T) {
	hfMux := http.NewServeMux()
	hfMux.HandleFunc("/api/models/org/finetune", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"org/finetune",
			"pipeline_tag":"text-generation",
			"config":{"architectures":["LlamaForCausalLM"]},
			"cardData":{"base_model":"org/instruct"}
		}`))
	})
	hfMux.HandleFunc("/api/models/org/instruct", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"org/instruct",
			"pipeline_tag":"text-generation",
			"cardData":{"base_model":"org/base","license":"apache-2.0"}
		}`))
	})
	hfMux.HandleFunc("/api/models/org/base", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"org/base",
			"pipeline_tag":"text-generation",
			"tags":["transformers"],
			"cardData":{"license":"apache-2.0"}
		}`))
	})
	hfSrv := httptest.NewServer(hfMux)
	defer hfSrv.Close()

	e := &Enumerator{HFBaseURL: hfSrv.URL, HFHTTPClient: hfSrv.Client()}
	m, err := e.Resolve(context.Background(), "org/finetune")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if !reflect.DeepEqual(m.Lineage, []string{"org/instruct", "org/base"}) {
		t.Fatalf("Lineage = %v", m.Lineage)
	}
	if m.Ancestor == nil {
		t.Fatal("Ancestor should be set from lineage tip")
	}
	if m.Ancestor.Root != "org/base" {
		t.Errorf("Ancestor.Root = %q, want org/base", m.Ancestor.Root)
	}
	if !m.Ancestor.Features.TextGeneration {
		t.Errorf("Ancestor features not resolved: %+v", m.Ancestor.Features)
	}
	if m.Ancestor.License == nil || m.Ancestor.License.ID != "apache-2.0" {
		t.Errorf("Ancestor license = %+v", m.Ancestor.License)
	}
	if m.Ancestor.Ancestor != nil {
		t.Error("Ancestor.Ancestor must be nil (non-recursive)")
	}
}

// TestEnumerateAncestorFromGuess verifies that when direct HF lookup
// fails, Ancestor is populated from the search-guessed parent.
func TestEnumerateAncestorFromGuess(t *testing.T) {
	hfMux := http.NewServeMux()
	// Direct lookup for the llama.cpp-style id -> 404.
	hfMux.HandleFunc("/api/models/qwen2.5-7b-instruct-q4_k_m", func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "no", http.StatusNotFound)
	})
	// Search returns the canonical Qwen entry.
	hfMux.HandleFunc("/api/models", func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Query().Get("search"), "qwen2") {
			http.Error(w, "unexpected", http.StatusBadRequest)
			return
		}
		_, _ = w.Write([]byte(`[{"id":"Qwen/Qwen2.5-7B-Instruct"}]`))
	})
	// Resolve the guessed parent.
	hfMux.HandleFunc("/api/models/Qwen/Qwen2.5-7B-Instruct", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"Qwen/Qwen2.5-7B-Instruct",
			"pipeline_tag":"text-generation",
			"tags":["transformers"],
			"config":{"architectures":["Qwen2ForCausalLM"],"torch_dtype":"bfloat16"},
			"cardData":{"license":"apache-2.0"}
		}`))
	})
	hfSrv := httptest.NewServer(hfMux)
	defer hfSrv.Close()

	e := &Enumerator{HFBaseURL: hfSrv.URL, HFHTTPClient: hfSrv.Client()}
	m, err := e.Resolve(context.Background(), "qwen2.5-7b-instruct-q4_k_m")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if m.Flags.HuggingFace {
		t.Error("HuggingFace flag should be false (404)")
	}
	if m.Ancestor == nil {
		t.Fatal("Ancestor should be guessed when direct lookup fails")
	}
	if m.Ancestor.Root != "Qwen/Qwen2.5-7B-Instruct" {
		t.Errorf("Ancestor.Root = %q", m.Ancestor.Root)
	}
	if !m.Ancestor.Flags.HuggingFace {
		t.Errorf("Ancestor should have huggingface flag set")
	}
	if m.Ancestor.Ancestor != nil {
		t.Error("Ancestor.Ancestor must be nil")
	}
}

// TestEnumerateAncestorGuessQuantizedNoLineage verifies that a model
// that resolves on HF but declares no base_model and is quantized
// triggers the search fallback. This is the AEON-7/Gemma-NVFP4 case.
func TestEnumerateAncestorGuessQuantizedNoLineage(t *testing.T) {
	hfMux := http.NewServeMux()
	hfMux.HandleFunc("/api/models/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4", func(w http.ResponseWriter, _ *http.Request) {
		// HF returns 200 but cardData is null and there is no
		// declared base_model.
		_, _ = w.Write([]byte(`{
			"id":"AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4",
			"tags":["safetensors","gemma4","compressed-tensors"],
			"config":{
				"architectures":["Gemma4ForConditionalGeneration"],
				"quantization_config":{"quant_method":"compressed-tensors"}
			}
		}`))
	})
	hfMux.HandleFunc("/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4/resolve/main/config.json", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"quantization_config":{"quant_method":"compressed-tensors","config_groups":{"g":{"format":"nvfp4-pack-quantized"}}}}`))
	})
	hfMux.HandleFunc("/api/models", func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Query().Get("search"), "gemma 4 26b") {
			http.Error(w, "unexpected", http.StatusBadRequest)
			return
		}
		_, _ = w.Write([]byte(`[{"id":"google/gemma-4-26b-a4b-it"}]`))
	})
	hfMux.HandleFunc("/api/models/google/gemma-4-26b-a4b-it", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"google/gemma-4-26b-a4b-it",
			"pipeline_tag":"text-generation",
			"config":{"architectures":["Gemma4ForConditionalGeneration"],"torch_dtype":"bfloat16"},
			"cardData":{"license":"gemma"}
		}`))
	})
	hfSrv := httptest.NewServer(hfMux)
	defer hfSrv.Close()

	e := &Enumerator{HFBaseURL: hfSrv.URL, HFHTTPClient: hfSrv.Client()}
	m, err := e.Resolve(context.Background(), "AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if !m.Flags.HuggingFace {
		t.Fatal("model itself should resolve on HF")
	}
	if m.Features.Quantization != "nvfp4" {
		t.Fatalf("Quantization = %q, want nvfp4", m.Features.Quantization)
	}
	if m.Ancestor == nil {
		t.Fatal("Ancestor should be guessed when HF model is quantized but has no lineage")
	}
	if m.Ancestor.Root != "google/gemma-4-26b-a4b-it" {
		t.Errorf("Ancestor.Root = %q", m.Ancestor.Root)
	}
}

// TestEnumerateAncestorAEONShape mirrors the real AEON-7 NVFP4 case:
// the model resolves on HF but cardData is null and there's no
// base_model:* tag on the model itself. After fork-marker stripping the
// search query becomes "gemma 4 26b a4b it"; HF returns the official
// upstream first (no quant tags), and the upstream itself declares its
// own pre-instruct base via base_model:* tags — the pivot walks one
// more step to land on that base.
func TestEnumerateAncestorAEONShape(t *testing.T) {
	hfMux := http.NewServeMux()
	// AEON-7-style model: 200, no cardData, compressed-tensors.
	hfMux.HandleFunc("/api/models/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4",
			"tags":["safetensors","gemma4","compressed-tensors","region:us"],
			"config":{"architectures":["Gemma4ForConditionalGeneration"],"quantization_config":{"quant_method":"compressed-tensors"}}
		}`))
	})
	hfMux.HandleFunc("/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4/resolve/main/config.json", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"quantization_config":{"quant_method":"compressed-tensors","config_groups":{"g":{"format":"nvfp4-pack-quantized"}}}}`))
	})
	hfMux.HandleFunc("/api/models", func(w http.ResponseWriter, r *http.Request) {
		// Verify that "uncensored" was stripped from the query.
		q := r.URL.Query().Get("search")
		if strings.Contains(q, "uncensored") {
			t.Errorf("query still contains uncensored: %q", q)
		}
		if q != "gemma 4 26b a4b it" {
			t.Errorf("unexpected query: %q", q)
		}
		// Mirror real HF response: a quant fork outranks the upstream
		// by downloads but carries `gguf`; quant-filter must drop it
		// so the upstream wins.
		_, _ = w.Write([]byte(`[
			{"id":"llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic-GGUF","tags":["gguf","transformers"]},
			{"id":"google/gemma-4-26B-A4B-it","tags":["transformers","safetensors","base_model:google/gemma-4-26B-A4B"]}
		]`))
	})
	hfMux.HandleFunc("/api/models/google/gemma-4-26B-A4B-it", func(w http.ResponseWriter, _ *http.Request) {
		// Upstream model: cardData is missing but the base_model is
		// surfaced via tags. parentOfInfo should pick it up so the
		// pivot walks one more step.
		_, _ = w.Write([]byte(`{
			"id":"google/gemma-4-26B-A4B-it",
			"pipeline_tag":"text-generation",
			"tags":["transformers","base_model:google/gemma-4-26B-A4B"],
			"config":{"architectures":["Gemma4ForConditionalGeneration"],"torch_dtype":"bfloat16"}
		}`))
	})
	hfMux.HandleFunc("/api/models/google/gemma-4-26B-A4B", func(w http.ResponseWriter, _ *http.Request) {
		// True base: no further base_model.
		_, _ = w.Write([]byte(`{
			"id":"google/gemma-4-26B-A4B",
			"pipeline_tag":"text-generation",
			"tags":["transformers","safetensors"],
			"config":{"architectures":["Gemma4ForConditionalGeneration"],"torch_dtype":"bfloat16"}
		}`))
	})
	hfSrv := httptest.NewServer(hfMux)
	defer hfSrv.Close()

	e := &Enumerator{HFBaseURL: hfSrv.URL, HFHTTPClient: hfSrv.Client()}
	m, err := e.Resolve(context.Background(), "AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if m.Features.Quantization != "nvfp4" {
		t.Fatalf("Quantization = %q, want nvfp4", m.Features.Quantization)
	}
	if m.Ancestor == nil {
		t.Fatal("Ancestor should be populated")
	}
	if m.Ancestor.Root != "google/gemma-4-26B-A4B" {
		t.Errorf("Ancestor.Root = %q, want google/gemma-4-26B-A4B (pivoted past the instruct upstream)", m.Ancestor.Root)
	}
}

// TestEnumerateAncestorPivotsAfterGuess verifies that when the guessed
// parent itself declares a base_model chain, Ancestor pivots to the
// deepest reachable base instead of stopping at the first hit.
//
// Shape: model is a quantized HF entry with no lineage of its own ->
// search returns "org/llama-3-8b-instruct" -> instruct's cardData
// declares "org/llama-3-8b" -> we should land on the base, not the
// instruct fork.
func TestEnumerateAncestorPivotsAfterGuess(t *testing.T) {
	hfMux := http.NewServeMux()
	hfMux.HandleFunc("/api/models/org/llama-3-8b-instruct-q4", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"org/llama-3-8b-instruct-q4",
			"tags":["transformers"],
			"config":{"architectures":["LlamaForCausalLM"],"quantization_config":{"quant_method":"awq"}}
		}`))
	})
	hfMux.HandleFunc("/api/models", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`[{"id":"org/llama-3-8b-instruct"}]`))
	})
	hfMux.HandleFunc("/api/models/org/llama-3-8b-instruct", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"org/llama-3-8b-instruct",
			"pipeline_tag":"text-generation",
			"cardData":{"base_model":"org/llama-3-8b"}
		}`))
	})
	hfMux.HandleFunc("/api/models/org/llama-3-8b", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"org/llama-3-8b",
			"pipeline_tag":"text-generation",
			"tags":["transformers"],
			"config":{"torch_dtype":"bfloat16"}
		}`))
	})
	hfSrv := httptest.NewServer(hfMux)
	defer hfSrv.Close()

	e := &Enumerator{HFBaseURL: hfSrv.URL, HFHTTPClient: hfSrv.Client()}
	m, err := e.Resolve(context.Background(), "org/llama-3-8b-instruct-q4")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if m.Ancestor == nil {
		t.Fatal("Ancestor should be populated")
	}
	if m.Ancestor.Root != "org/llama-3-8b" {
		t.Errorf("Ancestor.Root = %q, want org/llama-3-8b (pivoted past guess)", m.Ancestor.Root)
	}
	if m.Ancestor.Features.Quantization != "bf16" {
		t.Errorf("Ancestor features should be the base's, got %+v", m.Ancestor.Features)
	}
}

// TestEnumerateAncestorPivotCycleSafe verifies that a base_model cycle
// across two pivots (A -> B, B -> A) doesn't loop forever.
func TestEnumerateAncestorPivotCycleSafe(t *testing.T) {
	hfMux := http.NewServeMux()
	hfMux.HandleFunc("/api/models/me/model", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"id":"me/model","cardData":{"base_model":"org/A"}}`))
	})
	hfMux.HandleFunc("/api/models/org/A", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"id":"org/A","cardData":{"base_model":"org/B"}}`))
	})
	hfMux.HandleFunc("/api/models/org/B", func(w http.ResponseWriter, _ *http.Request) {
		// Cycle back to A; pivot must not loop forever.
		_, _ = w.Write([]byte(`{"id":"org/B","cardData":{"base_model":"org/A"}}`))
	})
	hfSrv := httptest.NewServer(hfMux)
	defer hfSrv.Close()

	e := &Enumerator{HFBaseURL: hfSrv.URL, HFHTTPClient: hfSrv.Client()}
	m, err := e.Resolve(context.Background(), "me/model")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if m.Ancestor == nil {
		t.Fatal("Ancestor should be set")
	}
	// Acceptable terminal: either A or B; the test's purpose is to
	// confirm we didn't hang.
	if m.Ancestor.Root != "org/A" && m.Ancestor.Root != "org/B" {
		t.Errorf("Ancestor.Root = %q, want org/A or org/B", m.Ancestor.Root)
	}
}

// TestEnumerateAncestorSkippedForBaseModel pins the gate that stops us
// from pointing a true base at one of its sibling derivatives. A model
// that resolves on HF, declares no base_model, and reports a native
// float dtype is assumed to be a base itself.
func TestEnumerateAncestorSkippedForBaseModel(t *testing.T) {
	hfMux := http.NewServeMux()
	hfMux.HandleFunc("/api/models/meta-llama/Meta-Llama-3-8B", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"meta-llama/Meta-Llama-3-8B",
			"pipeline_tag":"text-generation",
			"config":{"architectures":["LlamaForCausalLM"],"torch_dtype":"bfloat16"}
		}`))
	})
	searchHit := false
	hfMux.HandleFunc("/api/models", func(w http.ResponseWriter, _ *http.Request) {
		searchHit = true
		_, _ = w.Write([]byte(`[]`))
	})
	hfSrv := httptest.NewServer(hfMux)
	defer hfSrv.Close()

	e := &Enumerator{HFBaseURL: hfSrv.URL, HFHTTPClient: hfSrv.Client()}
	m, err := e.Resolve(context.Background(), "meta-llama/Meta-Llama-3-8B")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if m.Ancestor != nil {
		t.Errorf("Ancestor should be nil for a native-dtype HF model with no lineage, got %+v", m.Ancestor)
	}
	if searchHit {
		t.Error("search must not be called for native-dtype base candidates")
	}
}

// TestEnumerateAncestorSkipGuess verifies the SkipGuessParent flag.
func TestEnumerateAncestorSkipGuess(t *testing.T) {
	hfMux := http.NewServeMux()
	hfMux.HandleFunc("/api/models/qwen2.5-7b-instruct-q4_k_m", func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "no", http.StatusNotFound)
	})
	searchHit := false
	hfMux.HandleFunc("/api/models", func(w http.ResponseWriter, _ *http.Request) {
		searchHit = true
		_, _ = w.Write([]byte(`[]`))
	})
	hfSrv := httptest.NewServer(hfMux)
	defer hfSrv.Close()

	e := &Enumerator{HFBaseURL: hfSrv.URL, HFHTTPClient: hfSrv.Client(), SkipGuessParent: true}
	m, err := e.Resolve(context.Background(), "qwen2.5-7b-instruct-q4_k_m")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if m.Ancestor != nil {
		t.Errorf("Ancestor should be nil when SkipGuessParent is set, got %+v", m.Ancestor)
	}
	if searchHit {
		t.Error("search endpoint must not be called when SkipGuessParent is true")
	}
}
