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
	}
	for in, want := range cases {
		if got := normalizeForSearch(in); got != want {
			t.Errorf("normalizeForSearch(%q) = %q, want %q", in, got, want)
		}
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
