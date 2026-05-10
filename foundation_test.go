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
		"qwen2.5-7b-instruct-q4_k_m":             "qwen2 5 7b instruct",
		"meta-llama/Meta-Llama-3-8B-Instruct-AWQ": "meta llama 3 8b instruct",
		"TheBloke/Llama-3-8B-GGUF":               "llama 3 8b",
		"foo-IQ3_XXS":                            "foo",
		"default":                                "default",
		"":                                       "",
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

// TestEnumerateFoundationFromLineage verifies that when a model has a
// declared base_model chain, Foundation is the deepest ancestor and is
// fully resolved.
func TestEnumerateFoundationFromLineage(t *testing.T) {
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
	if m.Foundation == nil {
		t.Fatal("Foundation should be set from lineage tip")
	}
	if m.Foundation.Root != "org/base" {
		t.Errorf("Foundation.Root = %q, want org/base", m.Foundation.Root)
	}
	if !m.Foundation.Features.TextGeneration {
		t.Errorf("Foundation features not resolved: %+v", m.Foundation.Features)
	}
	if m.Foundation.License == nil || m.Foundation.License.ID != "apache-2.0" {
		t.Errorf("Foundation license = %+v", m.Foundation.License)
	}
	if m.Foundation.Foundation != nil {
		t.Error("Foundation.Foundation must be nil (non-recursive)")
	}
}

// TestEnumerateFoundationFromGuess verifies that when direct HF lookup
// fails, Foundation is populated from the search-guessed parent.
func TestEnumerateFoundationFromGuess(t *testing.T) {
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
	if m.Foundation == nil {
		t.Fatal("Foundation should be guessed when direct lookup fails")
	}
	if m.Foundation.Root != "Qwen/Qwen2.5-7B-Instruct" {
		t.Errorf("Foundation.Root = %q", m.Foundation.Root)
	}
	if !m.Foundation.Flags.HuggingFace {
		t.Errorf("Foundation should have huggingface flag set")
	}
	if m.Foundation.Foundation != nil {
		t.Error("Foundation.Foundation must be nil")
	}
}

// TestEnumerateFoundationSkipGuess verifies the SkipGuessParent flag.
func TestEnumerateFoundationSkipGuess(t *testing.T) {
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
	if m.Foundation != nil {
		t.Errorf("Foundation should be nil when SkipGuessParent is set, got %+v", m.Foundation)
	}
	if searchHit {
		t.Error("search endpoint must not be called when SkipGuessParent is true")
	}
}
