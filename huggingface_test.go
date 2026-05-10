package modelmeta

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func TestHFStringListUnmarshal(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{`null`, nil},
		{`""`, nil},
		{`"meta-llama/Llama-3"`, []string{"meta-llama/Llama-3"}},
		{`["a","b"]`, []string{"a", "b"}},
	}
	for _, c := range cases {
		var s hfStringList
		if err := json.Unmarshal([]byte(c.in), &s); err != nil {
			t.Fatalf("unmarshal %s: %v", c.in, err)
		}
		if len(s) != len(c.want) {
			t.Errorf("%s: len %d want %d (%v)", c.in, len(s), len(c.want), s)
			continue
		}
		for i := range s {
			if s[i] != c.want[i] {
				t.Errorf("%s: [%d]=%q want %q", c.in, i, s[i], c.want[i])
			}
		}
	}
}

func TestQuantFromName(t *testing.T) {
	cases := map[string]string{
		"meta-llama/Llama-3-70B-Instruct-AWQ": "awq",
		"foo/bar-GPTQ-4bit":                   "gptq",
		"some/Model-FP8":                      "fp8",
		"mistralai/Ministral-8B-Instruct":     "",
		"foo/bar.gguf":                        "gguf",
		"qwen/Qwen-bnb-4bit":                  "bnb-4bit",
	}
	for in, want := range cases {
		if got := quantFromName(in); got != want {
			t.Errorf("quantFromName(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestApplyFeatureFlags(t *testing.T) {
	f := Features{
		Pipeline:      "text-generation",
		Architectures: []string{"LlamaForCausalLM"},
	}
	applyFeatureFlags(&f, []string{"vision", "tool-use", "code"})
	if !f.TextGeneration || !f.Vision || !f.ToolUse || !f.Code {
		t.Fatalf("flags not applied: %+v", f)
	}

	emb := Features{Pipeline: "feature-extraction"}
	applyFeatureFlags(&emb, nil)
	if !emb.Embedding || emb.TextGeneration {
		t.Fatalf("embedding mis-classified: %+v", emb)
	}

	audio := Features{Pipeline: "automatic-speech-recognition"}
	applyFeatureFlags(&audio, nil)
	if !audio.Audio {
		t.Fatalf("audio not detected: %+v", audio)
	}
}

func TestExtractLicense(t *testing.T) {
	cases := []struct {
		name string
		card hfCardData
		tags []string
		want *License
	}{
		{name: "nothing declared", want: nil},
		{
			name: "cardData.license only",
			card: hfCardData{License: "apache-2.0"},
			want: &License{ID: "apache-2.0"},
		},
		{
			name: "other-style with name and link",
			card: hfCardData{License: "other", LicenseName: "Llama 3 Community", LicenseLink: "https://example.com/llama3"},
			want: &License{ID: "other", Name: "Llama 3 Community", Link: "https://example.com/llama3"},
		},
		{
			name: "fallback to license: tag",
			tags: []string{"transformers", "License:MIT", "text-generation"},
			want: &License{ID: "mit"},
		},
		{
			name: "cardData wins over tag fallback",
			card: hfCardData{License: "apache-2.0"},
			tags: []string{"license:mit"},
			want: &License{ID: "apache-2.0"},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := extractLicense(c.card, c.tags)
			if (got == nil) != (c.want == nil) {
				t.Fatalf("nil mismatch: got %v want %v", got, c.want)
			}
			if got == nil {
				return
			}
			if *got != *c.want {
				t.Fatalf("got %+v want %+v", *got, *c.want)
			}
		})
	}
}

func TestExtractFeaturesQuantFromName(t *testing.T) {
	got, tags := extractFeatures(nil, "meta-llama/Meta-Llama-3-70B-Instruct-AWQ")
	if got.Quantization != "awq" {
		t.Fatalf("Quantization = %q, want awq", got.Quantization)
	}
	if tags != nil {
		t.Fatalf("tags should be nil when info is nil, got %v", tags)
	}
}

func TestHFClientCacheAndNotFound(t *testing.T) {
	var hits int32
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/org/known", func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&hits, 1)
		_, _ = w.Write([]byte(`{"id":"org/known","pipeline_tag":"text-generation","tags":["transformers"],"config":{"architectures":["LlamaForCausalLM"]}}`))
	})
	mux.HandleFunc("/api/models/org/missing", func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "no", http.StatusNotFound)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	c := newHFClient(srv.URL, "", srv.Client())
	for i := 0; i < 3; i++ {
		info, err := c.fetch(context.Background(), "org/known")
		if err != nil {
			t.Fatalf("fetch: %v", err)
		}
		if info.Pipeline != "text-generation" {
			t.Fatalf("pipeline = %q", info.Pipeline)
		}
	}
	if hits != 1 {
		t.Errorf("cache miss: hits=%d, want 1", hits)
	}

	if _, err := c.fetch(context.Background(), "org/missing"); !errors.Is(err, errHFNotFound) {
		t.Fatalf("expected errHFNotFound, got %v", err)
	}
	// Repeat lookup on missing model should also short-circuit.
	if _, err := c.fetch(context.Background(), "org/missing"); !errors.Is(err, errHFNotFound) {
		t.Fatalf("expected cached errHFNotFound, got %v", err)
	}
}

func TestHFClientResolveLineage(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/org/child", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"id":"org/child","cardData":{"base_model":"org/parent"}}`))
	})
	mux.HandleFunc("/api/models/org/parent", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"id":"org/parent","cardData":{"base_model":["org/grand"]}}`))
	})
	mux.HandleFunc("/api/models/org/grand", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"id":"org/grand"}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	c := newHFClient(srv.URL, "", srv.Client())
	got := c.resolveLineage(context.Background(), "org/child", 0)
	want := []string{"org/parent", "org/grand"}
	if len(got) != len(want) {
		t.Fatalf("lineage = %v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("lineage[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestHFClientLineageCycle(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/org/a", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"id":"org/a","cardData":{"base_model":"org/b"}}`))
	})
	mux.HandleFunc("/api/models/org/b", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"id":"org/b","cardData":{"base_model":"org/a"}}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	c := newHFClient(srv.URL, "", srv.Client())
	got := c.resolveLineage(context.Background(), "org/a", 0)
	if len(got) != 1 || got[0] != "org/b" {
		t.Fatalf("cycle handling broke: %v", got)
	}
}
