package modelmeta

import (
	"context"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sort"
	"testing"
)

func TestGroupByRoot(t *testing.T) {
	in := []vllmModel{
		{ID: "org/Llama-3-8B-Instruct", Root: "org/Llama-3-8B-Instruct", MaxModelLen: 8192, OwnedBy: "org"},
		{ID: "llama-3-8b", Root: "org/Llama-3-8B-Instruct"},
		{ID: "default", Root: "org/Llama-3-8B-Instruct"},
		{ID: "embed", Root: "", MaxModelLen: 512, OwnedBy: "x"},
	}
	got := groupByRoot(in)
	if len(got) != 2 {
		t.Fatalf("groups = %d, want 2", len(got))
	}
	first := got[0]
	if first.root != "org/Llama-3-8B-Instruct" {
		t.Fatalf("root = %q", first.root)
	}
	if !reflect.DeepEqual(first.aliases, []string{"default", "llama-3-8b"}) {
		t.Fatalf("aliases = %v", first.aliases)
	}
	if first.maxModelLen != 8192 || first.ownedBy != "org" {
		t.Fatalf("attrs not propagated: %+v", first)
	}
	second := got[1]
	if second.root != "embed" || len(second.aliases) != 0 || second.maxModelLen != 512 {
		t.Fatalf("self-root group wrong: %+v", second)
	}
}

func TestDedupSort(t *testing.T) {
	got := dedupSort([]string{"b", "a", "b", "c", "a"})
	want := []string{"a", "b", "c"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("dedupSort = %v, want %v", got, want)
	}
	if dedupSort(nil) != nil {
		t.Fatalf("dedupSort(nil) should stay nil")
	}
}

// TestEnumerateEndToEnd wires a fake vLLM endpoint and a fake HuggingFace API
// together and asserts the deduplicated, feature-resolved output.
func TestEnumerateEndToEnd(t *testing.T) {
	vllmHandler := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"object":"list",
			"data":[
				{"id":"meta-llama/Meta-Llama-3-8B-Instruct","object":"model","root":"meta-llama/Meta-Llama-3-8B-Instruct","owned_by":"meta-llama","max_model_len":8192},
				{"id":"llama3","object":"model","root":"meta-llama/Meta-Llama-3-8B-Instruct","owned_by":"meta-llama"},
				{"id":"default","object":"model","root":"meta-llama/Meta-Llama-3-8B-Instruct","owned_by":"meta-llama"},
				{"id":"BAAI/bge-small-en","object":"model","root":"BAAI/bge-small-en","owned_by":"baai","max_model_len":512},
				{"id":"unknown/Model-AWQ","object":"model","root":"unknown/Model-AWQ","owned_by":"x"}
			]
		}`))
	})
	vllmSrv := httptest.NewServer(vllmHandler)
	defer vllmSrv.Close()

	hfMux := http.NewServeMux()
	hfMux.HandleFunc("/api/models/meta-llama/Meta-Llama-3-8B-Instruct", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"meta-llama/Meta-Llama-3-8B-Instruct",
			"pipeline_tag":"text-generation",
			"tags":["transformers","tool-use"],
			"config":{"architectures":["LlamaForCausalLM"],"max_position_embeddings":8192},
			"cardData":{"base_model":"meta-llama/Meta-Llama-3-8B"}
		}`))
	})
	hfMux.HandleFunc("/api/models/meta-llama/Meta-Llama-3-8B", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"id":"meta-llama/Meta-Llama-3-8B","pipeline_tag":"text-generation","config":{"architectures":["LlamaForCausalLM"]}}`))
	})
	hfMux.HandleFunc("/api/models/BAAI/bge-small-en", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"id":"BAAI/bge-small-en","pipeline_tag":"feature-extraction","tags":["sentence-transformers"]}`))
	})
	hfMux.HandleFunc("/api/models/unknown/Model-AWQ", func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "missing", http.StatusNotFound)
	})
	hfSrv := httptest.NewServer(hfMux)
	defer hfSrv.Close()

	e := &Enumerator{
		EndpointURL:  vllmSrv.URL,
		HFBaseURL:    hfSrv.URL,
		HTTPClient:   vllmSrv.Client(),
		HFHTTPClient: hfSrv.Client(),
	}
	models, err := e.Enumerate(context.Background())
	if err != nil {
		t.Fatalf("Enumerate: %v", err)
	}
	if len(models) != 3 {
		t.Fatalf("models = %d, want 3 (got %+v)", len(models), models)
	}
	// Sorted by Root.
	if !sort.SliceIsSorted(models, func(i, j int) bool { return models[i].Root < models[j].Root }) {
		t.Fatalf("not sorted: %+v", models)
	}

	byRoot := map[string]Model{}
	for _, m := range models {
		byRoot[m.Root] = m
	}

	llama := byRoot["meta-llama/Meta-Llama-3-8B-Instruct"]
	if llama.Root == "" {
		t.Fatalf("llama group missing")
	}
	if !reflect.DeepEqual(llama.Aliases, []string{"default", "llama3"}) {
		t.Errorf("llama aliases = %v", llama.Aliases)
	}
	if !llama.Features.TextGeneration || !llama.Features.ToolUse {
		t.Errorf("llama features = %+v", llama.Features)
	}
	if llama.MaxModelLen != 8192 {
		t.Errorf("llama max len = %d", llama.MaxModelLen)
	}
	if !reflect.DeepEqual(llama.Lineage, []string{"meta-llama/Meta-Llama-3-8B"}) {
		t.Errorf("llama lineage = %v", llama.Lineage)
	}

	bge := byRoot["BAAI/bge-small-en"]
	if !bge.Features.Embedding || bge.Features.TextGeneration {
		t.Errorf("bge features = %+v", bge.Features)
	}
	if len(bge.Aliases) != 0 {
		t.Errorf("bge aliases should be empty: %v", bge.Aliases)
	}

	awq := byRoot["unknown/Model-AWQ"]
	if awq.Features.Quantization != "awq" {
		t.Errorf("awq quant = %q, want awq", awq.Features.Quantization)
	}
	if awq.HFTags != nil {
		t.Errorf("HFTags should be nil on HF 404, got %v", awq.HFTags)
	}

	wantLlamaTags := []string{"transformers", "tool-use"}
	if !reflect.DeepEqual(llama.HFTags, wantLlamaTags) {
		t.Errorf("llama HFTags = %v, want %v", llama.HFTags, wantLlamaTags)
	}
	wantBgeTags := []string{"sentence-transformers"}
	if !reflect.DeepEqual(bge.HFTags, wantBgeTags) {
		t.Errorf("bge HFTags = %v, want %v", bge.HFTags, wantBgeTags)
	}
}

// TestEnumerateSkipHF ensures the library is usable without HF resolution.
func TestEnumerateSkipHF(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"foo/Bar-FP8","root":"foo/Bar-FP8","max_model_len":4096}]}`))
	}))
	defer srv.Close()

	e := &Enumerator{EndpointURL: srv.URL, HTTPClient: srv.Client(), SkipHF: true}
	models, err := e.Enumerate(context.Background())
	if err != nil {
		t.Fatalf("Enumerate: %v", err)
	}
	if len(models) != 1 || models[0].Features.Quantization != "fp8" {
		t.Fatalf("unexpected models: %+v", models)
	}
	if models[0].HFTags != nil {
		t.Errorf("HFTags must be nil with SkipHF, got %v", models[0].HFTags)
	}
}
