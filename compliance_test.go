package modelmeta

import (
	"context"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

func TestMatchComplianceTags(t *testing.T) {
	cases := []struct {
		name    string
		root    string
		aliases []string
		want    []string
	}{
		{
			name: "no match on plain instruct model",
			root: "meta-llama/Meta-Llama-3-8B-Instruct",
			want: nil,
		},
		{
			name: "Dolphin family, case-insensitive",
			root: "cognitivecomputations/dolphin-2.9-llama3-8b",
			want: []string{"Dolphin"},
		},
		{
			name: "Uncensored substring",
			root: "neuralmagic/Llama-3-Uncensored",
			want: []string{"Uncensored"},
		},
		{
			name: "OpenHermes also matches Hermes",
			root: "NousResearch/OpenHermes-2.5-Mistral-7B",
			want: []string{"Hermes", "OpenHermes"},
		},
		{
			name: "RP word boundary positive",
			root: "Sao10K/L3-RP-8B-v1",
			want: []string{"RP"},
		},
		{
			name: "RPCS3 must NOT trigger RP",
			root: "weird/RPCS3-llama",
			want: nil,
		},
		{
			name: "ERP triggers ERP only, not RP",
			root: "anon/L3-ERP-13B",
			want: []string{"ERP"},
		},
		{
			name: "alias matches when root does not",
			root: "harmless/Model-X",
			aliases: []string{
				"default",
				"abliterated-fast",
			},
			want: []string{"Abliterated", "ablit"},
		},
		{
			name:    "multiple distinct hits, deduped and sorted",
			root:    "TheBloke/MythoMax-Wizard-NSFW-RP",
			aliases: []string{"WIZARD", "Toxic-merge"},
			want:    []string{"NSFW", "RP", "Toxic", "Wizard"},
		},
		{
			name: "empty input",
			root: "",
			want: nil,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := matchComplianceTags(c.root, c.aliases)
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("matchComplianceTags(%q, %v) = %v, want %v", c.root, c.aliases, got, c.want)
			}
		})
	}
}

func TestEnumerateAttachesComplianceTags(t *testing.T) {
	body := `{
		"object":"list",
		"data":[
			{"id":"cognitivecomputations/dolphin-2.9-llama3-8b","root":"cognitivecomputations/dolphin-2.9-llama3-8b"},
			{"id":"dolphin","root":"cognitivecomputations/dolphin-2.9-llama3-8b"}
		]
	}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(body))
	}))
	defer srv.Close()

	e := &Enumerator{EndpointURL: srv.URL, HTTPClient: srv.Client(), SkipHF: true}
	models, err := e.Enumerate(context.Background())
	if err != nil {
		t.Fatalf("Enumerate: %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("models = %d, want 1", len(models))
	}
	if !reflect.DeepEqual(models[0].Features.ComplianceTags, []string{"Dolphin"}) {
		t.Fatalf("ComplianceTags = %v", models[0].Features.ComplianceTags)
	}
}
