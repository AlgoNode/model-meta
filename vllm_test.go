package modelmeta

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestBuildModelsURL(t *testing.T) {
	cases := []struct {
		in, want string
		err      bool
	}{
		{"http://localhost:8000", "http://localhost:8000/v1/models", false},
		{"http://localhost:8000/", "http://localhost:8000/v1/models", false},
		{"http://localhost:8000/v1", "http://localhost:8000/v1/models", false},
		{"http://localhost:8000/v1/", "http://localhost:8000/v1/models", false},
		{"http://localhost:8000/v1/models", "http://localhost:8000/v1/models", false},
		{"https://api.example.com/openai/v1", "https://api.example.com/openai/v1/models", false},
		{"", "", true},
		{"localhost:8000", "", true},
	}
	for _, c := range cases {
		got, err := buildModelsURL(c.in)
		if c.err {
			if err == nil {
				t.Errorf("buildModelsURL(%q) expected error, got %q", c.in, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("buildModelsURL(%q) unexpected err: %v", c.in, err)
			continue
		}
		if got != c.want {
			t.Errorf("buildModelsURL(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestFetchVLLMModelsSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer secret" {
			t.Errorf("missing auth: %q", got)
		}
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"m1","root":"org/m1","owned_by":"org","max_model_len":4096}]}`))
	}))
	defer srv.Close()

	models, err := fetchVLLMModels(context.Background(), srv.Client(), srv.URL, "secret")
	if err != nil {
		t.Fatalf("fetchVLLMModels: %v", err)
	}
	if len(models) != 1 || models[0].ID != "m1" || models[0].Root != "org/m1" || models[0].MaxModelLen != 4096 {
		t.Fatalf("unexpected payload: %+v", models)
	}
}

func TestFetchVLLMModelsNon2xx(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()

	_, err := fetchVLLMModels(context.Background(), srv.Client(), srv.URL, "")
	if err == nil || !strings.Contains(err.Error(), "status 500") {
		t.Fatalf("expected status error, got %v", err)
	}
}
