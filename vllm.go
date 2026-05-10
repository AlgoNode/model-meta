package modelmeta

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
)

// vllmModel mirrors the subset of the OpenAI /v1/models response that vLLM
// populates. Fields beyond id/root/owned_by are vLLM extensions and may be
// absent on other servers.
type vllmModel struct {
	ID          string `json:"id"`
	Object      string `json:"object"`
	OwnedBy     string `json:"owned_by"`
	Root        string `json:"root"`
	Parent      string `json:"parent"`
	MaxModelLen int    `json:"max_model_len"`
}

type vllmModelsResponse struct {
	Object string      `json:"object"`
	Data   []vllmModel `json:"data"`
}

// fetchVLLMModels calls GET {endpoint}/v1/models and decodes the response.
// endpoint may include the /v1 suffix or not; both are accepted.
func fetchVLLMModels(ctx context.Context, client *http.Client, endpoint, apiKey string) ([]vllmModel, error) {
	u, err := buildModelsURL(endpoint)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("vllm models request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<14))
		return nil, fmt.Errorf("vllm models: unexpected status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var out vllmModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("vllm models decode: %w", err)
	}
	return out.Data, nil
}

func buildModelsURL(endpoint string) (string, error) {
	endpoint = strings.TrimSpace(endpoint)
	if endpoint == "" {
		return "", fmt.Errorf("empty endpoint")
	}
	u, err := url.Parse(endpoint)
	if err != nil {
		return "", fmt.Errorf("parse endpoint: %w", err)
	}
	if u.Scheme == "" || u.Host == "" {
		return "", fmt.Errorf("endpoint missing scheme or host: %q", endpoint)
	}
	p := strings.TrimRight(u.Path, "/")
	switch {
	case strings.HasSuffix(p, "/v1/models"):
		// already complete
	case strings.HasSuffix(p, "/v1"):
		p += "/models"
	default:
		p += "/v1/models"
	}
	u.Path = p
	return u.String(), nil
}
