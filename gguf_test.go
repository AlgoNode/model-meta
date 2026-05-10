package modelmeta

import (
	"testing"
)

func TestIsGGUFRepo(t *testing.T) {
	cases := []struct {
		name string
		info *hfModelInfo
		want bool
	}{
		{name: "nil", info: nil, want: false},
		{name: "no signal", info: &hfModelInfo{}, want: false},
		{name: "library_name", info: &hfModelInfo{LibraryName: "GGUF"}, want: true},
		{name: "info tag", info: &hfModelInfo{Tags: []string{"transformers", "gguf"}}, want: true},
		{name: "cardData tag", info: &hfModelInfo{CardData: hfCardData{Tags: []string{"GGUF"}}}, want: true},
		{name: "sibling filename", info: &hfModelInfo{Siblings: []hfSibling{{Filename: "model-Q4_K_M.gguf"}}}, want: true},
		{name: "non-gguf sibling", info: &hfModelInfo{Siblings: []hfSibling{{Filename: "config.json"}}}, want: false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := isGGUFRepo(c.info); got != c.want {
				t.Errorf("isGGUFRepo = %v, want %v", got, c.want)
			}
		})
	}
}

func TestQuantFromGGUF(t *testing.T) {
	cases := map[string]string{
		"Meta-Llama-3-8B-Instruct.Q4_K_M.gguf":       "q4_k_m",
		"meta-llama-3-8b-instruct-Q5_K_S.gguf":       "q5_k_s",
		"some.model.Q8_0.gguf":                       "q8_0",
		"Foo-IQ2_XXS.gguf":                           "iq2_xxs",
		"foo-IQ4_NL.gguf":                            "iq4_nl",
		"Bar.F16.gguf":                               "f16",
		"baz-BF16.gguf":                              "bf16",
		"qux.F32.gguf":                               "f32",
		"no-quant-here.gguf":                         "",
		"Model-Q4_K_M.gguf-not-actually-gguf":        "",
		"Model-Q4_K_M-00001-of-00002.gguf":           "q4_k_m",
	}
	for in, want := range cases {
		if got := quantFromGGUF(in); got != want {
			t.Errorf("quantFromGGUF(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestPickCanonicalGGUF(t *testing.T) {
	siblings := []hfSibling{
		{Filename: "README.md"},
		{Filename: "config.json"},
		{Filename: "Model.Q4_K_M.gguf", Size: 4_500_000_000},
		{Filename: "Model.Q5_K_M.gguf", Size: 5_500_000_000},
		{Filename: "Model.Q8_0.gguf", Size: 8_500_000_000},
		{Filename: "Model.IQ3_XXS.gguf", Size: 3_000_000_000},
	}

	t.Run("vllm id pins quant", func(t *testing.T) {
		got := pickCanonicalGGUF(siblings, "TheBloke/Model-GGUF-Q5_K_M")
		if got != "Model.Q5_K_M.gguf" {
			t.Errorf("got %q", got)
		}
	})
	t.Run("preference picks q4_k_m by default", func(t *testing.T) {
		got := pickCanonicalGGUF(siblings, "TheBloke/Model-GGUF")
		if got != "Model.Q4_K_M.gguf" {
			t.Errorf("got %q", got)
		}
	})
	t.Run("split shards: only first kept", func(t *testing.T) {
		s := []hfSibling{
			{Filename: "Model-Q4_K_M-00001-of-00003.gguf"},
			{Filename: "Model-Q4_K_M-00002-of-00003.gguf"},
			{Filename: "Model-Q4_K_M-00003-of-00003.gguf"},
		}
		got := pickCanonicalGGUF(s, "")
		if got != "Model-Q4_K_M-00001-of-00003.gguf" {
			t.Errorf("got %q", got)
		}
	})
	t.Run("no .gguf siblings", func(t *testing.T) {
		got := pickCanonicalGGUF([]hfSibling{{Filename: "config.json"}}, "")
		if got != "" {
			t.Errorf("got %q", got)
		}
	})
	t.Run("only unrecognized quant: largest by size", func(t *testing.T) {
		s := []hfSibling{
			{Filename: "weird-a.gguf", Size: 100},
			{Filename: "weird-b.gguf", Size: 500},
		}
		got := pickCanonicalGGUF(s, "")
		if got != "weird-b.gguf" {
			t.Errorf("got %q", got)
		}
	})
}

// TestExtractFeaturesQuantPriority pins down the resolution order in
// extractFeatures so future refactors don't silently change it.
func TestExtractFeaturesQuantPriority(t *testing.T) {
	cases := []struct {
		name string
		info *hfModelInfo
		id   string
		want string
	}{
		{
			name: "explicit AWQ wins over torch_dtype and GGUF",
			info: &hfModelInfo{
				Tags:     []string{"gguf"},
				Config:   hfConfig{TorchDtype: "bfloat16", QuantizationConfig: hfQuantConfig{QuantMethod: "awq"}},
				Siblings: []hfSibling{{Filename: "Model.Q4_K_M.gguf"}},
			},
			id:   "x/y",
			want: "awq",
		},
		{
			name: "NVFP4 from compressed-tensors format",
			info: &hfModelInfo{Config: hfConfig{QuantizationConfig: hfQuantConfig{QuantMethod: "compressed-tensors", Format: "nvfp4-pack-quantized"}}},
			id:   "x/y",
			want: "nvfp4",
		},
		{
			name: "GGUF wins over torch_dtype when no explicit method",
			info: &hfModelInfo{
				Tags:     []string{"gguf"},
				Config:   hfConfig{TorchDtype: "bfloat16"},
				Siblings: []hfSibling{{Filename: "Model.Q4_K_M.gguf"}},
			},
			id:   "x/y",
			want: "q4_k_m",
		},
		{
			name: "torch_dtype only",
			info: &hfModelInfo{Config: hfConfig{TorchDtype: "bfloat16"}},
			id:   "x/y",
			want: "bf16",
		},
		{
			name: "id suffix when nothing on info",
			info: &hfModelInfo{},
			id:   "Some/Model-FP8",
			want: "fp8",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f, _ := extractFeatures(c.info, c.id)
			if f.Quantization != c.want {
				t.Errorf("Quantization = %q, want %q", f.Quantization, c.want)
			}
		})
	}
}
