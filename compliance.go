package modelmeta

import (
	"regexp"
	"sort"
)

// compliancePattern pairs a display label with the case-insensitive regex
// used to detect it. The label is what ends up in Features.ComplianceTags;
// the regex is applied against the model root and each alias.
type compliancePattern struct {
	label string
	re    *regexp.Regexp
}

// compliancePatterns is the curated watchlist. Plain words match as
// case-insensitive substrings; entries using \b are word-boundary anchored to
// avoid false positives (e.g. RPCS3, ERPNext).
var compliancePatterns = []compliancePattern{
	{"Uncensored", regexp.MustCompile(`(?i)Uncensored`)},
	{"Abliterated", regexp.MustCompile(`(?i)Abliterated`)},
	{"ablit", regexp.MustCompile(`(?i)ablit`)},
	{"Toxic", regexp.MustCompile(`(?i)Toxic`)},
	{"Dolphin", regexp.MustCompile(`(?i)Dolphin`)},
	{"Lumimaid", regexp.MustCompile(`(?i)Lumimaid`)},
	{"Capybara", regexp.MustCompile(`(?i)Capybara`)},
	{"Hermes", regexp.MustCompile(`(?i)Hermes`)},
	{"OpenHermes", regexp.MustCompile(`(?i)OpenHermes`)},
	{"NSFW", regexp.MustCompile(`(?i)NSFW`)},
	{"RP", regexp.MustCompile(`(?i)\bRP\b`)},
	{"ERP", regexp.MustCompile(`(?i)\bERP\b`)},
	{"Unfiltered", regexp.MustCompile(`(?i)Unfiltered`)},
	{"Heretic", regexp.MustCompile(`(?i)Heretic`)},
	{"Aggressive", regexp.MustCompile(`(?i)Aggressive`)},
	{"Wizard", regexp.MustCompile(`(?i)Wizard`)},
}

// matchComplianceTags returns the labels whose pattern matches the root id or
// any of the aliases. Result is deduplicated and sorted for determinism.
func matchComplianceTags(root string, aliases []string) []string {
	if root == "" && len(aliases) == 0 {
		return nil
	}
	candidates := make([]string, 0, 1+len(aliases))
	if root != "" {
		candidates = append(candidates, root)
	}
	candidates = append(candidates, aliases...)

	seen := make(map[string]struct{})
	var out []string
	for _, p := range compliancePatterns {
		for _, c := range candidates {
			if p.re.MatchString(c) {
				if _, ok := seen[p.label]; !ok {
					seen[p.label] = struct{}{}
					out = append(out, p.label)
				}
				break
			}
		}
	}
	sort.Strings(out)
	return out
}
