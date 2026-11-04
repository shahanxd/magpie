"""A single harm-category taxonomy, so sources stratify on the same axis.

HarmBench's ``SemanticCategory`` and StrongREJECT's ``category`` use different
label sets. We map both onto one taxonomy so the parallel core can be balanced
across categories and so category appears as a covariate in the Cox model
(PLAN §9) consistently across sources. Unknown labels map to ``other`` rather
than being dropped.
"""

from __future__ import annotations

# Unified categories (deliberately coarse; finer labels live in covariates).
CATEGORIES: tuple[str, ...] = (
    "weapons_bio_chem",
    "cyber",
    "illegal_goods",
    "non_violent_crime",
    "fraud_deception",
    "misinformation",
    "harassment_hate",
    "violence",
    "sexual",
    "privacy",
    "other",
)

_MAP: dict[str, str] = {
    # HarmBench SemanticCategory
    "cybercrime_intrusion": "cyber",
    "chemical_biological": "weapons_bio_chem",
    "illegal": "illegal_goods",
    "misinformation_disinformation": "misinformation",
    "harassment_bullying": "harassment_hate",
    "harmful": "other",
    "copyright": "other",
    # StrongREJECT category
    "illegal goods and services": "illegal_goods",
    "hate, harassment and discrimination": "harassment_hate",
    "non-violent crimes": "non_violent_crime",
    "disinformation and deception": "misinformation",
    "violence": "violence",
    "sexual content": "sexual",
    # common synonyms
    "privacy": "privacy",
    "fraud": "fraud_deception",
    "deception": "fraud_deception",
    "cybercrime": "cyber",
}


def normalize_category(raw: str | None) -> str:
    """Map a source category label to the unified taxonomy (``other`` if unknown)."""
    if not raw:
        return "other"
    key = raw.strip().lower()
    if key in _MAP:
        return _MAP[key]
    if key in CATEGORIES:
        return key
    return "other"
