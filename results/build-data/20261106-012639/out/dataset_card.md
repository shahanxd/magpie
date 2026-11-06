# Magpie core v1

Parallel red-team + over-refusal core for Magpie (Tiny Aya Fire robustness).

- records: **1200** (600 harmful, 600 benign/over-refusal)
- forms: {<Language.EN: 'en'>: 400, 'hi-Deva': 400, 'hing': 400}
- categories: {'cyber': 69, 'harassment_hate': 69, 'illegal_goods': 69, 'misinformation': 66, 'non_violent_crime': 66, 'other': 63, 'over_refusal': 600, 'sexual': 66, 'violence': 66, 'weapons_bio_chem': 66}
- sources: {'harmbench_standard': 90, 'harmbench_standard+translated': 180, 'orbench_hard': 100, 'orbench_hard+translated': 200, 'strongreject': 110, 'strongreject+translated': 220, 'xstest_safe': 100, 'xstest_safe+translated': 200}

Harmful seeds: HarmBench (standard) and StrongREJECT. Over-refusal: XSTest-safe and OR-Bench-Hard. Hinglish and Devanagari by tiny-aya (v1 translator) with a back-translation agreement gate, pending native-speaker verification.

**Harmful prompts are gated.** Non-commercial safety research only (Tiny Aya CC-BY-NC-4.0 + Cohere Labs AUP; sources under their own licences).

Provenance: git `ae8ae2e`, translator `aya`.
