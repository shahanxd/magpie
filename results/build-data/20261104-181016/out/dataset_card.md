# Magpie core v1

Parallel red-team + over-refusal core for Magpie (Tiny Aya Fire robustness).

- records: **632** (286 harmful, 346 benign/over-refusal)
- forms: {<Language.EN: 'en'>: 400, 'hi-Deva': 44, 'hing': 188}
- categories: {'cyber': 40, 'harassment_hate': 32, 'illegal_goods': 33, 'misinformation': 26, 'non_violent_crime': 34, 'other': 29, 'over_refusal': 346, 'sexual': 31, 'violence': 32, 'weapons_bio_chem': 29}
- sources: {'harmbench_standard': 90, 'harmbench_standard+translated': 42, 'orbench_hard': 100, 'orbench_hard+translated': 63, 'strongreject': 110, 'strongreject+translated': 44, 'xstest_safe': 100, 'xstest_safe+translated': 83}

Harmful seeds: HarmBench (standard) and StrongREJECT. Over-refusal: XSTest-safe and OR-Bench-Hard. Hinglish and Devanagari by tiny-aya (v1 translator) with a back-translation agreement gate, pending native-speaker verification.

**Harmful prompts are gated.** Non-commercial safety research only (Tiny Aya CC-BY-NC-4.0 + Cohere Labs AUP; sources under their own licences).

Provenance: git `3149dd3`, translator `fire`.
