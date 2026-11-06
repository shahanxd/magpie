# What needs a human — simple steps

Everything else is automated and committed. These five need you (links included). Order: **1 and 4
first** (they unblock the rest), then 2, 3, 5 any time before the Nov 14 deadline.

---

## 1. Publish to Hugging Face (the submission lives here)

You need a free HF account: https://huggingface.co/join

**a) The article.** Copy the text from `docs/ARTICLE.md` (in this repo) into a new HF article:
→ https://huggingface.co/new-blog  (title it "Magpie", paste the markdown, add the hero image
`docs/figures/hero_ladder.png`). Tag your teammates. *If the Lexsi hackathon page says to submit the
article somewhere specific, follow that instead — this is the generic HF route.*

**b) The dataset collection.** The core is already on HF (private): `shahanxd/magpie-core-v2`.
- Keep it **gated** (harmful completions must stay gated). It already is.
- Group it into a collection: → https://huggingface.co/collections → "New collection" → add the
  dataset.

**c) The hardened model (optional, allowed).** The RR-hardened `fire_rr` is a *defense* variant, so
it's fine to publish (only *attacked/abliterated* weights must stay private — we never publish
those).
- Get the files: open the Kaggle kernel https://www.kaggle.com/code/shahanxd/magpie-harden-rr →
  "Output" → download the `ckpt/fire_rr/` folder.
- New model repo: → https://huggingface.co/new (type: Model) → upload that folder.

**d) The Space (demo).** I'm building the files under `space/` in this repo.
- New Space: → https://huggingface.co/new-space → SDK **Gradio** → upload everything in `space/`
  (or point it at the GitHub repo's `space/` folder). It runs with no GPU.

**e) The GitHub repo.** Make https://github.com/shahanxd/magpie public (Settings → General →
"Change visibility" → Public) so judges can see the code.

---

## 2. Submit the upstream contributions (C5)

I'm preparing ready-to-apply patches under `contrib/` (a refusal matcher + script-aware BoN for
SafeTune, and an Indic red-team pack for AuditKIT). For each one:
1. Find the repo's GitHub URL (from its PyPI page — e.g. https://pypi.org/project/safetune/ ,
   https://pypi.org/project/auditkit/ — "Homepage"/"Source").
2. Fork it on GitHub (top-right "Fork").
3. In your fork: apply the patch (`git apply contrib/<name>.patch`), commit, push.
4. Open a Pull Request from your fork to the original repo. `contrib/README.md` has the exact
   commands and a suggested PR description.

*(These repos aren't in my GitHub access, so I can prepare the patches but can't open the PRs.)*

---

## 3. Native-speaker check (10–20 min, one Hindi speaker)

1. Download the review sheet from the gated dataset: it's `review_hing.csv` / the review sheet in
   `shahanxd/magpie-core-v2` on HF.
2. A Hindi/Hinglish speaker fills the `ok (y/n)` column (and `fixed_translation` where needed) —
   just checking the translations read naturally. Short session; skip anything uncomfortable.
3. Send me the filled CSV (or commit it) and I'll fold the corrections in and report the agreement
   rate.

---

## 4. Ask the organizers (Discord ticket)

In the Lexsi hackathon Discord, open a ticket and ask two things:
1. "Can we publish the **hardened** model weights? (We are **not** publishing attacked/abliterated
   weights.)" — our plan assumes hardened = yes, abliterated = no.
2. "Double-blind vs tagging teammates on the HF article — which do you want?"

*(I don't have the Discord invite link — use the one from your hackathon registration email /
the Lexsi hackathon page.)*

---

## 5. Laptop demo (optional, nice-to-have)

On your machine with the RTX 3050: run Tiny Aya Fire as a GGUF with llama.cpp
(https://github.com/ggml-org/llama.cpp) and show one L0 refusal + one prefill break, to demonstrate
the on-device threat live. I can write the exact commands if you want — this environment just can't
reach your laptop.

---

**TL;DR for today:** make the GitHub repo public (1e), paste `docs/ARTICLE.md` into an HF article
(1a), and drop the two questions in the organizer Discord (4). The rest can follow.
