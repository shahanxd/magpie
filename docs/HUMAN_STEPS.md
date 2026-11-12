# What's left for a human -- simple steps

## Already done (no action needed)

- **The page judges open** -- article + all results, live and free:
 https://huggingface.co/spaces/shahanxd/magpie (the Article tab *is* the article)
- **Hardened model** with weights and a proper model card (CC-BY-NC-4.0):
 https://huggingface.co/shahanxd/fire_rr
- **Dataset card** corrected: https://huggingface.co/datasets/shahanxd/magpie-core-v1
 (this repo holds the final v2 Aya core; the "v1" name is just historical)
- Organizer answers: publishing is fine, and tagging teammates is fine (judges see a PII-masked copy).

---

## Your steps, in order

### 1. Put the code on `main` (important -- judges see `main`)
Right now `main` on GitHub is empty; all the work is on the branch `claude/sweet-dijkstra-7e7z0p`.
- Easiest: tell Claude **"open the PR"**, then click **Merge** on GitHub.
- Or yourself: open https://github.com/shahanxd/magpie/compare/main...claude/sweet-dijkstra-7e7z0p
 → **Create pull request** → **Merge pull request**.

### 2. Let judges see the dataset (2 clicks)
It is private, so judges get a 404. Open https://huggingface.co/datasets/shahanxd/magpie-core-v1/settings
- **Change repository visibility** → **Public**
- On the same page, **Access requests / Gated access** → **Enable**, approval **Automatic**.
 (The terms form -- affiliation + intended use -- is already in the card.)

### 3. Finish the collection (1 minute)
Open https://huggingface.co/collections/shahanxd/lexsi-6aedf40eefc742173da5acbc → **Add to collection**,
add the Space **`shahanxd/magpie`** and the model **`shahanxd/fire_rr`** (keep the dataset).
*(Your HF token can't edit collections, so this one is yours.)*

### 4. Article link for the submission form
HF articles now need PRO. Submit the Space as the article: **https://huggingface.co/spaces/shahanxd/magpie**.
Optional Discord check: "Articles need HF PRO -- is a Space page OK as our article?"

### 5. Native-speaker check (when you have a Hindi speaker)
Download: https://huggingface.co/datasets/shahanxd/magpie-core-v1/blob/main/native_review_15pct.csv
(click **Download**). Fill the `ok (y/n)` column, and `fixed_translation` where something reads
wrong. Send the CSV back to Claude to fold in and report the agreement rate.

### 6. Upstream PRs (C5) -- in the browser, no git needed (~3 min each)

**SafeTune**
1. Open https://github.com/Lexsi-Labs/SafeTune → **Fork** → **Create fork**.
2. In your fork: **Add file → Create new file**. Name it `src/safetune/multilingual_refusal.py`.
3. Paste the contents of
 https://github.com/shahanxd/magpie/blob/claude/sweet-dijkstra-7e7z0p/contrib/safetune_multilingual_refusal.py
 (click **Raw**, select all, copy).
4. **Commit changes…** → choose **Create a new branch** → **Propose changes**.
5. On the next page: **Create pull request**. Title and description: copy section **1** of
 https://github.com/shahanxd/magpie/blob/claude/sweet-dijkstra-7e7z0p/contrib/README.md

**AuditKIT** -- same steps with:
- repo https://github.com/Lexsi-Labs/AuditKIT
- new file `src/auditkit/redteam/probes/indic.py`
- contents of `contrib/auditkit_indic_redteam.py`, PR text from section **3** of `contrib/README.md`.

### 7. Laptop demo -- skipped for now.
