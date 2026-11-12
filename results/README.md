# results/

Run outputs land here, one directory per run (`<job>/<timestamp>/`). Everything
under `results/` is **gitignored** except this README: runs are reproducible
from a config + git SHA (see `magpie.config.provenance`), so the tree is not
source, and **harmful completions never enter git** -- they stay here locally
and in the gated HF dataset only (PLAN §16).

Layout of a run directory:

- `_run.json` -- the runner's record (kernel ref, version, git SHA, status, timings).
- `summary.json` / `summary.md` -- human-readable summary of the run.
- `table.json` / `table.md` -- the aggregated result table (ASR with CI, median
  cost-to-break, over-refusal). Carries rates and costs, no prompt/response text.
- `outcomes_<model>.jsonl` -- one survival `Outcome` per prompt (no text).
- `attempts_notext_<model>.jsonl` -- attempts with judgements but prompt/response
  stripped (committable if ever needed).
- `raw/` -- full attempts/records **with** prompt and response text; the payload
  for the gated HF dataset. Never committed.
- `logs/` -- captured subprocess logs from the kernel.

To publish a committable table from a run, force-add just that file, e.g.
`git add -f results/l0/<ts>/table.md`.
