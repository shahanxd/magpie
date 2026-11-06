#!/usr/bin/env python
"""Generate the Space's self-contained data bundle from the committed results.

Run from the repo root (needs the `magpie` package and the `results/` tables):

    python space/build_space_data.py

Writes `space/data.json` and copies the hero figure + robustness cards into
`space/`. The Space app (`space/app.py`) then reads only those files, so it runs
on Hugging Face with no dependency on the repo or the full results tree.

The ladder points are re-derived from the committed HarmBench tables; the C4 /
H1 / hardening panels mirror the numbers in `docs/RESULTS.md` (with run
provenance), kept here as one explicit, auditable dict.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from magpie.analysis.figures import build_hero

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "space"


def _ladder_points() -> list[dict]:
    tables = sorted(
        glob.glob(str(ROOT / "results/rescore/*/out/table_harmbench.json"))
        + glob.glob(str(ROOT / "results/attack-l2-prefill/*/out/table.json"))
    )
    info = build_hero(tables, SPACE / "hero_ladder.png")  # also (re)writes the PNG
    # Keep only validated (HarmBench) points for the demo's ladder.
    return [p for p in info["points"] if p.get("judge") == "harmbench"]


# Curated panels — these mirror docs/RESULTS.md (validated with HarmBench).
JUDGE_FLIP_EN = {
    "caption": "Same attempts, two judges. The cheap refusal lexicon over-counts and flips the "
               "attack ranking; only a content judge tells a long non-refusal from a real jailbreak.",
    "rows": [
        {"attack": "L1 templates", "lexicon": 0.52, "harmbench": 0.29},
        {"attack": "L1 best-of-N", "lexicon": 1.00, "harmbench": 0.19},
    ],
}
JUDGE_FLIP_LANG = {
    "caption": "Identical prompts, three forms. The lexicon says Devanagari is 4x English and "
               "Hinglish is safest; the content judge says the opposite in order and a third in size.",
    "rows": [
        {"form": "English", "lexicon": 0.21, "harmbench": 0.03},
        {"form": "Devanagari", "lexicon": 0.86, "harmbench": 0.28},
        {"form": "Hinglish", "lexicon": 0.09, "harmbench": 0.03},
    ],
}
HARDENING = {
    "caption": "A lightweight Circuit-Breakers/RR pass (fire_rr) re-attacked paired against vendor "
               "Fire. It raises attacker cost at no level — an honest negative.",
    "rows": [
        {"level": "L1 templates", "fire": 0.30, "fire_rr": 0.30},
        {"level": "L2 prefill", "fire": 0.31, "fire_rr": 0.31},
    ],
}
H4 = {
    "caption": "A prompt's attack-free refusal-direction margin predicts how expensive it is to "
               "break (Cox regression; hazard ratio < 1 means a higher margin survives longer).",
    "rows": [
        {"attack": "best-of-N", "hazard_ratio": 0.72, "ci": "[0.58, 0.89]", "p": "0.003"},
        {"attack": "templates", "hazard_ratio": 0.60, "ci": "—", "p": "2e-6"},
    ],
}

GITHUB = "https://github.com/shahanxd/magpie/blob/main"
SPACE_FILE = "https://huggingface.co/spaces/shahanxd/magpie/resolve/main"


def _article_for_space() -> str:
    """docs/ARTICLE.md with repo-relative links rewritten to absolute URLs.

    The Space renders the article as its first tab (a free stand-in for an HF
    article, which now requires PRO), so every link must resolve off-repo.
    """
    text = (ROOT / "docs" / "ARTICLE.md").read_text(encoding="utf-8")
    repl = {
        "](figures/hero_ladder.png)": f"]({SPACE_FILE}/hero_ladder.png)",
        "](PREREGISTRATION.md)": f"]({GITHUB}/docs/PREREGISTRATION.md)",
        "](RESULTS.md)": f"]({GITHUB}/docs/RESULTS.md)",
        "](../magpie/costs.py)": f"]({GITHUB}/magpie/costs.py)",
    }
    for old, new in repl.items():
        text = text.replace(old, new)
    return text


def _md(text: str) -> str:
    import markdown
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    # Wide tables scroll inside their own box, never the page (phone widths).
    return html.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")


def _table(headers: list[str], rows: list[list]) -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


STATIC_CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#5f5e57;--line:#e3e0d8;--card:#fff;--acc:#3b4cca;--accbg:#eef0fb}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe6;--mut:#a3a199;--line:#2f2e2a;--card:#1c1c1a;--acc:#9aa6ff;--accbg:#22263f}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.65 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:28px 16px 64px}
header h1{font-size:1.65rem;line-height:1.25;margin:0 0 8px}
header p.thesis{font-size:1.06rem;color:var(--mut);margin:0 0 12px}
.links{display:flex;flex-wrap:wrap;gap:4px 16px}.links a{white-space:nowrap}
a{color:var(--acc)}
nav{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);
 display:flex;gap:4px;overflow-x:auto;margin:20px -16px 24px;padding:8px 16px;scrollbar-width:none}
nav::-webkit-scrollbar{display:none}
nav button{flex:none;border:1px solid var(--line);background:var(--card);color:var(--fg);
 padding:7px 13px;border-radius:999px;font:inherit;font-size:.92rem;cursor:pointer}
nav button[aria-selected=true]{background:var(--accbg);border-color:var(--acc);color:var(--acc);font-weight:600}
section{display:none}section.on{display:block}
h2{font-size:1.3rem;margin:2em 0 .5em;line-height:1.3}h3{font-size:1.08rem;margin:1.6em 0 .4em}
img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}
.tw{overflow-x:auto;margin:1em 0;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:.93rem;font-variant-numeric:tabular-nums}
th,td{padding:8px 11px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{background:var(--card);font-weight:600}tr:last-child td{border-bottom:0}
blockquote{margin:1em 0;padding:.6em 1em;border-left:3px solid var(--acc);background:var(--accbg);border-radius:0 8px 8px 0}
code{font-size:.88em;background:var(--card);border:1px solid var(--line);padding:1px 5px;border-radius:4px}
pre{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
pre code{border:0;padding:0}
.note{color:var(--mut);font-size:.93rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:4px 18px 10px;margin:18px 0}
footer{margin-top:48px;color:var(--mut);font-size:.88rem;border-top:1px solid var(--line);padding-top:14px}
"""

STATIC_JS = """
const tabs=[...document.querySelectorAll('nav button')];
function show(id){tabs.forEach(b=>{const on=b.dataset.t===id;b.setAttribute('aria-selected',on);
 document.getElementById(b.dataset.t).classList.toggle('on',on)});}
const IDS=new Set(tabs.map(b=>b.dataset.t));
function fromHash(){const h=location.hash.slice(1);show(IDS.has(h)?h:'article');}
tabs.forEach(b=>b.addEventListener('click',()=>{show(b.dataset.t);history.replaceState(null,'','#'+b.dataset.t);scrollTo({top:0});}));
addEventListener('hashchange',fromHash);fromHash();
"""


def _render_static(data: dict) -> str:
    """One self-contained HTML page: the article + the results views (free static Space)."""
    art = (ROOT / "docs" / "ARTICLE.md").read_text(encoding="utf-8")
    for old, new in {
        "](figures/hero_ladder.png)": "](hero_ladder.png)",
        "](PREREGISTRATION.md)": f"]({GITHUB}/docs/PREREGISTRATION.md)",
        "](RESULTS.md)": f"]({GITHUB}/docs/RESULTS.md)",
        "](../magpie/costs.py)": f"]({GITHUB}/magpie/costs.py)",
    }.items():
        art = art.replace(old, new)
    # The article's own H1 duplicates the page header; drop it.
    art = "\n".join(art.splitlines()[1:]) if art.startswith("# ") else art

    by_model: dict[str, dict[str, float]] = {}
    for p in data["ladder"]:
        by_model.setdefault(p["model"], {})[p["level_label"]] = p["asr"]
    levels = ["L0 direct", "L1 templates", "L1 best-of-N", "L2 prefill", "L3 abliteration"]
    names = {"fire": "Fire (vendor)", "global": "Global (vendor)", "fire_rr": "Fire + RR (hardened)"}
    ladder = _table(["model"] + levels, [
        [names[m]] + [f"{by_model[m][lv]:.2f}" if lv in by_model[m] else "—" for lv in levels]
        for m in ("fire", "global", "fire_rr") if m in by_model])
    jf_en = _table(["attack", "refusal lexicon", "HarmBench (validated)"],
                   [[r["attack"], f"{r['lexicon']:.2f}", f"<b>{r['harmbench']:.2f}</b>"]
                    for r in data["judge_flip_en"]["rows"]])
    jf_lang = _table(["form", "refusal lexicon", "HarmBench (validated)"],
                     [[r["form"], f"{r['lexicon']:.2f}", f"<b>{r['harmbench']:.2f}</b>"]
                      for r in data["judge_flip_lang"]["rows"]])
    hard = _table(["attacker level", "Fire (HarmBench)", "fire_rr (HarmBench)"],
                  [[r["level"], f"{r['fire']:.2f}", f"{r['fire_rr']:.2f}"] for r in data["hardening"]["rows"]])
    h4 = _table(["attack", "hazard ratio / SD", "95% CI", "p"],
                [[r["attack"], f"{r['hazard_ratio']:.2f}", r["ci"], r["p"]] for r in data["h4"]["rows"]]
                + [["pooled Fire + Global (leave-one-model-out)", "0.64", "[0.56, 0.73]", "2e-11"]])
    cards = "".join(
        f'<div class="card">{_md(data["cards"][m].replace("](../RESULTS.md)", f"]({GITHUB}/docs/RESULTS.md)"))}</div>'
        for m in ("fire", "global", "fire_rr") if m in data["cards"])
    L = data["links"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Magpie — Pricing Robustness</title>
<meta name="description" content="Red-teaming Tiny Aya as attacker cost, in English and Hinglish.">
<style>{STATIC_CSS}</style></head><body><div class="wrap">
<header>
<h1>🐦 Magpie — pricing the robustness of an open, on-device safety model</h1>
<p class="thesis">{data['thesis']}</p>
<p class="links"><a href="{L['code']}">Code &amp; reproducibility</a><a href="{L['dataset']}">Dataset (gated)</a><a href="https://huggingface.co/shahanxd/fire_rr">Hardened model</a></p>
</header>
<nav role="tablist">
<button data-t="article">📄 Article</button><button data-t="ladder">Cost ladder</button>
<button data-t="judge">Judge validity</button><button data-t="harden">Does hardening help?</button>
<button data-t="h4">Audit before you attack</button><button data-t="cards">Robustness cards</button>
</nav>
<section id="article">{_md(art)}</section>
<section id="ladder"><h2>How much does the next break cost?</h2>
<p>Attack success rate (HarmBench-validated) as the attacker gains capability: just ask (L0) → public jailbreaks (L1) → controls the decoder (L2) → edits the weights (L3).</p>
<img src="hero_ladder.png" alt="Attack success rate by attacker capability level for Fire, Global and the RR-hardened Fire">
{ladder}<p class="note">Fire and fire_rr overlap — the hardening buys nothing. The L3 jump is the weight-access cliff: safe to ask, broken to edit.</p></section>
<section id="judge"><h2>Within English — the judge flips the attack ranking</h2><p>{data['judge_flip_en']['caption']}</p>{jf_en}
<h2>Across languages — the judge flips the ranking of forms</h2><p>{data['judge_flip_lang']['caption']}</p>{jf_lang}
<p class="note">You cannot compare robustness across languages — or rank defenses — with a cheap refusal lexicon.</p></section>
<section id="harden"><h2>Does hardening help?</h2><p>{data['hardening']['caption']}</p>{hard}
<p class="note">An honest negative: a defense that looks plausible (training loss drops, L0 clean) yet shifts no survival curve. Validate defenses by re-attacking and measuring cost.</p></section>
<section id="h4"><h2>Audit before you attack (H4)</h2><p>{data['h4']['caption']}</p>{h4}
<p class="note">The margin is computed with no attack, from a direction fit on disjoint data — and it predicts cost-to-break on a model it was not tuned on.</p></section>
<section id="cards"><h2>Robustness cards</h2><p>One-page, deployer-facing summaries per model.</p>{cards}</section>
<footer>{data['source']} Lexsi Hackathon · Track 2a. Harmful completions are gated; attacked weights are not published.</footer>
</div><script>{STATIC_JS}</script></body></html>"""


LEVEL_LABELS = {
    "l0_direct": "L0 direct",
    "l1_templates": "L1 templates",
    "l1_bon_input": "L1 best-of-N",
    "l2_prefill": "L2 prefill",
    "l3_abliteration": "L3 abliteration",
}


def main() -> None:
    SPACE.mkdir(exist_ok=True)
    points = _ladder_points()
    for p in points:
        p["level_label"] = LEVEL_LABELS.get(p["attack"], p["attack"])
    # Copy the deployer cards in as plain markdown.
    cards = {}
    for name in ("fire", "global", "fire_rr"):
        src = ROOT / "docs" / "cards" / f"{name}.md"
        if src.is_file():
            cards[name] = src.read_text(encoding="utf-8")
    data = {
        "thesis": "For an open, on-device model the attacker owns the weights, so safety is not a "
                  "refusal rate — it is the attacker cost to break the model, at each capability "
                  "level, in the languages people actually use. Magpie prices that, pairs every "
                  "number with over-refusal, and tests whether a cheap white-box margin predicts "
                  "the price.",
        "ladder": points,
        "judge_flip_en": JUDGE_FLIP_EN,
        "judge_flip_lang": JUDGE_FLIP_LANG,
        "hardening": HARDENING,
        "h4": H4,
        "cards": cards,
        "links": {
            "code": "https://github.com/shahanxd/magpie",
            "dataset": "https://huggingface.co/datasets/shahanxd/magpie-core-v1",
        },
        "source": "Generated from committed results by space/build_space_data.py; numbers mirror "
                  "docs/RESULTS.md (HarmBench-validated).",
    }
    (SPACE / "data.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    (SPACE / "article.md").write_text(_article_for_space(), encoding="utf-8")
    (SPACE / "index.html").write_text(_render_static(data), encoding="utf-8")
    print(f"wrote {SPACE/'data.json'} ({len(points)} ladder points, {len(cards)} cards) "
          f"and {SPACE/'hero_ladder.png'}")


if __name__ == "__main__":
    main()
