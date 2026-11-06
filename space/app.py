"""Magpie — an interactive view of the results. Hugging Face Space (Gradio).

Reads only `data.json` and `hero_ladder.png` (both committed beside this file),
so it runs on Hugging Face with no GPU and no dependency on the full repo.
Regenerate the data with `python space/build_space_data.py` from the repo root.
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "data.json").read_text(encoding="utf-8"))
HERO = str(HERE / "hero_ladder.png")
_ART = HERE / "article.md"
ARTICLE = _ART.read_text(encoding="utf-8") if _ART.is_file() else ""

MODEL_ORDER = ["fire", "global", "fire_rr"]
MODEL_LABEL = {"fire": "Fire (vendor)", "global": "Global (vendor)", "fire_rr": "Fire + RR (hardened)"}
LEVEL_ORDER = ["L0 direct", "L1 templates", "L1 best-of-N", "L2 prefill", "L3 abliteration"]


def _ladder_table() -> tuple[list[list], list[str]]:
    """ASR (HarmBench) by model × attacker level — wide table."""
    by_model: dict[str, dict[str, float]] = {}
    for p in DATA["ladder"]:
        by_model.setdefault(p["model"], {})[p["level_label"]] = p["asr"]
    rows = []
    for m in MODEL_ORDER:
        if m not in by_model:
            continue
        rows.append([MODEL_LABEL.get(m, m)] + [
            (f"{by_model[m][lv]:.2f}" if lv in by_model[m] else "—") for lv in LEVEL_ORDER
        ])
    return rows, ["model ＼ attacker level"] + LEVEL_ORDER


def _flip_rows(panel: str, key: str) -> tuple[list[list], list[str]]:
    d = DATA[panel]
    rows = [[r[key], f"{r['lexicon']:.2f}", f"{r['harmbench']:.2f}"] for r in d["rows"]]
    return rows, [key, "refusal lexicon", "HarmBench (validated)"]


def _hardening_rows() -> tuple[list[list], list[str]]:
    rows = [[r["level"], f"{r['fire']:.2f}", f"{r['fire_rr']:.2f}"] for r in DATA["hardening"]["rows"]]
    return rows, ["attacker level", "Fire (HarmBench)", "fire_rr (HarmBench)"]


def _h4_rows() -> tuple[list[list], list[str]]:
    rows = [[r["attack"], f"{r['hazard_ratio']:.2f}", r["ci"], r["p"]] for r in DATA["h4"]["rows"]]
    return rows, ["attack", "hazard ratio / SD", "95% CI", "p"]


CSS = "footer{visibility:hidden} .md-note{color:#666;font-size:0.92em}"

with gr.Blocks(title="Magpie — pricing robustness", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🐦 Magpie — pricing the robustness of an open, on-device safety model")
    gr.Markdown(DATA["thesis"])
    gr.Markdown(
        f"[📄 Code]({DATA['links']['code']}) · [🗂️ Dataset (gated)]({DATA['links']['dataset']}) · "
        "Tiny Aya (Cohere Labs), English + Hinglish + Devanagari. "
        "<span class='md-note'>Every number is HarmBench-content-validated.</span>"
    )

    if ARTICLE:
        with gr.Tab("📄 Article"):
            gr.Markdown(ARTICLE)

    with gr.Tab("Cost ladder"):
        gr.Markdown("**How much does the next break cost?** Attack success rate (content-validated) "
                    "as the attacker gains capability — just ask (L0) → public jailbreaks (L1) → "
                    "controls the decoder (L2) → edits the weights (L3).")
        gr.Image(HERO, show_label=False, container=False)
        r, h = _ladder_table()
        gr.Dataframe(value=r, headers=h, interactive=False, wrap=True)
        gr.Markdown("<span class='md-note'>Fire and fire_rr overlap — hardening buys nothing. The "
                    "L3 jump is the weight-access cliff: safe to ask, broken to edit.</span>")

    with gr.Tab("Judge validity (the headline)"):
        gr.Markdown("### Within English — the judge flips the attack ranking")
        gr.Markdown(DATA["judge_flip_en"]["caption"])
        r, h = _flip_rows("judge_flip_en", "attack")
        gr.Dataframe(value=r, headers=h, interactive=False)
        gr.Markdown("### Across languages — the judge flips the ranking of *forms*")
        gr.Markdown(DATA["judge_flip_lang"]["caption"])
        r, h = _flip_rows("judge_flip_lang", "form")
        gr.Dataframe(value=r, headers=h, interactive=False)
        gr.Markdown("<span class='md-note'>You cannot compare robustness across languages — or rank "
                    "defenses — with a cheap refusal lexicon.</span>")

    with gr.Tab("Does hardening help?"):
        gr.Markdown(DATA["hardening"]["caption"])
        r, h = _hardening_rows()
        gr.Dataframe(value=r, headers=h, interactive=False)
        gr.Markdown("<span class='md-note'>An honest negative: a defense that looks plausible (loss "
                    "drops, L0 clean) yet shifts no survival curve. Validate defenses by "
                    "re-attacking and measuring cost, not by training loss.</span>")

    with gr.Tab("Audit before you attack (H4)"):
        gr.Markdown(DATA["h4"]["caption"])
        r, h = _h4_rows()
        gr.Dataframe(value=r, headers=h, interactive=False)
        gr.Markdown("<span class='md-note'>A cheap, attack-free white-box margin predicts "
                    "cost-to-break — so a red-team budget can be focused before any attack runs.</span>")

    with gr.Tab("Robustness cards"):
        choice = gr.Dropdown(choices=[MODEL_LABEL[m] for m in MODEL_ORDER if m in DATA["cards"]],
                             value=MODEL_LABEL["fire"], label="Model")
        card = gr.Markdown(DATA["cards"].get("fire", ""))
        _label_to_key = {v: k for k, v in MODEL_LABEL.items()}

        def _show(label: str) -> str:
            return DATA["cards"].get(_label_to_key.get(label, ""), "")

        choice.change(_show, inputs=choice, outputs=card)

    gr.Markdown(f"<span class='md-note'>{DATA['source']}</span>")

if __name__ == "__main__":
    demo.launch()
