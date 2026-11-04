"""Hinglish: code-mixing measurement and natural spelling variation.

This module underpins contribution **C2** — *ordinary Hinglish typing is itself
a best-of-N attack*. Where the standard BoN attack perturbs a prompt with
synthetic noise (random capitals, ASCII flips, scrambles), Magpie samples the
variation real Hinglish users produce:

* **No standard spelling.** ``nahi`` / ``nahin`` / ``nhi`` / ``nahee`` all occur.
* **Transliteration freedom.** ``w↔v`` (``world``/``vorld``), ``ee↔i``, ``aa↔a``,
  ``oo↔u``, ``ph↔f``, geminate collapse (``accha``/``acha``).
* **SMS shorthand.** vowel dropping (``kar``→``kr``, ``bata``→``bta``), ``hai``→``h``.
* **Code-mix ratio shifts.** swapping content words between Hindi and English.

Everything here is pure-Python, deterministic under a seed, and operates on
arbitrary text, so it is covered by the CPU test-suite. The romanized-Hindi
lexicon is intentionally small and curated (the plan has it LLM-drafted and
team-verified); it is a seed that the data team extends, not a claim of
coverage. No external NLP dependency is required.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Romanized-Hindi lexicon (for language tagging and CMI). Curated, benign,
# high-frequency function and content words. Extend via the data pipeline.
# ---------------------------------------------------------------------------

HI_FUNCTION_WORDS: frozenset[str] = frozenset("""
main mai mein me hum ham tum tu aap aapko mujhe mujhko humein tumhe tumhein usko iska uska
hai hain ho hu hoon hunga hogi hoga honge tha thi the tha raha rahi rahe raha karo karna karne
kiya kiye karke kar kr liye liya diya dena lena hona jana aana gaya gayi kya kyun kyon kaise
kaisa kaisi kahan kaha kab kitna kitni kitne kaun kon jo joh woh wo yeh ye is us in un na nahi
nahin nhi mat haan han ji bhi toh to hi bas sirf aur ya par lekin magar kyunki agar warna
phir fir ab abhi tab jab tak se ko ka ki ke mera meri mere tera teri tere apna apni apne sab
sabko kuch kuchh thoda bahut bohot zyada kam accha acha theek thik sahi galat chalo chaliye
dekho dekhna suno batao bata bol bolo samajh samjha pata malum zaroor zaroorat chahiye chahta
chahti wala wali waale waala ek do teen char paanch bina saath sath andar bahar upar niche aage
peeche pehle baad waqt samay din raat subah shaam
""".split())

HI_CONTENT_WORDS: frozenset[str] = frozenset("""
khana khaana paani pani ghar gharwale dost dukaan dukan kaam naukri paisa paise rupaye rupay
shaadi parivaar pyaar mohabbat dil dimaag sar haath pair aankh naam zindagi jaan duniya desh
gaana gaon sheher raasta rasta safar yaatra kitaab padhai padhna likhna seekhna sikhna madad
sawaal jawaab baat baatein cheez cheezein tareeka tarika kahani kahaani mausam garmi sardi
baarish dhoop chai doodh dudh sabzi roti daal chawal mithai tyohar
""".split())

HI_WORDS: frozenset[str] = HI_FUNCTION_WORDS | HI_CONTENT_WORDS

# Romanized Hindi words that are also common English words. A per-token lexicon
# tagger cannot resolve "is"/"in"/"the"/"main" from the word alone, so it would
# wrongly flag plain English as code-mixed. We treat these as English by default
# (the conservative choice: Hindi detection leans on unambiguous markers like
# nahi/hai/kar/ghar). This is a known CMI pitfall for romanized text; the data
# pipeline's verified samples carry per-token language tags when precision matters.
HI_AMBIGUOUS: frozenset[str] = frozenset("""
main me the to is in us do hi na par char pair ho tha he she bas kar
""".split())

# A tiny bilingual content-word map for code-mix ratio shifts (en -> common hing).
EN_TO_HI: dict[str, str] = {
    "water": "paani", "food": "khana", "house": "ghar", "home": "ghar", "friend": "dost",
    "friends": "dost", "money": "paisa", "work": "kaam", "job": "naukri", "book": "kitaab",
    "books": "kitaab", "study": "padhai", "help": "madad", "question": "sawaal",
    "answer": "jawaab", "thing": "cheez", "way": "tareeka", "story": "kahani", "life": "zindagi",
    "world": "duniya", "country": "desh", "city": "sheher", "road": "rasta", "weather": "mausam",
    "rain": "baarish", "tea": "chai", "milk": "doodh", "love": "pyaar", "heart": "dil",
    "name": "naam", "day": "din", "night": "raat", "morning": "subah", "time": "samay",
    "very": "bahut", "good": "accha", "bad": "bura", "please": "please", "now": "abhi",
    "today": "aaj", "tomorrow": "kal", "people": "log", "children": "bacche", "woman": "aurat",
    "man": "aadmi", "write": "likho", "read": "padho", "tell": "batao", "give": "do",
}
HI_TO_EN: dict[str, str] = {}
for _en, _hi in EN_TO_HI.items():
    HI_TO_EN.setdefault(_hi, _en)

# A "word" is a run of Latin letters or a run of Devanagari (block U+0900–U+097F,
# which includes the matras and virama that \w would split a word on).
_WORD_RE = re.compile(r"[A-Za-z]+|[ऀ-ॿ]+")
_TOKEN_RE = re.compile(r"[A-Za-z]+|[ऀ-ॿ]+|\d+|\s+|[^\sA-Za-zऀ-ॿ\d]")


# ---------------------------------------------------------------------------
# Language tagging and the Code-Mixing Index (Das & Gambäck 2014)
# ---------------------------------------------------------------------------


def _is_devanagari(token: str) -> bool:
    return any("DEVANAGARI" in unicodedata.name(c, "") for c in token)


def tag_token(token: str) -> str:
    """Tag a token ``hi`` / ``en`` / ``univ`` (language-independent).

    Heuristic: Devanagari or a romanized-Hindi lexicon hit → ``hi``; an
    alphabetic Latin token → ``en``; digits / punctuation / whitespace →
    ``univ``. The lexicon is lowercased and strips simple English plural ``s``.
    """
    if _is_devanagari(token):
        return "hi"
    if not _WORD_RE.fullmatch(token):
        return "univ"
    low = token.lower()
    if low in HI_AMBIGUOUS:
        return "en"  # English/Hindi homograph: default to English
    if low in HI_WORDS:
        return "hi"
    return "en"


@dataclass(frozen=True)
class CMIResult:
    """Code-mixing summary for one text."""

    cmi: float  # 0 (monolingual) .. 50 (evenly two-language mixed)
    n_tokens: int
    n_lang_tokens: int  # excludes language-independent tokens
    counts: dict[str, int]
    matrix_language: str  # the dominant language
    is_code_mixed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "cmi": self.cmi, "n_tokens": self.n_tokens, "n_lang_tokens": self.n_lang_tokens,
            "counts": dict(self.counts), "matrix_language": self.matrix_language,
            "is_code_mixed": self.is_code_mixed,
        }


def cmi(text: str) -> CMIResult:
    """Code-Mixing Index (Das & Gambäck 2014, utterance level).

    ``CMI = 100 · (1 − max_L w_L / (N − u))`` where ``N`` is the token count,
    ``u`` the language-independent tokens, and ``max_L w_L`` the count of the
    most frequent language. ``0`` when there are no language tokens or only one
    language is present.
    """
    counts = {"hi": 0, "en": 0, "univ": 0}
    for tok in _WORD_RE.findall(text):
        counts[tag_token(tok)] += 1
    # Also count digit / punctuation groups as language-independent for N.
    univ_extra = len(re.findall(r"\d+|[^\w\s]", text))
    n_lang = counts["hi"] + counts["en"]
    n_tokens = n_lang + counts["univ"] + univ_extra
    if n_lang == 0:
        value = 0.0
        matrix = "none"
    else:
        max_lang = max(counts["hi"], counts["en"])
        matrix = "hi" if counts["hi"] >= counts["en"] else "en"
        value = 100.0 * (1.0 - max_lang / n_lang)
    return CMIResult(
        cmi=round(value, 2), n_tokens=n_tokens, n_lang_tokens=n_lang,
        counts={"hi": counts["hi"], "en": counts["en"],
                "univ": counts["univ"] + univ_extra},
        matrix_language=matrix, is_code_mixed=counts["hi"] > 0 and counts["en"] > 0,
    )


# ---------------------------------------------------------------------------
# Spelling / transliteration variants of a single romanized token
# ---------------------------------------------------------------------------

# Reversible-ish transliteration swaps real users make. Each is (pattern, repl).
_TRANSLIT_SWAPS: list[tuple[str, str]] = [
    ("v", "w"), ("w", "v"), ("ph", "f"), ("f", "ph"), ("z", "j"),
    ("ee", "i"), ("oo", "u"), ("aa", "a"), ("ck", "k"), ("sh", "s"),
]
# SMS shorthand: drop short vowels and collapse known long forms.
_SHORTHAND: dict[str, str] = {
    "nahi": "nhi", "nahin": "nhi", "hai": "h", "hain": "h", "kar": "kr", "karo": "kro",
    "karna": "krna", "bata": "bta", "batao": "btao", "kya": "ky", "kyun": "q", "kyon": "q",
    "please": "plz", "matlab": "mtlb", "message": "msg", "tumhara": "tmhara", "haan": "haa",
    "theek": "thik", "accha": "acha", "bahut": "bhot", "zyada": "zada", "aur": "or",
}
# Spelling variants for a few high-frequency words that have no single spelling.
_SPELLING_SETS: list[frozenset[str]] = [
    frozenset({"nahi", "nahin", "nhi", "nahee"}),
    frozenset({"accha", "acha", "achha"}),
    frozenset({"kyun", "kyon", "kyu", "q"}),
    frozenset({"theek", "thik", "thek"}),
    frozenset({"bahut", "bohot", "bhot", "bahot"}),
    frozenset({"hai", "hae", "h"}),
    frozenset({"kaise", "kaisay", "kese"}),
    frozenset({"paani", "pani"}),
    frozenset({"doodh", "dudh"}),
]
_SPELLING_MAP: dict[str, frozenset[str]] = {w: s for s in _SPELLING_SETS for w in s}


def _geminate_variants(word: str) -> set[str]:
    """Collapse / expand doubled consonants: ``accha`` ↔ ``acha``."""
    out: set[str] = set()
    collapsed = re.sub(r"([bcdfghjklmnpqrstvwxyz])\1", r"\1", word)
    if collapsed != word:
        out.add(collapsed)
    return out


def romanization_variants(word: str, max_variants: int = 8) -> list[str]:
    """Plausible alternative spellings of one romanized token (lowercased).

    Union of: curated spelling sets, single transliteration swaps, geminate
    collapse, and SMS shorthand. The input is always included, first.
    """
    low = word.lower()
    variants: list[str] = [low]
    seen = {low}

    def add(w: str) -> None:
        if w and w != low and w not in seen:
            seen.add(w)
            variants.append(w)

    for w in _SPELLING_MAP.get(low, frozenset()):
        add(w)
    for pat, repl in _TRANSLIT_SWAPS:
        if pat in low:
            add(low.replace(pat, repl, 1))
    for w in _geminate_variants(low):
        add(w)
    if low in _SHORTHAND:
        add(_SHORTHAND[low])
    return variants[:max_variants]


def _match_case(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src[:1].isupper():
        return repl.capitalize()
    return repl


# ---------------------------------------------------------------------------
# Natural variation as a BoN augmenter
# ---------------------------------------------------------------------------


@dataclass
class HinglishVariationConfig:
    """Per-transform probabilities for the natural-variation augmenter."""

    p_spelling: float = 0.5      # swap a romanized-Hindi word for another spelling
    p_translit: float = 0.3      # apply a transliteration swap (w/v, ee/i, ...)
    p_shorthand: float = 0.3     # SMS shorthand (vowel dropping)
    p_codemix_swap: float = 0.2  # swap an en content word to hing or vice versa
    p_drop_punct: float = 0.15   # drop a trailing '?'/'.' the way casual typing does
    max_edits_per_token: int = 1


@dataclass
class HinglishAugmenter:
    """Sample natural-looking Hinglish rewrites of a prompt (deterministic per seed).

    Unlike synthetic BoN noise, every rewrite is a spelling a real user might
    plausibly type, so comprehension is preserved — which is exactly what makes
    it a fair *natural* best-of-N distribution (C2) and, applied to benign
    prompts, a test of orthographic over-refusal (H2b).
    """

    config: HinglishVariationConfig = field(default_factory=HinglishVariationConfig)

    def _rewrite_token(self, token: str, rng: random.Random) -> str:
        if not _WORD_RE.fullmatch(token):
            return token
        low = token.lower()
        tag = tag_token(token)
        edited = token
        # code-mix swap first (changes the word identity)
        if rng.random() < self.config.p_codemix_swap:
            if tag == "en" and low in EN_TO_HI:
                return _match_case(token, EN_TO_HI[low])
            if tag == "hi" and low in HI_TO_EN:
                return _match_case(token, HI_TO_EN[low])
        if tag != "hi":
            return edited
        if rng.random() < self.config.p_spelling and low in _SPELLING_MAP:
            choices = [w for w in _SPELLING_MAP[low] if w != low]
            if choices:
                edited = _match_case(token, rng.choice(choices))
                return edited
        if rng.random() < self.config.p_shorthand and low in _SHORTHAND:
            return _match_case(token, _SHORTHAND[low])
        if rng.random() < self.config.p_translit:
            applicable = [(p, r) for p, r in _TRANSLIT_SWAPS if p in low]
            if applicable:
                p, r = rng.choice(applicable)
                return _match_case(token, low.replace(p, r, 1))
        return edited

    def rewrite(self, text: str, rng: random.Random) -> str:
        parts = _TOKEN_RE.findall(text)
        out = [self._rewrite_token(p, rng) for p in parts]
        s = "".join(out)
        if rng.random() < self.config.p_drop_punct:
            s = re.sub(r"[?.]+\s*$", "", s)
        return s

    def variants(self, text: str, n: int, seed: int = 0) -> list[str]:
        """Up to ``n`` distinct natural rewrites of ``text`` (the original first).

        Deterministic in ``(text, seed)``. Falls short of ``n`` only when the
        text admits fewer distinct rewrites than requested.
        """
        rng = random.Random(f"{seed}|{text}")
        out: list[str] = [text]
        seen = {text}
        attempts = 0
        cap = max(n * 20, 50)
        while len(out) < n and attempts < cap:
            attempts += 1
            cand = self.rewrite(text, rng)
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
        return out


def natural_bon_variants(text: str, n: int, seed: int = 0,
                         config: HinglishVariationConfig | None = None) -> list[str]:
    """Convenience: natural Hinglish BoN variants of ``text`` (original first)."""
    return HinglishAugmenter(config or HinglishVariationConfig()).variants(text, n, seed)
