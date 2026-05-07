"""
Prompt builder for the sentiment_confidence_2class_{zh,en,ru} task.

Template: {EXAMPLE_POS_N} / {EXAMPLE_NEG_N} placeholders (N = 1..n_per_class)
Pool:     {lang}_frame_pool_ha.json  — entries across all frames, each with a
          "sentiment" field ("pos" or "neg"). Entries from all frames are pooled
          together so examples span diverse topics.

Public API
----------
build_sentiment_classification_prompt(
    span_text,
    lang="en",
    tokenizer=None,
    template_path=None,
    pool_path=None,
    n_per_class=5,
    seed=42,
) -> str
"""

import json
import re
import random
from pathlib import Path

_BASE     = Path(__file__).resolve().parent.parent
_TMPL_DIR = _BASE / "prompts/sentiment"
_POOL_DIR = _BASE / "test_data"

_DEFAULT_TEMPLATES = {
    "zh": _TMPL_DIR / "sentiment_confidence_2class_zh.txt",
    "en": _TMPL_DIR / "sentiment_confidence_2class_en.txt",
    "ru": _TMPL_DIR / "sentiment_confidence_2class_ru.txt",
}
_DEFAULT_POOLS = {
    "zh": _POOL_DIR / "zh_frame_pool_ha.json",
    "en": _POOL_DIR / "en_frame_pool_ha.json",
    "ru": _POOL_DIR / "ru_frame_pool_ha.json",
}

_SYSTEM_RE = re.compile(r"<start_of_turn>system\n(.*?)<end_of_turn>", re.DOTALL)
_USER_RE   = re.compile(r"<start_of_turn>user\n(.*?)<end_of_turn>",   re.DOTALL)

_filled_cache: dict = {}


def _collect_by_sentiment(pool: dict) -> dict:
    """Return {"pos": [span_text, ...], "neg": [...]} from a frame-keyed pool."""
    by_sent: dict = {"pos": [], "neg": []}
    for entries in pool.values():
        for entry in entries:
            s = entry.get("sentiment")
            if s in by_sent:
                by_sent[s].append(entry["sentence_text"])
    return by_sent


def _fill_examples(tmpl: str, pool: dict, n_per_class: int, rng: random.Random) -> str:
    by_sent = _collect_by_sentiment(pool)
    for label in ("POS", "NEG"):
        key    = label.lower()
        pool_l = by_sent.get(key, [])
        k      = min(n_per_class, len(pool_l))
        sampled = rng.sample(pool_l, k) if k > 0 else []
        for i in range(1, n_per_class + 1):
            ph = f"{{EXAMPLE_{label}_{i}}}"
            if i <= len(sampled):
                tmpl = tmpl.replace(ph, sampled[i - 1].strip())
            else:
                tmpl = re.sub(r'\n   - \{EXAMPLE_' + label + r'_' + str(i) + r'\}', '', tmpl)
    return tmpl


def _apply_chat_template(system: str, user: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _get_filled_template(lang, template_path, pool_path, n_per_class, seed) -> str:
    tmpl_path = Path(template_path) if template_path else _DEFAULT_TEMPLATES[lang]
    pool_path = Path(pool_path)     if pool_path     else _DEFAULT_POOLS[lang]
    cache_key = (str(tmpl_path), str(pool_path), n_per_class, seed)

    if cache_key not in _filled_cache:
        tmpl = tmpl_path.read_text(encoding="utf-8")
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
        rng  = random.Random(seed)
        tmpl = _fill_examples(tmpl, pool, n_per_class, rng)
        _filled_cache[cache_key] = tmpl

    return _filled_cache[cache_key]


def build_sentiment_classification_prompt(
    span_text: str,
    lang: str = "en",
    tokenizer=None,
    template_path=None,
    pool_path=None,
    n_per_class: int = 5,
    seed: int = 42,
) -> str:
    filled = _get_filled_template(lang, template_path, pool_path, n_per_class, seed)

    sys_m  = _SYSTEM_RE.search(filled)
    user_m = _USER_RE.search(filled)

    if sys_m and user_m:
        system    = sys_m.group(1).strip()
        user_tmpl = user_m.group(1).strip()
        user      = user_tmpl.replace("{INSERT_SPAN_HERE}", span_text)
    else:
        result = filled.replace("{INSERT_SPAN_HERE}", span_text)
        return result

    if tokenizer is None:
        return (
            f"<start_of_turn>system\n{system}<end_of_turn>\n"
            f"<start_of_turn>user\n{user}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
    return _apply_chat_template(system, user, tokenizer)


# ── CLI quick-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    tests = {
        "en": "AI has made doctors faster at diagnosing cancer.",
        "zh": "人工智能正在帮助医生更快速准确地诊断疾病。",
        "ru": "ИИ помогает врачам быстрее ставить диагнозы.",
    }
    for lang, span in tests.items():
        p = build_sentiment_classification_prompt(span, lang=lang, n_per_class=5)
        unfilled = re.findall(r'\{EXAMPLE_(?:POS|NEG)_\d+\}', p)
        print(f"\n[{lang}] {len(p)} chars | unfilled: {unfilled}")
        idx = p.find("Positive") if lang == "en" else p.find("Positive（")
        if idx < 0:
            idx = p.find("Positive")
        print(p[idx:idx+400] if idx >= 0 else p[:400])
    print("\nBuilder OK.")
