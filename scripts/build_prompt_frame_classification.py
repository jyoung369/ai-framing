"""
Prompt builder for the frame_confidence_25class_{zh,en,ru}_v2 task.

Template format: {EXAMPLE_{SLUG}_{N}} placeholders are embedded directly
under each frame's definition (N = 1..n_per_frame).

Pool format: {lang}_frame_pool_ha.json
  {frame_name: [{"sentence_text": "...", "output": {...}, "sentiment": "..."}, ...]}

The builder samples n_per_frame sentences per frame, fills every placeholder,
and replaces any remaining unfilled slots (frames with < n entries in the pool)
with an empty string so the model sees a clean dash line that is removed.

Public API
----------
build_frame_classification_prompt(
    span_text,          # str — single sentence to classify
    lang="en",
    tokenizer=None,
    template_path=None,
    pool_path=None,
    n_per_frame=5,
    seed=42,
) -> str
"""

import json
import re
import random
from pathlib import Path

_BASE     = Path(__file__).resolve().parent.parent
_TMPL_DIR = _BASE / "prompts/frame"
_POOL_DIR = _BASE / "test_data"

_DEFAULT_TEMPLATES = {
    "zh": _TMPL_DIR / "frame_confidence_25class_zh.txt",
    "en": _TMPL_DIR / "frame_confidence_25class_en.txt",
    "ru": _TMPL_DIR / "frame_confidence_25class_ru.txt",
}
_DEFAULT_POOLS = {
    "zh": _POOL_DIR / "zh_frame_pool_ha.json",
    "en": _POOL_DIR / "en_frame_pool_ha.json",
    "ru": _POOL_DIR / "ru_frame_pool_ha.json",
}

FRAME_SLUGS = {
    "Energy":                 "ENERGY",
    "Environment":            "ENVIRONMENT",
    "Healthcare":             "HEALTHCARE",
    "User Experience":        "USER_EXPERIENCE",
    "Weapons":                "WEAPONS",
    "Copyright Infringement": "COPYRIGHT_INFRINGEMENT",
    "Education":              "EDUCATION",
    "Information Access":     "INFORMATION_ACCESS",
    "Intelligence":           "INTELLIGENCE",
    "Misinformation":         "MISINFORMATION",
    "Impersonation":          "IMPERSONATION",
    "Privacy":                "PRIVACY",
    "Safety":                 "SAFETY",
    "Scamming":               "SCAMMING",
    "Security":               "SECURITY",
    "Accountability":         "ACCOUNTABILITY",
    "Bias":                   "BIAS",
    "Social Companions":      "SOCIAL_COMPANIONS",
    "Social Media":           "SOCIAL_MEDIA",
    "Societal Integration":   "SOCIETAL_INTEGRATION",
    "Productivity":           "PRODUCTIVITY",
    "Economy":                "ECONOMY",
    "Jobs":                   "JOBS",
    "General":                "GENERAL",
    "Other":                  "OTHER",
}

_SYSTEM_RE = re.compile(r"<start_of_turn>system\n(.*?)<end_of_turn>", re.DOTALL)
_USER_RE   = re.compile(r"<start_of_turn>user\n(.*?)<end_of_turn>",   re.DOTALL)

# Cache: (tmpl_path, pool_path, n_per_frame, seed) → template string with examples filled
_filled_cache: dict = {}


def _fill_examples(tmpl: str, pool: dict, n_per_frame: int, rng: random.Random) -> str:
    """Replace all {EXAMPLE_{SLUG}_{N}} placeholders with sampled sentences."""
    for frame, slug in FRAME_SLUGS.items():
        entries = pool.get(frame, [])
        k = min(n_per_frame, len(entries))
        sampled = rng.sample(entries, k) if k > 0 else []
        for i in range(1, n_per_frame + 1):
            ph = f"{{EXAMPLE_{slug}_{i}}}"
            if i <= len(sampled):
                replacement = sampled[i - 1]["sentence_text"].strip()
            else:
                # Remove the entire bullet line for this placeholder
                tmpl = re.sub(r'\n   - \{EXAMPLE_' + slug + r'_' + str(i) + r'\}', '', tmpl)
                tmpl = re.sub(r'\n    - \{EXAMPLE_' + slug + r'_' + str(i) + r'\}', '', tmpl)
                continue
            tmpl = tmpl.replace(ph, replacement)
    return tmpl


def _apply_chat_template(system: str, user: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _get_filled_template(
    lang: str,
    template_path,
    pool_path,
    n_per_frame: int,
    seed: int,
) -> str:
    tmpl_path = Path(template_path) if template_path else _DEFAULT_TEMPLATES[lang]
    pool_path = Path(pool_path)     if pool_path     else _DEFAULT_POOLS[lang]
    cache_key = (str(tmpl_path), str(pool_path), n_per_frame, seed)

    if cache_key not in _filled_cache:
        tmpl = tmpl_path.read_text(encoding="utf-8")
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
        rng  = random.Random(seed)
        tmpl = _fill_examples(tmpl, pool, n_per_frame, rng)
        _filled_cache[cache_key] = tmpl

    return _filled_cache[cache_key]


def build_frame_classification_prompt(
    span_text: str,
    lang: str = "en",
    tokenizer=None,
    template_path=None,
    pool_path=None,
    n_per_frame: int = 5,
    seed: int = 42,
) -> str:
    """
    Build a per-sentence 25-frame confidence classification prompt.

    Args:
        span_text:     the sentence to classify
        lang:          "zh", "en", or "ru"
        tokenizer:     if provided, applies the model's chat template
        template_path: override default template file
        pool_path:     override default pool file
        n_per_frame:   number of example sentences shown under each frame definition
        seed:          random seed for example sampling (same seed → same examples)
    Returns:
        Formatted prompt string (or tokenized string if tokenizer given)
    """
    filled = _get_filled_template(lang, template_path, pool_path, n_per_frame, seed)

    sys_m  = _SYSTEM_RE.search(filled)
    user_m = _USER_RE.search(filled)

    if sys_m and user_m:
        system    = sys_m.group(1).strip()
        user_tmpl = user_m.group(1).strip()
        user      = user_tmpl.replace("{INSERT_SPAN_HERE}", span_text)
    else:
        # Fallback: raw substitution
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
        "en": "AI can now diagnose skin cancer with accuracy rivaling dermatologists.",
        "zh": "人工智能正在帮助医生更快速准确地诊断疾病。",
        "ru": "ИИ помогает врачам быстрее ставить диагнозы.",
    }
    for lang, span in tests.items():
        p = build_frame_classification_prompt(span, lang=lang, n_per_frame=5)
        unfilled = re.findall(r'\{EXAMPLE_[A-Z_]+_\d+\}', p)
        print(f"\n[{lang}] {len(p)} chars | unfilled placeholders: {unfilled}")
        # Show a snippet around the Healthcare definition
        idx = p.find("Healthcare")
        if idx >= 0:
            print(p[idx:idx+300])
    print("\nBuilder OK.")
