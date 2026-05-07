"""
Prompt builder for AI-sentiment sentence identification (binary True/False).

Pool format:
  {
    "true":  [{"sentence_text": "...", "label": true},  ...],
    "false": [{"sentence_text": "...", "label": false}, ...]
  }

Template placeholders:
  {EXAMPLES_BLOCK}       → n True + n False few-shot examples, interleaved
  {INSERT_SENTENCE_HERE} → the single test sentence

Public API
----------
build_identify_prompt(
    sentence,          # str
    n_examples=5,      # n per class (total = 2*n)
    tokenizer=None,    # if given, applies model-specific chat template
    template_path=None,
    pool_path=None,
)
"""

import json
import random
import re
from pathlib import Path

_SYSTEM_RE = re.compile(r"<start_of_turn>system\n(.*?)<end_of_turn>", re.DOTALL)
_USER_RE   = re.compile(r"<start_of_turn>user\n(.*?)<end_of_turn>",   re.DOTALL)


def _extract_system_user(tmpl: str):
    sys_m  = _SYSTEM_RE.search(tmpl)
    user_m = _USER_RE.search(tmpl)
    system = sys_m.group(1).strip()  if sys_m  else ""
    user   = user_m.group(1).strip() if user_m else ""
    return system, user


def _apply_chat_template(system: str, user: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _format_examples_block(true_examples: list, false_examples: list,
                           use_confidence: bool = True,
                           use_prob_dist: bool = False) -> str:
    """
    Interleave true and false examples into a numbered block.
    Each entry: sentence_text (str), label (bool).
    use_prob_dist: strategy 4 — output {"True": X, "False": Y, "is_target": bool}
    """
    pairs = list(zip(true_examples, false_examples))
    ordered = []
    for t, f in pairs:
        ordered.append((t["sentence_text"], True))
        ordered.append((f["sentence_text"], False))

    lines = []
    for i, (sent, label) in enumerate(ordered, start=1):
        if use_prob_dist:
            true_score  = 90 if label else 10
            false_score = 10 if label else 90
            out = json.dumps({"True": true_score, "False": false_score, "is_target": label},
                             ensure_ascii=False)
        elif use_confidence:
            out = json.dumps({"is_target": label, "confidence": 90}, ensure_ascii=False)
        else:
            out = json.dumps({"is_target": label}, ensure_ascii=False)
        lines.append(f"### Example {i}\nSentence: {sent}\nOutput: {out}")
    return "\n\n".join(lines)


def build_identify_prompt(
    sentence: str,
    n_examples: int = 5,
    tokenizer=None,
    template_path=None,
    pool_path=None,
    nothink: bool = False,
    use_confidence: bool = True,
    use_prob_dist: bool = False,
) -> str:
    """
    Build a single-sentence identification prompt.

    Args:
        sentence:      test sentence string
        n_examples:    number of examples per class (total = 2 * n_examples)
        tokenizer:     if provided, applies model-specific chat template
        template_path: path to the prompt template file
        pool_path:     path to the pool JSON file
        nothink:       if True, appends /no_think to user content (Qwen3)
        use_prob_dist: strategy 4 — examples show {"True": X, "False": Y, "is_target": bool}
    Returns:
        Formatted prompt string
    """
    with open(template_path, encoding="utf-8") as f:
        tmpl = f.read()

    with open(pool_path, encoding="utf-8") as f:
        pool = json.load(f)

    true_pool  = pool.get("true",  [])
    false_pool = pool.get("false", [])

    n_true  = min(n_examples, len(true_pool))
    n_false = min(n_examples, len(false_pool))
    n       = min(n_true, n_false)

    true_sample  = random.sample(true_pool,  n)
    false_sample = random.sample(false_pool, n)

    examples_block = _format_examples_block(true_sample, false_sample,
                                             use_confidence=use_confidence,
                                             use_prob_dist=use_prob_dist)

    tmpl = tmpl.replace("{EXAMPLES_BLOCK}", examples_block)

    sentence_field = sentence + "\n/no_think" if nothink else sentence
    tmpl = tmpl.replace("{INSERT_SENTENCE_HERE}", sentence_field)

    if tokenizer is None:
        return tmpl

    system, user = _extract_system_user(tmpl)
    return _apply_chat_template(system, user, tokenizer)


# ── CLI quick-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    pool_path = (
        Path(__file__).resolve().parent.parent
        / "test_data/zh_identify_pool.json"
    )
    tmpl_path = (
        Path(__file__).resolve().parent.parent
        / "prompts/identify/identify_zh.txt"
    )

    test_sent = "随着数字经济的发展，人工智能越来越多地渗透到社会的方方面面，成为未来产业发展中最重要的力量之一。"
    prompt = build_identify_prompt(
        test_sent, n_examples=5,
        template_path=tmpl_path, pool_path=pool_path,
    )
    print(prompt)
