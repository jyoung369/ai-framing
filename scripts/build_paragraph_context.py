"""
Paragraph context helper for Strategy 3 experiments.

Loads paragraph data from {lang}_sentiment_test.jsonl and provides
2-sentence before/after context for a target span or sentence.

Public API
----------
load_paragraphs(lang) -> dict[str, str]
    Returns {id: paragraph_text} from the sentiment test file.

get_context_text(target_text, paragraph_text, lang, n_before=2, n_after=2) -> str
    Returns a formatted string:
        [Context before]
        <sentence>
        ...
        [Target]
        <target_text>
        [Context after]
        <sentence>
        ...
    If context sentences are not found, returns target_text unchanged.
"""

import json
import re
from pathlib import Path

_DEFAULT_BASE = Path(__file__).resolve().parent.parent / "test_data"

_ZH_RE   = re.compile(r'(?<=[。！？；\n])')
_ENRU_RE = re.compile(r'(?<=[.!?])\s+')

_paragraph_cache: dict = {}


def _split_sentences(text: str, lang: str) -> list[str]:
    pattern = _ZH_RE if lang == "zh" else _ENRU_RE
    parts = pattern.split(text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) >= 5]


def load_paragraphs(lang: str, data_dir: Path = None) -> dict[str, str]:
    """Return {id: paragraph_text} for all paragraphs in the sentiment test file.

    data_dir: directory containing {lang}_sentiment_test.jsonl.
              Defaults to test_data/ relative to repo root.
              Pass the parent of --test-data for adjudication experiments.
    """
    base = Path(data_dir) if data_dir is not None else _DEFAULT_BASE
    cache_key = (lang, str(base))
    if cache_key in _paragraph_cache:
        return _paragraph_cache[cache_key]
    path = base / f"{lang}_sentiment_test.jsonl"
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            result[rec["id"]] = rec["paragraph_text"]
    _paragraph_cache[cache_key] = result
    return result


def get_context_text(
    target_text: str,
    paragraph_text: str,
    lang: str,
    n_before: int = 2,
    n_after: int = 2,
) -> str:
    """
    Find target_text in paragraph_text, extract n_before/n_after sentences,
    and return a formatted context string.
    Falls back to target_text alone if target not found in paragraph.
    """
    sentences = _split_sentences(paragraph_text, lang)
    if not sentences:
        return target_text

    # Find the sentence(s) that contain or most overlap with target_text
    target_clean = target_text.strip()
    best_idx = None

    # Exact match first
    for i, s in enumerate(sentences):
        if target_clean in s or s in target_clean:
            best_idx = i
            break

    # Fallback: longest common substring by character overlap
    if best_idx is None:
        best_overlap = 0
        for i, s in enumerate(sentences):
            overlap = sum(1 for c in target_clean if c in s)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

    if best_idx is None:
        return target_text

    before = sentences[max(0, best_idx - n_before): best_idx]
    after  = sentences[best_idx + 1: best_idx + 1 + n_after]

    parts = []
    if before:
        parts.append("[Context before]\n" + "\n".join(before))
    parts.append("[Target]\n" + target_clean)
    if after:
        parts.append("[Context after]\n" + "\n".join(after))

    return "\n\n".join(parts)
