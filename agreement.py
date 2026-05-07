import krippendorff
import json
import re
import math
import os
import statistics as stat
from pathlib import Path
from collections import defaultdict

# TODO: Cite https://pypi.org/project/krippendorff/ for the implementation in the paper

# ---------------------------------------------------------------------------
# Constants – populate before running
# ---------------------------------------------------------------------------
all_frames = ['accountability', 'bias', 'copyright', 'economy', 'energy', 'environment', 'education', 'healthcare', 'impersonation', 'info', 'intelligence', 'jobs', 'misinformation', 'privacy', 'productivity', 'companions', 'safety', 'scamming', 'security', 'media', 'integration', 'user', 'weapons', 'general', 'other']

# ---------------------------------------------------------------------------
# Core helpers (unchanged from original)
# ---------------------------------------------------------------------------

def tokenize_sentences(text: str, language: str) -> list[tuple[str, int, int]]:
    """Tokenize text into (sentence, start, end) tuples with character offsets."""
    abbreviations = {
        "Mr.", "Mrs.", "Ms.", "Dr.", "U.S.", "U.K.",
        "e.g.", "i.e.", "Sen.", "A.I.", "vs.", "N.F.L.",
    }

    if language == "chinese":
        pattern = r"[。！？；][\"”’]?"
    else:
        pattern = r"[.!?][\"”’')\]]?\s+"

    sentences = []

    # Title becomes its own sentence(s)
    title = text.split("\n", 1)[0]
    last_end = 0
    for match in re.finditer(pattern, title):
        start, end = last_end, match.end()
        sent = text[start:end].strip()
        if sent and sent.split(" ")[-1] not in abbreviations:
            sentences.append((sent, start, match.start()))
            last_end = end
    if last_end < len(title):
        sentences.append((text[last_end:len(title)], last_end, len(title)))

    last_end = len(title) + 1
    for match in re.finditer(pattern, text):
        start, end = last_end, match.end()
        sent = text[start:end].strip()
        if sent and sent.split(" ")[-1] not in abbreviations:
            sentences.append((sent, start, match.start()))
            last_end = end
    if last_end < len(text) and text[last_end:].strip():
        sentences.append((text[last_end:].strip(), last_end, len(text)))

    return sentences


def spans_overlap(start1: int, end1: int, start2: int, end2: int) -> bool:
    return start1 < end2 and start2 < end1


def collect_spans(article: dict) -> list[tuple[int, int, str, str]]:
    """Return list of (start, end, category, sentiment_val) for every edit."""
    spans = []
    for edit in article.get("edits", []):
        annotation = edit.get("annotation")
        if not annotation:
            raise ValueError(f"No annotation on edit in article")
        sentiment = annotation.get("sentiment")
        if not sentiment:
            raise Exception("No sentiment")
        sentiment_val = sentiment.get("val")
        if not sentiment_val:
            raise Exception("No sentiment")
        start, end = edit["input_idx"][0]
        spans.append((start, end, edit.get("category"), sentiment_val))
    return spans


def is_peoples_daily_skip(language: str, source_name: str, idx: int) -> bool:
    """Replicate the original skip logic for known-bad Peoples Daily articles."""
    return (
        language == "chinese"
        and source_name == "Peoples Daily"
        and idx in {1, 3, 8}
    )


# ---------------------------------------------------------------------------
# Sentence-level unit builder
# ---------------------------------------------------------------------------

def build_sentence_units(
    articles1: list[dict],
    articles2: list[dict],
    language: str,
    source_name: str,
) -> list[dict]:
    """
    For every sentence in every article, produce one unit dict:
        {
            "annotated_1": "True"/"False",
            "annotated_2": "True"/"False",
            "sentiment_1": "pos"/"neg"/"both"/"none",
            "sentiment_2": "pos"/"neg"/"both"/"none",
            "frames_1": set[str],   # frames assigned by annotator 1
            "frames_2": set[str],   # frames assigned by annotator 2
        }
    """
    assert len(articles1) == len(articles2), "Article lists must be the same length"

    units = []
    print(f"Source Name: {source_name}")
    for idx, (article1, article2) in enumerate(zip(articles1, articles2)):
        if is_peoples_daily_skip(language, source_name, idx):
            continue

        assert article1["source"] == article2["source"], (
            f"Source mismatch at index {idx} in {source_name}"
        )

        spans1 = collect_spans(article1)
        spans2 = collect_spans(article2)
        sentences = tokenize_sentences(article1["source"], language)

        for _, sent_start, sent_end in sentences:
            # Collect sentiments and frames for each annotator on this sentence
            sents1, frames1 = set(), set()
            sents2, frames2 = set(), set()

            for s_start, s_end, frame, sentiment in spans1:
                if spans_overlap(sent_start, sent_end, s_start, s_end):
                    sents1.add(sentiment)
                    if frame:
                        frames1.add(frame)

            for s_start, s_end, frame, sentiment in spans2:
                if spans_overlap(sent_start, sent_end, s_start, s_end):
                    sents2.add(sentiment)
                    if frame:
                        frames2.add(frame)

            annotated_1 = len(sents1) > 0
            annotated_2 = len(sents2) > 0

            # Collapse sentiment set → single label (mirrors article-level logic)
            def sentiment_label(s: set) -> str:
                if "pos" in s and "neg" in s:
                    return "both"
                if "pos" in s:
                    return "pos"
                if "neg" in s:
                    return "neg"
                return "none"

            unit = {
                "annotated_1": str(annotated_1),
                "annotated_2": str(annotated_2),
                "sentiment_1": sentiment_label(sents1),
                "sentiment_2": sentiment_label(sents2),
                "frames_1": frames1,
                "frames_2": frames2,
            }
            units.append(unit)

    return units


# ---------------------------------------------------------------------------
# Alpha calculators
# ---------------------------------------------------------------------------

def calc_k_alpha(reliability_data: list[list], value_domain: list[str]) -> float:
    """
    Wrapper around krippendorff.alpha.
    reliability_data: rows = coders, columns = units.
    Returns None on ValueError.
    """
    try:
        return krippendorff.alpha(
            reliability_data=reliability_data,
            value_domain=value_domain,
            level_of_measurement="nominal",
        )
    except ValueError:
        raise Exception("HIT VALUE ERROR, returning None")


def calc_annotation_alpha(units: list[dict]) -> float:
    """
    Dimension 1: was the sentence annotated at all?
    Every sentence is included; value domain is True/False.
    """
    row1 = [u["annotated_1"] for u in units]
    row2 = [u["annotated_2"] for u in units]
    return calc_k_alpha([row1, row2], value_domain=["True", "False"])


def calc_sentiment_alpha(units: list[dict]) -> float:
    """
    Dimension 2: sentiment of annotated sentences.
    """
    row1, row2 = [], []
    for u in units:
        row1.append(u["sentiment_1"])
        row2.append(u["sentiment_2"])
    return calc_k_alpha([row1, row2], value_domain=["pos", "neg", "both"])


def calc_frame_alpha(units: list[dict], frames: list[str]) -> dict[str, float]:
    """
    Dimension 3: per-frame binary alpha, aggregated to a mean.
    Returns a dict of {frame: alpha} (NaN frames excluded).
    """
    alpha_by_frame: dict[str, float] = {}
    for frame in frames:
        row1 = [str(frame in u["frames_1"]) for u in units]
        row2 = [str(frame in u["frames_2"]) for u in units]
        alpha = calc_k_alpha([row1, row2], value_domain=["True", "False"])
        if alpha is not None and not math.isnan(alpha):
            alpha_by_frame[frame] = alpha
    return alpha_by_frame


def mean_alpha(alpha_dict: dict[str, float]) -> float | None:
    values = list(alpha_dict.values())
    return stat.mean(values) if values else None


# ---------------------------------------------------------------------------
# Data loading (unchanged logic from original)
# ---------------------------------------------------------------------------

def load_source(source_dir: Path) -> tuple[list, list]:
    """
    Load annotator 1 and annotator 2 files from a source directory.
    Applies the original article-range slicing logic.
    Returns (annotator_1_articles, annotator_2_articles).
    """
    annotator_1_file = annotator_2_file = None
    for f in source_dir.iterdir():
        if f.name[0] == "1":
            annotator_1_file = f
        elif f.name[0] == "2":
            annotator_2_file = f

    if annotator_1_file is None or annotator_2_file is None:
        raise FileNotFoundError(f"Missing annotation file(s) in {source_dir}")

    with open(annotator_1_file) as f:
        data1 = json.load(f)
    with open(annotator_2_file) as f:
        data2 = json.load(f)

    assert len(data1) == len(data2)
    return data1, data2


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_language_alphas(language_dir: Path, language: str) -> None:
    """
    Iterates over all source directories in language_dir, builds sentence-level
    units, and prints the three aggregate Krippendorff alpha values.
    """
    all_units: list[dict] = []
    annotated_units = []

    for source_dir in sorted(language_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        data1, data2 = load_source(source_dir)
        units = build_sentence_units(data1, data2, language, source_dir.name)
        all_units.extend(units)
        for unit in units:
            if (unit["annotated_1"] == "True" and unit["annotated_2"] == "True"):
                annotated_units.append(unit)

    print(annotated_units)
    annotation_alpha = calc_annotation_alpha(all_units)
    sentiment_alpha  = calc_sentiment_alpha(annotated_units)
    frame_alpha_dict = calc_frame_alpha(annotated_units, all_frames)
    print(frame_alpha_dict)
    frame_alpha_mean = mean_alpha(frame_alpha_dict)
    print(frame_alpha_mean)

    print("=" * 50)
    print(f"Language: {language}")
    print(f"  Total sentence units : {len(all_units)}")
    print(f"  Annotation   : {annotation_alpha:.4f}")
    print(f"  Sentiment    : {sentiment_alpha:.4f}")
    print(f"  Frame      : {frame_alpha_mean:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    base_path = "/srv/nlprx-lab/share5/jyoung369/ai-framing/news-dataloader/adjudication"
    language = "english"
    language_dir = Path(f"{base_path}/{language}_annotations")

    compute_language_alphas(language_dir, language)