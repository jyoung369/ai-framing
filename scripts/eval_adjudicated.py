"""
Evaluate frame/sentiment/identify predictions against adjudicated gold labels.

Usage:
  python scripts/eval_adjudicated.py --task fc       --lang en --predictions results/fc_predictions_myrun.jsonl
  python scripts/eval_adjudicated.py --task sc       --lang zh --predictions results/sc_predictions_myrun.jsonl
  python scripts/eval_adjudicated.py --task identify --lang ru --predictions results/identify_predictions_myrun.jsonl
"""

import argparse
import json
import re
from pathlib import Path

_BASE     = Path(__file__).resolve().parent.parent
_LANG_DIR = {"en": "english", "zh": "chinese", "ru": "russian"}
_TRAIL    = re.compile(r'[。！？；\s""\'\"]+$')


def norm(t: str) -> str:
    return _TRAIL.sub("", t.strip())


# ── GT loaders ─────────────────────────────────────────────────────────────────

def load_fc_sc_gt(lang: str) -> dict:
    """Returns {(source, norm_span): {"frames", "sentiments", "primary_frame", "primary_sent"}}"""
    adj_dir = _BASE / "dataset" / "adjudication_data" / _LANG_DIR[lang]
    gt = {}
    for source_dir in sorted(adj_dir.iterdir()):
        adj_file = source_dir / "adjudication.json"
        if not adj_file.exists():
            continue
        for rec in json.loads(adj_file.read_text(encoding="utf-8")):
            for span_rec in rec.get("spans", []):
                frames = [a["frame"]     for a in span_rec["annotations"] if a.get("frame")]
                sents  = [a["sentiment"] for a in span_rec["annotations"]
                          if a.get("sentiment") in ("pos", "neg")]
                key = (source_dir.name, norm(span_rec["span"]))
                if key not in gt:
                    gt[key] = {
                        "frames":        set(frames),
                        "sentiments":    set(sents),
                        "primary_frame": frames[0] if frames else "none",
                        "primary_sent":  sents[0]  if sents  else "none",
                    }
                else:
                    gt[key]["frames"].update(frames)
                    gt[key]["sentiments"].update(sents)
    return gt


def load_identify_gt(lang: str) -> dict:
    """Returns {norm_sentence: bool} from test_data/{lang}_identify_test.jsonl."""
    path = _BASE / "test_data" / f"{lang}_identify_test.jsonl"
    gt = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gt[norm(rec["sentence_text"])] = bool(rec["label"])
    return gt


# ── Metrics ────────────────────────────────────────────────────────────────────

def micro_f1(y_true, y_pred, labels) -> float:
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        for lbl in labels:
            if t == lbl and p == lbl:   tp += 1
            elif t != lbl and p == lbl: fp += 1
            elif t == lbl and p != lbl: fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def best_micro_f1(y_true, y_pred, confs, labels):
    best_f1, best_thresh = 0.0, 0
    for thresh in range(101):
        yp = [p if c >= thresh else "none" for p, c in zip(y_pred, confs)]
        f1 = micro_f1(y_true, yp, labels)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    return round(best_f1, 4), best_thresh


# ── Confidence extraction ──────────────────────────────────────────────────────

def _fc_confidence(parsed: dict) -> float:
    if not parsed:
        return 100.0
    if "confidence" in parsed:
        return float(parsed["confidence"])
    scores = [v for v in parsed.values() if isinstance(v, (int, float))]
    return float(max(scores)) if scores else 100.0


def _sc_confidence(parsed: dict) -> float:
    if not parsed:
        return 100.0
    if "confidence" in parsed:
        return float(parsed["confidence"])
    pos, neg = parsed.get("Positive"), parsed.get("Negative")
    if pos is not None and neg is not None:
        return float(max(pos, neg))
    return 100.0


# ── Task evaluators ────────────────────────────────────────────────────────────

def eval_fc(pred_path: Path, lang: str) -> dict:
    gt = load_fc_sc_gt(lang)
    y_true, y_pred, confs = [], [], []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            rec      = json.loads(line)
            gt_entry = gt.get((rec.get("source", ""), norm(rec.get("span_text", ""))))
            if gt_entry is None:
                continue
            pred   = rec.get("predicted_top_frame") or "none"
            parsed = rec.get("parsed_output") or {}
            gt_lbl = pred if pred in gt_entry["frames"] else gt_entry["primary_frame"]
            y_true.append(gt_lbl)
            y_pred.append(pred)
            confs.append(_fc_confidence(parsed))
    labels  = sorted(set(y_true) - {"none"})
    base_f1 = round(micro_f1(y_true, y_pred, labels), 4)
    best_f1, best_thresh = best_micro_f1(y_true, y_pred, confs, labels)
    return {"micro_f1": base_f1, "best_micro_f1": best_f1,
            "best_threshold": best_thresh, "n": len(y_true)}


def eval_sc(pred_path: Path, lang: str) -> dict:
    gt = load_fc_sc_gt(lang)
    y_true, y_pred, confs = [], [], []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            rec      = json.loads(line)
            gt_entry = gt.get((rec.get("source", ""), norm(rec.get("span_text", ""))))
            if gt_entry is None:
                continue
            pred   = rec.get("predicted_sentiment") or "none"
            if pred not in ("pos", "neg"):
                pred = "none"
            parsed = rec.get("parsed_output") or {}
            p_sent = gt_entry["primary_sent"]
            gt_lbl = pred if pred in gt_entry["sentiments"] else (
                p_sent if p_sent in ("pos", "neg") else "none"
            )
            y_true.append(gt_lbl)
            y_pred.append(pred)
            confs.append(_sc_confidence(parsed))
    base_f1 = round(micro_f1(y_true, y_pred, ["pos", "neg"]), 4)
    best_f1, best_thresh = best_micro_f1(y_true, y_pred, confs, ["pos", "neg"])
    return {"micro_f1": base_f1, "best_micro_f1": best_f1,
            "best_threshold": best_thresh, "n": len(y_true)}


def eval_identify(pred_path: Path, lang: str) -> dict:
    gt = load_identify_gt(lang)
    y_true, y_pred, confs = [], [], []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            rec   = json.loads(line)
            label = gt.get(norm(rec.get("sentence_text", "")))
            if label is None:
                continue
            pred = rec.get("predicted_label", False)
            conf = float(rec.get("confidence", 100))
            y_true.append("true" if label else "false")
            y_pred.append("true" if pred  else "false")
            confs.append(conf)
    base_f1 = round(micro_f1(y_true, y_pred, ["true"]), 4)
    best_f1, best_thresh = best_micro_f1(y_true, y_pred, confs, ["true"])
    return {"micro_f1": base_f1, "best_micro_f1": best_f1,
            "best_threshold": best_thresh, "n": len(y_true)}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions against adjudicated GT.")
    parser.add_argument("--task",        required=True, choices=["fc", "sc", "identify"])
    parser.add_argument("--lang",        required=True, choices=["en", "zh", "ru"])
    parser.add_argument("--predictions", required=True, type=Path)
    args = parser.parse_args()

    print(f"Task: {args.task.upper()}  Lang: {args.lang.upper()}")
    print(f"Predictions: {args.predictions}\n")

    if args.task == "fc":
        metrics = eval_fc(args.predictions, args.lang)
    elif args.task == "sc":
        metrics = eval_sc(args.predictions, args.lang)
    else:
        metrics = eval_identify(args.predictions, args.lang)

    print(f"  Micro F1      : {metrics['micro_f1']:.4f}  (n={metrics['n']})")
    print(f"  Best Micro F1 : {metrics['best_micro_f1']:.4f}  (threshold >= {metrics['best_threshold']})")

    out_path = args.predictions.with_suffix(".eval.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
